
import re
import shutil
import os
import json
import math
import cv2
import torch
from typing import Tuple, List
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import preprocess_caption
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, contrast=None,octrans=None):
        self.image_paths = image_paths
        self.transform = transform
        self.contrast = contrast
        self.octrans=octrans

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image_pil = Image.open(img_path).convert("RGB")
        # image_cv = cv2.imread(img_path)
        # image_pil = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        if self.contrast is not None:
            img_contrast = ImageEnhance.Contrast(image_pil)
            image_contrast = img_contrast.enhance(self.contrast)
        else:
            image_contrast = image_pil
        if self.transform is not None:
            image_trans, _ = self.transform(image_contrast,target=None)
        else:
            image_trans = image_contrast
        if self.octrans is not None:
            image_trans=self.octrans(image_contrast)

        return image_trans, img_path

    def __len__(self):
        return len(self.image_paths)

def trans():
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform

def totensor():
    transform = T.Compose([
            # 调整尺寸
        T.ToTensor()             # 转为 Tensor
        # T.Normalize(               # 归一化
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])
    return transform
def batch_load_images(image_path_lst, batch_size):
    num_batches = (len(image_path_lst) + batch_size - 1) // batch_size

    batches_image_path = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_path_lst))
        batch_paths = image_path_lst[start_idx:end_idx]

        batches_image_path.append(batch_paths)

    return num_batches, batches_image_path

def predict_batch(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        gpu_id: int = 0
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    '''
    return:
        bboxes_batch: list of tensors of shape (n, 4)
        predicts_batch: list of tensors of shape (n,)
        phrases_batch: list of list of strings of shape (n,)
        n is the number of boxes in one image
    '''
    caption = preprocess_caption(caption=caption)
    if device != torch.device('cpu'):
        model = DataParallel(model,device_ids=[gpu_id])
    model = model.to(device)
    image = images.to(device)
    captions = [caption for _ in range(len(images))]
    with torch.no_grad():
        with autocast():
            outputs = model(image, captions=captions)  # <------- 我对我的用例的所有图像使用相同的提示词
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (num_batch, nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (num_batch, nq, 4)

    # import ipdb; ipdb.set_trace()
    mask = prediction_logits.max(dim=2)[0] > box_threshold  # mask: torch.Size([num_batch, 256])

    bboxes_batch = []
    predicts_batch = []
    phrases_batch = []  # list of lists
    
    tokenizer = model.module.tokenizer
    tokenized = tokenizer(caption)
    for i in range(prediction_logits.shape[0]):
        logits = prediction_logits[i][mask[i]]  # logits.shape = (n, 256)
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit  # logit is a tensor of shape (256,) torch.Size([256])
            in logits  # torch.Size([7, 256])
        ]
        boxes = prediction_boxes[i][mask[i]]  # boxes.shape = (n, 4)
        phrases_batch.append(phrases)
        bboxes_batch.append(boxes)
        predicts_batch.append(logits.max(dim=1)[0])

    return bboxes_batch, predicts_batch, phrases_batch

def predict_batch2(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = images.to(device)

    print(f"Image shape: {image.shape}") # Image shape: torch.Size([num_batch, 3, 800, 1200])
    with torch.no_grad():
        outputs = model(image, captions=[caption for _ in range(len(images))]) # <------- I use the same caption for all the images for my use-case

    print(f'{outputs["pred_logits"].shape}') # torch.Size([num_batch, 900, 256]) 
    print(f'{outputs["pred_boxes"].shape}') # torch.Size([num_batch, 900, 4])
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=2)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases

def move_abnorfile(result, abnormalcrop, crop_dir):
    name = result['name']  # 假设 name 是 result 的第一个元素
    pred_class = result['pred_class']  #
    # 复制提前
    if pred_class == 'abnormal':
        imgfile = os.path.join(crop_dir, os.path.basename(name))
        target_file = os.path.join(abnormalcrop, os.path.basename(imgfile))
        try:
            shutil.move(imgfile, target_file)
        except (IOError, OSError) as e:
            print(f"Error moving file: {e}")



def remove_area_outliers(args,boxes):
    boxes_np = [box.cpu().numpy() for box in boxes]
    areas = [(box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) for box in boxes_np]

    if args.cut_size == 20:
        large_area_indices = [np.where((area > 100000) | (area < 1500))[0] for area in areas]
    elif args.cut_size == 10:
        large_area_indices = [np.where((area > 10000) | (area < 800))[0] for area in areas]
    elif args.cut_size == 5:
        large_area_indices = [np.where(area > 1500)[0] for area in areas]

    for i in range(len(boxes)):
        boxes[i] = np.delete(boxes[i], large_area_indices[i], axis=0)
    return boxes





def cropallecellthread(input_paths, crop_dir):
    """
    多线程裁剪图像中的所有细胞。

    该函数接收一个包含图像路径的列表和一个目标裁剪目录，它会遍历所有图像，
    根据对应的标注文件裁剪出每个细胞，并将裁剪结果保存到目标目录中。

    :param input_paths: 图像路径列表，每个路径指向一个包含细胞的图像。
    :param crop_dir: 裁剪结果的目标保存目录。
    """
    # 遍历输入的图像路径列表
    for input_path in input_paths:
        # 根据图像路径生成对应的标注文件路径
        jsonpath = input_path.replace('patch', 'json').replace('.png', '.json')
        # 检查标注文件是否存在
        if os.path.exists(jsonpath):
            # 打开标注文件
            with open(jsonpath, 'r') as f:
                # 加载标注文件的内容到json对象
                json_data = json.load(f)
            # 获取标注数据中的第一个值，通常是一个包含所有物体标注信息的字典
            first_value = list(json_data.values())[0]
            # 使用线程池执行裁剪操作
            with ThreadPoolExecutor() as executor:
                # 遍历第一个值中的每个物体标注，为每个物体启动一个裁剪任务
                for key, value in first_value.items():
                    # 构建每个物体裁剪结果的保存路径
                    single_file_output_path = f"{crop_dir}/{key}"
                    # 提取物体的边界框信息
                    bndbox = value["bndbox"]
                    # 提交裁剪任务给线程池执行
                    executor.submit(crop_image, input_path, single_file_output_path, bndbox)

def cropallecellthreadV2(input_paths, crop_dir):
    with ThreadPoolExecutor() as executor:
        for input_path in input_paths:
            executor.submit(crop_images, crop_dir, input_path)


            # # futures = []
            # with ThreadPoolExecutor() as executor:
            #     for key, value in first_value.items():
            #         single_file_output_path =f"{crop_dir}/{key}"
            #         bndbox = value["bndbox"]
            #         executor.submit(crop_image, input_path, single_file_output_path, bndbox)
            #         # futures.append(executor.submit(crop_image, input_path, single_file_output_path, bndbox))


def crop_images(crop_dir, input_path):
    jsonpath = input_path.replace('patch', 'json').replace('.png', '.json')
    with open(jsonpath, 'r') as f:
        json_data = json.load(f)
    first_value = list(json_data.values())[0]
    for key, value in first_value.items():
        singlefileoutput_path = os.path.join(crop_dir, key)
        bndbox = value["bndbox"]
        crop_image(input_path, singlefileoutput_path, bndbox)



def croppatches(input_paths,crop_dir):
    for input_path in input_paths:
        cropallecell(input_path,crop_dir)
# def cropallecellthread(input_path, crop_dir):
#     json_path = Path(input_path.replace('patch', 'json')).with_suffix('.json')
#     if json_path.exists():
#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
#         first_value = list(json_data.values())[0]
#
#         futures = []
#         with ThreadPoolExecutor() as executor:
#             for key, value in first_value.items():
#                 single_file_output_path = Path(crop_dir) / key
#                 bndbox = value["bndbox"]
#                 futures.append(executor.submit(crop_image, input_path, single_file_output_path, bndbox))
#
#         for future in as_completed(futures):
#             future.result()
def cropallecell(input_path,crop_dir):
    jsonpath= input_path.replace('patch', 'json').replace('.png', '.json')
    if os.path.exists(jsonpath):
        with open(jsonpath, 'r') as f:
            json_data = json.load(f)
        first_value = list(json_data.values())[0]
        for key, value in first_value.items():
            singlefileoutput_path=os.path.join(crop_dir, key)
            bndbox=value["bndbox"]
            crop_image(input_path,singlefileoutput_path,bndbox)


def crop_image(input_path, output_path, bndbox):
    """
    切割图像并保存至指定路径

    参数:
    input_path (str): 输入图像的文件路径
    output_path (str): 输出图像的文件路径
    bndbox (dict): 包含切割区域坐标的字典, 键分别为 'xmin', 'ymin', 'xmax', 'ymax'
    """
    # 读取输入图像
    image = cv2.imread(input_path)

    # 获取切割坐标
    x = int(bndbox['xmin'])
    y = int(bndbox['ymin'])
    width = int(math.ceil(bndbox['xmax'] - bndbox['xmin']))
    height = int(math.ceil(bndbox['ymax'] - bndbox['ymin']))

    # 切割图像
    cropped_image = image[y:y + height, x:x + width]

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # 保存输出图像
    cv2.imwrite(output_path, cropped_image)


def getPeripheralCrop(patchdir,outerRingResidual=0,center_x=None):
    total_patches = len(os.listdir(patchdir))
    max_x  = math.floor(total_patches ** 0.5)
    radius= max_x // 2 - outerRingResidual
    outer_patches = []
    inter_patches = []

    for filename in os.listdir(patchdir):
        if filename.endswith('.png'):
            match = re.search(r'(\d+)-lsil_(\d+)_(\d+).png', filename)
            if match:
                patch_x = int(match.group(2))/512
                patch_y = int(match.group(3))/512
                max_x = max(max_x, patch_x)
                if center_x is None :
                    center_x = (max_x + 1) // 2
                distance_sqe = ((patch_x - center_x) ** 2 + (patch_y -center_x ) ** 2)
                if distance_sqe > radius**2:
                    outer_patches.append(filename)
                else:
                    inter_patches.append(filename)
    return  outer_patches,inter_patches




def mycollate(batches):
    images, image_path = tuple(zip(*batches))
    images = torch.stack(images, dim=0)
    # gts = torch.as_tensor(gts)

    return images, image_path
    # return images, gts, image_path



def predict_batchV2(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        gpu_id: int = 0
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    '''
    return:
        bboxes_batch: list of tensors of shape (n, 4)
        predicts_batch: list of tensors of shape (n,)
        phrases_batch: list of list of strings of shape (n,)
        n is the number of boxes in one image
    '''
    caption = preprocess_caption(caption=caption)
    if device != torch.device('cpu'):
        model = DataParallel(model,device_ids=[gpu_id])
    model = model.to(device)
    image = images.to(device)
    captions = [caption for _ in range(len(images))]
    with torch.no_grad():
        with autocast():
            outputs = model(image, captions=captions)  # <------- I use the same caption for all the images for my use-case
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (num_batch, nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (num_batch, nq, 4)

    # import ipdb; ipdb.set_trace()
    mask = prediction_logits.max(dim=2)[0] > box_threshold  # mask: torch.Size([num_batch, 256])

    bboxes_batch = []
    # predicts_batch = []
    # phrases_batch = []  # list of lists
    # tokenizer = model.module.tokenizer
    # tokenized = tokenizer(caption)
    for i in range(prediction_logits.shape[0]):
        # logits = prediction_logits[i][mask[i]]  # logits.shape = (n, 256)
        # phrases = [
        #     get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        #     for logit  # logit is a tensor of shape (256,) torch.Size([256])
        #     in logits  # torch.Size([7, 256])
        # ]
        boxes = prediction_boxes[i][mask[i]]  # boxes.shape = (n, 4)
        # phrases_batch.append(phrases)
        bboxes_batch.append(boxes)
        # predicts_batch.append(logits.max(dim=1)[0])


    return bboxes_batch


# class myLoadImageFromFile(object):
#     """Load an image from file.
# 
#     Required keys are "img_prefix" and "img_info" (a dict that must contain the
#     key "filename"). Added or updated keys are "filename", "img", "img_shape",
#     "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).
# 
#     Args:
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is an uint8 array.
#             Defaults to False.
#         color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
#             Defaults to 'color'.
# 
#     """
# 
#     def __init__(self,
#                  to_float32=False,
#                  color_type='color',
#                  file_client_args=dict(backend='disk')):
#         self.to_float32 = to_float32
#         self.color_type = color_type
# 
#     def get(self, filepath):
#         """Read data from a given ``filepath`` with 'rb' mode.
# 
#         Args:
#             filepath (str or Path): Path to read data.
# 
#         Returns:
#             bytes: Expected bytes object.
#         """
#         with open(filepath, 'rb') as f:
#             value_buf = f.read()
#         return value_buf
# 
#     def __call__(self, results):
# 
#         filename = results['img_info']['filename']
# 
#         img_bytes = self.get(filename)
#         img = imfrombytes(img_bytes, flag=self.color_type)
# 
#         if self.to_float32:
#             img = img.astype(np.float32)
# 
#         results['filename'] = filename
#         results['ori_filename'] = results['img_info']['filename']
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results['img_norm_cfg'] = dict(
#             mean=np.zeros(num_channels, dtype=np.float32),
#             std=np.ones(num_channels, dtype=np.float32),
#             to_rgb=False)
#         return results

    # def __repr__(self):
    #     repr_str = (f'{self.__class__.__name__}('
    #                 f'to_float32={self.to_float32}, '
    #                 f"color_type='{self.color_type}', "
    #                 f'file_client_args={self.file_client_args})')
    #     return repr_str

    # return bboxes_batch

