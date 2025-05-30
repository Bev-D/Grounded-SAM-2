import os
import cv2
import torch
import argparse

import time  # 在文件顶部添加
from PIL import Image
import shutil
import json
import copy
import shutil
from torch.utils.data import DataLoader
import numpy as np
import supervision as sv
import utils.FileTool as filetool
from utils.batch_util import ImageDataset,predict_batch,totensor,predict_batch2
from torchvision.ops import box_convert
from tqdm import tqdm
from api.utils.model_manager import ModelManager
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict,annotate
from utils.common_utils import CommonUtils
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images, create_video_from_image_array
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from utils.draw_utils import display_image_with_boxes
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from groundingdino.util import box_ops



def process_video_with_sam_gdino(
        video_path,
        client_id,
        text_prompt: str ,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        output_dir: str = None ,
        prompt_type: str = "mask",  # point, box, mask
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_image: bool =False
):
    """
    使用 Grounding DINO 和 SAM 2 处理视频并进行对象跟踪。

    参数:
        video_path (str): 输入视频路径。
        text_prompt (str): 文本提示语句，如 "car."。
        box_threshold (float): Grounding DINO 的框置信度阈值。
        text_threshold (float): 文本匹配置信度阈值。
        output_dir (str): 输出目录根路径。
        prompt_type (str): 提示类型，支持 'point', 'box', 'mask'。
        device (str): 计算设备 ('cuda' or 'cpu')。

    返回:
        str: 输出视频路径。
    """
    
    start_time = time.time()  # 记录开始时间
    if video_path==None:
        print("视频路径为空，跳过处理")

        return {
            "status": "warning",
            "message": "没有合法视频可供下载",
            "data": {
                "output_video_path": None,
                "class_names": [],
                "total_frames": 0,
            },
            "error": None
        }
    """ Step 1: 设置路径 """
    # todo：格式化路径
    # 构建输出目录
    if output_dir is None or output_dir=="":
        output_dir = "./results/"+str(client_id)
    
    if ',' in video_path:
        file_list = [f.strip() for f in video_path.split(',')]
        output_file_path = os.path.join(output_dir, "Images_Task")
        source_video_frame_dir=os.path.join(output_file_path, "Images_Source")
        CommonUtils.creat_dirs(output_file_path)
        CommonUtils.creat_dirs(source_video_frame_dir)
        for file_name in file_list:
            if filetool.is_url(file_name):
                print(f"检测到 URL，正在下载：{video_path}")
                image_name = filetool.download_file(video_path)  # 下载为本地文件
                file_name = os.path.basename(image_name)
                destination_path = os.path.join(output_file_path, file_name)
                shutil.copy(src=image_name, dst=destination_path)
               
    else:
        if filetool.is_url(video_path):
            print(f"检测到 URL，正在下载：{video_path}")
            video_path = filetool.download_file(video_path)  # 下载为本地文件
          
        else:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"本地文件不存在：{video_path}")
        file_name = os.path.basename(video_path)
        file_stem = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(output_dir, file_stem)
        output_video_path = os.path.join(output_dir, file_name)
        source_video_frame_dir = os.path.join(output_file_path, "video_frames")
        CommonUtils.creat_dirs(output_file_path)
        CommonUtils.creat_dirs(source_video_frame_dir)
        # os.makedirs(source_video_frame_dir, exist_ok=True)
    # video_path=os.path.normpath(os.path.abspath(video_path))
    # output_file_path = os.path.join(output_dir, file_stem)
    # output_video_path = os.path.join(output_dir, file_name)
    mask_data_dir = os.path.join(output_file_path, "mask_data")
    CommonUtils.creat_dirs(mask_data_dir)
    json_data_dir = os.path.join(output_file_path, "json_data")
    CommonUtils.creat_dirs(json_data_dir)
    # source_video_frame_dir = os.path.join(output_file_path, "video_frames")
    save_tracking_results_dir = os.path.join(output_file_path, "tracking_results")
    os.makedirs(save_tracking_results_dir, exist_ok=True)
    

    """ Step 2: 加载模型 """
    # grounding_model = load_model(
    #     model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    #     model_checkpoint_path="gdino_checkpoints/groundingdino_swint_ogc.pth",
    #     device=device
    # )

    # sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    # sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    # image_predictor = SAM2ImagePredictor(sam2_image_model)

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # ✅ 使用 ModelManager 获取模型
    

    sam2_model = ModelManager.get_sam2_model()
    video_predictor =ModelManager.get_sam2_video_predictor()
    grounding_model = ModelManager.get_grounding_model()
    sam2_image_model = ModelManager.get_sam2_model()
    image_predictor = ModelManager.get_sam2_predictor()

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    # processor = AutoProcessor.from_pretrained(model_id)
    # grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


    # Step 3: 视频帧提取（先检查是否已经存在）
    if os.path.exists(source_video_frame_dir) and len(os.listdir(source_video_frame_dir)) > 0:
        print(f"⚠️ 帧目录 {source_video_frame_dir} 已存在且非空，跳过帧提取步骤")
        # frame_names = sorted(
        #     [p for p in os.listdir(source_video_frame_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
        #     key=lambda x: int(os.path.splitext(x)[0])
        # )
    else:
        print(f"✅ 开始提取视频帧到 {source_video_frame_dir}")
        # 清除旧文件并重新生成
        if os.path.exists(source_video_frame_dir):
            shutil.rmtree(source_video_frame_dir)
        os.makedirs(source_video_frame_dir, exist_ok=True)
    
        frame_generator = sv.get_video_frames_generator(video_path, stride=1)
        with sv.ImageSink(target_dir_path=source_video_frame_dir, overwrite=True, image_name_pattern="{:05d}.jpg") as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)
    
    frame_names = sorted(
            [os.path.join(source_video_frame_dir, p) for p in os.listdir(source_video_frame_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )


    sam2_masks = MaskDictionaryModel()

    dataset = ImageDataset(frame_names,transform=totensor())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    with tqdm(total=len(dataloader)) as _tqdm:
        _tqdm.set_description(f'detect: ')

        for images, img_paths in dataloader:
            boxes, logits, phrases = predict_batch(
                model=grounding_model,
                images=images,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            imagesformask=[]
            for i in range(len(boxes)):
            
                if len(boxes[i]) == 0:
                    print(f"No objects of the '{text_prompt}' prompt detected in the image.")
                    # shutil.copy(img_paths[i], savedir)
                else:
                    # Save the boxes
                    box_save_path = os.path.join(save_tracking_results_dir, os.path.basename(img_paths[i]))
                    image_source = Image.open(img_paths[i]).convert("RGB")
                    image = np.asarray(image_source)
                    imagesformask.append(image)
                    annotated_frame = annotate(image_source=image, boxes=boxes[i], logits=logits[i], phrases=phrases[i])
                    # cv2.imwrite(box_save_path, annotated_frame)
            boxes_np = [box.cpu().numpy() for box in boxes[i]]
            image_predictor.set_image_batch(imagesformask)
            imagesformask.clear()
            masks, scores, logits = image_predictor.predict_batch(
                point_coords_batch=None,
                point_labels_batch=None,
                box_batch=boxes_np,
                multimask_output=False,
            )
            # visualization results
            # convert the shape to (n, H, W)
            for img_idx in range(len(masks)):
                if masks[img_idx].ndim == 3:
                    masks = masks[img_idx]
                    
                class_ids = np.array(list(range(len(logits[img_idx]))))
                confidences = scores[img_idx]
                class_names = np.array(list(range(len(logits))))
            # class_ids = np.array(list(range(len(class_names))))
            
                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(class_names, confidences)
                ]
            
                img = cv2.imread(img_paths[i])
                detections = sv.Detections(
                            xyxy=np.array(boxes[i]),
                            mask=masks.astype(bool),
                            class_id=class_ids
                            )

                mask_save_path = os.path.join(save_tracking_results_dir, os.path.basename(img_paths[i]))
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                
                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                # cv2.imwrite(0,mask_save_path), annotated_frame)
                
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                cv2.imwrite(mask_save_path, annotated_frame)
            
                
                    # image_pil = Image.open(img_paths[i]).convert("RGB")
                    # # input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                    # display_image_with_boxes(image_pil, boxes[i], logits[i], box_save_path)
                    
                    
            
            _tqdm.update()
    
    return {"mask_data_dir": mask_data_dir, "json_data_dir": json_data_dir}
    
    
    
    # image_predictor.set_image_batch()
    # 
    # 
    # 
    # 
    # 
    # 
    # # 初始化视频预测器的状态
    # inference_state = video_predictor.init_state(video_path=source_video_frame_dir, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
    # step = 20 # 为 Grounding DINO 预测器采样帧的步骤
    # sam2_masks = MaskDictionaryModel()
    # 
    # objects_count = 0
    # 
    # for start_frame_idx in range(0, len(frame_names), step):
    #     # prompt grounding dino to get the box coordinates on specific frame
    #     print("start_frame_idx/total_number_of_frames：", start_frame_idx,  "/", len(frame_names))
    #     # continue
    #     img_path = os.path.join(source_video_frame_dir, frame_names[start_frame_idx])
    #     image = Image.open(img_path)
    #     image_base_name = frame_names[start_frame_idx].split(".")[0]
    #     mask_dict = MaskDictionaryModel(promote_type =prompt_type, mask_name = f"mask_{image_base_name}.npy")
    # 
    #     # run Grounding DINO on the image
    #     inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         outputs = grounding_model(**inputs)
    # 
    #     results = processor.post_process_grounded_object_detection(
    #         outputs,
    #         inputs.input_ids,
    #         box_threshold=box_threshold,
    #         text_threshold=text_threshold,
    #         target_sizes=[image.size[::-1]]
    #     )
    # 
    #     # prompt SAM image predictor to get the mask for the object
    #     # 提示 SAM 图像预测器获取对象的掩码
    #     image_predictor.set_image(np.array(image.convert("RGB")))
    # 
    #     # process the detection results
    #     # 处理检测结果
    #     input_boxes = results[0]["boxes"] # .cpu().numpy()
    #     # print("results[0]",results[0])
    #     OBJECTS = results[0]["labels"]
    #     if input_boxes.shape[0] != 0:
    #         # prompt SAM 2 image predictor to get the mask for the object
    #         # 提示 SAM 2 图像预测器获取对象的掩码
    #         masks, scores, logits = image_predictor.predict(
    #             point_coords=None,
    #             point_labels=None,
    #             box=input_boxes,
    #             multimask_output=False,
    #         )
    #         # convert the mask shape to (n, H, W)
    #         if masks.ndim == 2:
    #             masks = masks[None]
    #             scores = scores[None]
    #             logits = logits[None]
    #         elif masks.ndim == 4:
    #             masks = masks.squeeze(1)
    # 
    # 
    #         """
    #         Step 3: Register each object's positive points to video predictor
    #         第 3 步：将每个对象的正点注册到视频预测器
    #         """
    # 
    #         # If you are using point prompts, we uniformly sample positive points based on the mask
    #         # 如果您使用的是点提示，我们将根据掩码统一对正点进行采样
    #         if mask_dict.promote_type == "mask":
    #             mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
    #         else:
    #             raise NotImplementedError("SAM 2 video predictor only support mask prompts,SAM 2 视频预测器仅支持掩码提示")
    # 
    # 
    # 
    #         """
    #         Step 4: Propagate the video predictor to get the segmentation results for each frame
    #         第 4 步：传播视频预测器以获取每帧的分割结果
    #         """
    #         objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    #         print("objects_count", objects_count)
    #     else:
    #         print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
    #         mask_dict = sam2_masks
    # 
    # 
    #     if len(mask_dict.labels) == 0:
    #         mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
    #         print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
    #         continue
    #     else:
    #         video_predictor.reset_state(inference_state)
    # 
    #         for object_id, object_info in mask_dict.labels.items():
    #             # frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
    #             video_predictor.add_new_mask(
    #                 inference_state,
    #                 start_frame_idx,
    #                 object_id,
    #                 object_info.mask,
    #             )
    # 
    #         video_segments = {}  # 输出以下 {step} 帧跟踪蒙版
    #         for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
    #             frame_masks = MaskDictionaryModel()
    # 
    #             for i, out_obj_id in enumerate(out_obj_ids):
    #                 out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
    #                 object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
    #                 object_info.update_box()
    #                 frame_masks.labels[out_obj_id] = object_info
    #                 image_base_name = frame_names[out_frame_idx].split(".")[0]
    #                 frame_masks.mask_name = f"mask_{image_base_name}.npy"
    #                 frame_masks.mask_height = out_mask.shape[-2]
    #                 frame_masks.mask_width = out_mask.shape[-1]
    # 
    #             video_segments[out_frame_idx] = frame_masks
    #             sam2_masks = copy.deepcopy(frame_masks)
    # 
    #         print("video_segments:", len(video_segments))
    # 
    #     """
    #     Step 5: save the tracking masks and json files
    #     """
    #     for frame_idx, frame_masks_info in video_segments.items():
    #         mask = frame_masks_info.labels
    #         mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
    #         for obj_id, obj_info in mask.items():
    #             mask_img[obj_info.mask == True] = obj_id
    # 
    #         mask_img = mask_img.numpy().astype(np.uint16)
    #         np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)
    # 
    #         json_data = frame_masks_info.to_dict()
    #         json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
    #         with open(json_data_path, "w") as f:
    #             json.dump(json_data, f)
    # 
    #     # 获取当前循环处理的图像名称列表
    #     current_image_names = frame_names[start_frame_idx:start_frame_idx + step]
    #   
    #     CommonUtils.draw_masks_and_box_with_supervision_imagenamelist(source_video_frame_dir,current_image_names, mask_data_dir, json_data_dir, save_tracking_results_dir)
    #     
    # 
    # # """
    # # Step 6: Draw the results and save the video
    # # """
    # # CommonUtils.draw_masks_and_box_with_supervision(source_video_frame_dir, mask_data_dir, json_data_dir, save_tracking_results_dir)
    # # 
    # create_video_from_images(save_tracking_results_dir, output_video_path, frame_rate=15)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time:", total_time)
    # return {
    #     "status": "success",
    #     "message": "视频处理完成并成功生成输出文件。",
    #     "data": {
    #         "output_video_path": output_video_path,
    #         "tracking_result_dir": save_tracking_results_dir if save_image else None,
    #         "class_names": [],
    #         "total_frames": len(video_segments),
    #     },
    #     "error": None
    # }
    # 
    # 
    # # 提示 grounding dino 获取特定帧的 box 坐标
    # img_path = os.path.join(source_video_frame_dir, frame_names[ann_frame_idx])
    # image_source, image = load_image(img_path)
    # image_base_name = frame_names[start_frame_idx].split(".")[0]
    # mask_dict = MaskDictionaryModel(promote_type = prompt_type, mask_name = f"mask_{image_base_name}.npy")
    # 
    # 
    # """ Step 4: Grounding DINO 检测 """
    # boxes, confidences, labels = predict(
    #     model=grounding_model,
    #     image=image,
    #     caption=text_prompt,
    #     box_threshold=box_threshold,
    #     text_threshold=text_threshold
    # )
    # 
    # h, w, _ = image_source.shape
    # boxes = boxes * torch.Tensor([w, h, w, h])
    # input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # class_names = labels
    # 
    # if len(input_boxes) == 0:
    #     print("⚠️ 没有检测到任何对象，跳过 SAM 图像预测和跟踪流程")
    #     return {
    #         "status": "warning",
    #         "message": "No objects detected by Grounding DINO.",
    #         "data": {
    #             "output_video_path": output_video_path,
    #             "class_names": [],
    #             "total_frames": 0,
    #         },
    #         "error": None
    #     }
    # 
    # """ Step 5: SAM 图像预测 """
    # image_predictor.set_image(image_source)
    # 
    # torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    # if torch.cuda.get_device_properties(0).major >= 8:
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True
    # 
    # masks, scores, logits = image_predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_boxes,
    #     multimask_output=False
    # )
    # if masks.ndim == 4:
    #     masks = masks.squeeze(1)
    # 
    # """ Step 6: 添加提示信息到视频追踪器 """
    # assert prompt_type in ["point", "box", "mask","noprompt"], "prompt_type 必须是 ['point', 'box', 'mask','noprompt']"
    # 
    # if prompt_type == "point":
    #     all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
    #     for object_id, (label, points) in enumerate(zip(class_names, all_sample_points), start=1):
    #         labels_np = np.ones((points.shape[0]), dtype=np.int32)
    #         video_predictor.add_new_points_or_box(inference_state, ann_frame_idx, object_id, points, labels_np)
    # 
    # elif prompt_type == "box":
    #     for object_id, (label, box) in enumerate(zip(class_names, input_boxes), start=1):
    #         video_predictor.add_new_points_or_box(inference_state, ann_frame_idx, object_id, box=box)
    # 
    # elif prompt_type == "mask":
    #     for object_id, (label, mask) in enumerate(zip(class_names, masks), start=1):
    #         labels_np = np.ones((1), dtype=np.int32)
    #         video_predictor.add_new_mask(inference_state, ann_frame_idx, object_id, mask)
    # elif prompt_type == "noprompt":
    #     enable_tracker=False
    # 
    # 
    # """ Step 7: 跟踪传播 """
    # video_segments = {}
    # for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    #     video_segments[out_frame_idx] = {
    #         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    #         for i, out_obj_id in enumerate(out_obj_ids)
    #     }
    # 
    # """ Step 8: 可视化 & 保存结果 """
    # ID_TO_OBJECTS = {i: obj for i, obj in enumerate(class_names, start=1)}
    # annotated_frames = []  # 存储 annotated_frame 的数组
    # 
    # # 使用 tqdm 包裹 items() 返回的迭代器，并排序确保顺序一致
    # sorted_segments = sorted(video_segments.items(), key=lambda x: x[0])
    # 
    # for frame_idx, segments in tqdm(sorted_segments, desc="Processing Video Frames", total=len(video_segments)):
    #     img_path = os.path.join(source_video_frame_dir, frame_names[frame_idx])
    #     img = cv2.imread(img_path)
    # 
    #     object_ids = list(segments.keys())
    #     masks = list(segments.values())
    #     masks = np.concatenate(masks, axis=0)
    # 
    #     detections = sv.Detections(
    #         xyxy=sv.mask_to_xyxy(masks),
    #         mask=masks,
    #         class_id=np.array(object_ids, dtype=np.int32),
    #     )
    # 
    #     box_annotator = sv.BoxAnnotator()
    #     label_annotator = sv.LabelAnnotator()
    #     mask_annotator = sv.MaskAnnotator()
    # 
    #     annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    #     annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    #     annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # 
    #     if save_image:
    #         cv2.imwrite(os.path.join(save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
    #     else:
    #         annotated_frames.append(annotated_frame)
    # 
    # 
    # """ Step 9: 生成输出视频 """
    # 
    # if save_image:
    #     create_video_from_images(save_tracking_results_dir, output_video_path)
    # else:
    #     create_video_from_image_array(annotated_frames, output_video_path)
    # 
    # return {
    #     "status": "success",
    #     "message": "视频处理完成并成功生成输出文件。",
    #     "data": {
    #         "output_video_path": output_video_path,
    #         "tracking_result_dir": save_tracking_results_dir if save_image else None,
    #         "class_names": class_names,
    #         "total_frames": len(video_segments),
    #     },
    #     "error": None
    # }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Grounding DINO 和 SAM 2 进行视频对象跟踪")

    parser.add_argument("--video-path", type=str, required=True,
                        help="输入视频路径（例如: ./assets/Rec_0007.mp4）")

    parser.add_argument("--text-prompt", type=str, default="car.",
                        help="文本提示语句（默认: 'car.'）")

    parser.add_argument("--client_id", type=str, required=True,
                        help="任务ID")

    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Grounding DINO 的 box 置信度阈值（默认: 0.35）")

    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding DINO 的 text 置信度阈值（默认: 0.25）")

    parser.add_argument("--output-dir", type=str, default="./outputs/videotest",
                        help="输出目录根路径（默认: ./outputs/videotest）")

    parser.add_argument("--prompt-type", type=str, choices=["point", "box", "mask","noprompt"], default="box",
                        help="提示类型，支持 'point', 'box', 'mask'（默认: 'box'）")

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="运行设备，'cuda' 或 'cpu'（默认: 'cuda'）")

    parser.add_argument("--save_image", action="store_true",
                        help="是否保存每帧结果为图像文件（默认: False）")

    args = parser.parse_args([
        "--video-path", "E:/2025/09_无人机平台\视频素材/smoketest.MP4",
        "--text-prompt", "smoke. building.",
        "--client_id", "smoke",
        "--prompt-type", "mask",
        "--box-threshold","0.4",
        "--text-threshold","0.3",
        "--save_image"
    ])

    result_video = process_video_with_sam_gdino(
        video_path=args.video_path,
        text_prompt=args.text_prompt,
        client_id=args.client_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        prompt_type=args.prompt_type,
        device=args.device,
        save_image=args.save_image
    )

    print(f"✅ 输出视频已保存至: {result_video}")
