import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

"""
Hyper parameters
"""
TEXT_PROMPT = "car. tire."
IMG_PATH = "notebooks/images/truck.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image_source, image = load_image(img_path)

sam2_predictor.set_image(image_source)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# process the box prompt for SAM 2
# 获取图像的高、宽和通道数，这里只关注高和宽，通道数忽略
h, w, _ = image_source.shape

# 将bounding box坐标从相对值转换为绝对值，以匹配图像的实际尺寸
# 这里假设boxes是相对尺寸的bounding box坐标，格式为中心坐标加宽高（cxcywh）
boxes = boxes * torch.Tensor([w, h, w, h])

# 将bounding box格式从中心坐标加宽高（cxcywh）转换为左上角和右下角坐标（xyxy）
# 这种转换是为了适应某些算法或库对bounding box格式的要求
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()



# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
对模型的输出进行后处理，以获取掩码、分数和 logits，以便进行可视化
"""
# 如果masks的维度是4，意味着它有多个通道，但实际只需要一个通道
# 所以要挤掉第一个维度，确保masks的形状是（n, H, W）
if masks.ndim == 4:
    masks = masks.squeeze(1)

# 将confidences从tensor转换为numpy数组，然后转为python列表
# 这样做是为了后续处理方便，因为列表操作更加灵活
confidences = confidences.numpy().tolist()

# 直接用类名列表作为类别ID的来源
class_names = labels

# 生成类别ID数组，每个类别对应一个唯一的ID
# 这里使用了numpy数组和range函数，以及list构造器转换为列表
class_ids = np.array(list(range(len(class_names))))

# 生成标签列表，每个标签包含类别名和对应的置信度
# 使用了列表推导式，以及字符串格式化来组合类别名和置信度
# 这样的标签对于后续的结果展示或者日志记录非常有用
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]


"""
Visualize image with supervision useful API
使用监督可视化图像有用的 API
Detections：格式化图像风格结果的数据
"""
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

# 创建一个框注释器对象，用于在图像上绘制检测框
box_annotator = sv.BoxAnnotator()

# 使用框注释器在图像副本上绘制检测框，生成带有注释的帧
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

# 创建一个标签注释器对象，用于在图像上添加文本标签
label_annotator = sv.LabelAnnotator()

# 使用标签注释器在已绘制检测框的图像上添加文本标签，进一步丰富注释信息
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

# 将带有框和标签注释的图像保存到指定的输出目录中
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

# 创建一个掩码注释器对象，用于在图像上叠加检测到的掩码
mask_annotator = sv.MaskAnnotator()

# 使用掩码注释器在带有框和标签注释的图像上叠加掩码，提供更丰富的视觉信息
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

# 将带有框、标签和掩码注释的图像保存到指定的输出目录中
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)


"""
Dump the results in standard format and save as json files
以标准格式转储结果并另存为 json 文件
"""

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)