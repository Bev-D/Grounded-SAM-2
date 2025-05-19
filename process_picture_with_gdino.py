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
import argparse
import utils.FileTool as filetool
from utils.path_utils import convert_local_to_url_path


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_picture_API(
    img_path: str,
    text_prompt: str,
    client_id,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = None,
    save_visualizations: bool = True,
    dump_json_results: bool = True
):
    """
    使用 Grounding DINO + SAM2 对输入图像进行图文检测与分割，返回结构化结果。

    Args:
        img_path (str): 图像文件路径
        text_prompt (str): 文本提示，如 "car. tire."
        box_threshold (float): 检测框置信度阈值
        text_threshold (float): 文本匹配阈值
        device (str): 计算设备 ("cuda" 或 "cpu")
        output_dir (str): 输出目录
        save_visualizations (bool): 是否保存可视化图像
        dump_json_results (bool): 是否转储 JSON 结果

    Returns:
        dict: 包含检测框、掩码（RLE）、类别、得分等信息
    """
    if filetool.is_url(img_path):
        print(f"检测到 URL，正在下载：{img_path}")
        img_path = filetool.download_file(img_path)  # 下载为本地文件
    else:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"本地文件不存在：{img_path}")
    # 构建输出目录
    if output_dir is None or output_dir=="":
        output_dir = "./results/"+str(client_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_name = os.path.basename(img_path)
    img_stem = os.path.splitext(img_name)[0]
    
    # 定义输出文件路径
    mask_output_path = os.path.join(output_dir, img_stem + "_mask.jpg")
    annotated_output_path = os.path.join(output_dir, img_stem + "_annotated.jpg")
    json_output_path = os.path.join(output_dir, client_id + "_image.json")
    

    # 加载模型
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = load_model(
        model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="gdino_checkpoints/groundingdino_swint_ogc.pth",
        device=device
    )

    # 加载图像
    image_source, image = load_image(img_path)

    # 设置图像
    sam2_predictor.set_image(image_source)

    # 使用 Grounding DINO 预测检测框
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt.lower(),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    


    # 检查是否检测到目标
    if len(input_boxes) == 0:
        print("未检测到目标")
        results = {
        "client_id": client_id,
        "image_path": img_path,
        "text_prompt": text_prompt,
        "mask_image_path": None,  # 或者保存一个空白图像路径
        "annotated_image_path": None,
        "json_result_path": json_output_path if dump_json_results else None,
        "annotations": [],
        "img_width": w,
        "img_height": h,
    }
  

        # # 可选：保存空的 JSON 文件
        # if dump_json_results:
        #     with open(json_output_path, "w") as f:
        #         json.dump(results, f, indent=4)
    
        return results
    
    
    # 使用 SAM2 进行图像分割
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    # 后处理
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))
    labels_list = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    # RLE 编码转换
    mask_rles = [single_mask_to_rle(mask) for mask in masks]





    # 可视化
    if save_visualizations:
        img = cv2.imread(img_path)
        detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_list)

        cv2.imwrite(annotated_output_path, annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        cv2.imwrite(mask_output_path, annotated_frame)

    # 构建返回结果（URL 路径用）
    url_mask_path = convert_local_to_url_path(mask_output_path)
    url_annotated_path = convert_local_to_url_path(annotated_output_path)
    url_json_path = convert_local_to_url_path(json_output_path)

    # 构建输出结果
    results = {
        "client_id":client_id,
        "image_path": img_path,
        "text_prompt": text_prompt,
        "mask_image_path": url_mask_path,                   # ✅ 使用 URL 路径
        "annotated_image_path": url_annotated_path,         # 新增：标注图像路径
        "json_result_path": url_json_path,              # 新增：JSON 结果路径
       # ✅ 使用 URL 路径
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box.tolist(),
                "segmentation": mask_rle,
                "score": score
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores.tolist())
        ],
        "img_width": w,
        "img_height": h,
    }
    # 保存 JSON
    if dump_json_results:
        with open(json_output_path, "w") as f:
            json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Grounding DINO 和 SAM2 对图像进行图文检测与分割")

    # 必需参数
    parser.add_argument("--img-path", type=str, required=True,
                        help="输入图像路径（例如: notebooks/images/truck.jpg）")
    parser.add_argument("--client_id", type=str, required=True,
                        help="任务ID")
    parser.add_argument("--text-prompt", type=str, default="car. tire.",
                        help="文本提示语句（默认: 'car. tire.'）")
    # 可选参数
    
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Grounding DINO 的 box 置信度阈值（默认: 0.35）")
    
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding DINO 的 text 置信度阈值（默认: 0.25）")
    
    parser.add_argument("--output-dir", type=str,
                        help="输出目录根路径（默认: outputs/videotest）")
    
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="运行设备，'cuda' 或 'cpu'（默认: 'cuda'）")
    
    parser.add_argument("--save-visualizations", action="store_true",default=True,
                        help="是否保存可视化图像（默认: True）")
    
    parser.add_argument("--dump-json-results", action="store_true",default=True,
                        help="是否转储 JSON 结果（默认: True）")
    
    args = parser.parse_args([
        "--img-path", "http://10.60.9.26:8000/A01605.JPG",
        "--client_id", "A01607",
        "--text-prompt", "boat."
    ])
    
    
    # 调用 API 函数
    result = process_picture_API(
        img_path=args.img_path,
        client_id=args.client_id,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
        dump_json_results=args.dump_json_results
    )
    # ========== 新增代码：下载图片到本地测试目录 ==========

    # ====================================================
    