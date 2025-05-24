# video_inference_service.py
import json
import os
import cv2
import numpy as np
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import partial

import supervision as sv
from torchvision.ops import box_convert
from pycocotools.mask import encode as mask_encode
from tqdm import tqdm

from grounding_dino.groundingdino.util.inference import load_model, predict, load_image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from api.utils.model_manager import ModelManager
from utils.path_utils import convert_local_to_url_path
from utils.FileTool import is_url, download_file


def single_mask_to_rle(mask):
    rle = mask_encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


async def run_video_inference(
    video_path: str,
    text_prompt: str,
    client_id: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Optional[str] = None,
    save_visualizations: bool = True,
    dump_json_results: bool = True
) -> Dict[str, Any]:
    """
    使用 Grounding DINO + SAM2 对输入视频进行图文检测与跟踪，返回结构化结果。

    Args:
        video_path (str): 视频文件路径
        text_prompt (str): 文本提示，如 "car."
        client_id (str): 客户端 ID（用于隔离输出）
        box_threshold (float): 检测框置信度阈值
        text_threshold (float): 文本匹配阈值
        device (str): 计算设备 ("cuda" 或 "cpu")
        output_dir (str): 输出目录根路径
        save_visualizations (bool): 是否保存可视化图像
        dump_json_results (bool): 是否转储 JSON 结果

    Returns:
        dict: 包含输出视频路径、掩码路径、JSON 路径、标注信息等
    """
        
    # 构建输出目录
    if output_dir is None or output_dir=="":
        output_dir = "./results/"+str(client_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = os.path.basename(video_path)
    file_stem = os.path.splitext(file_name)[0]
    
    
    # 定义输出文件路径
    source_video_frame_dir = output_dir / "video_frames"
    tracking_result_dir = output_dir / "tracking_results"
    json_output_path = output_dir / f"{client_id}_video.json"

    source_video_frame_dir.mkdir(exist_ok=True)
    tracking_result_dir.mkdir(exist_ok=True)

    # 下载远程视频（如果需要）

    if is_url(video_path):
        print(f"检测到 URL，正在下载：{video_path}")
        video_path = download_file(video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"本地视频不存在：{video_path}")

    # 加载模型
    grounding_model = ModelManager.get_grounding_model()
    sam2_model = ModelManager.get_sam2_model()
    
    # 不知道哪种比较好
    # image_predictor = SAM2ImagePredictor(sam2_model)
    image_predictor = ModelManager.get_sam2_predictor()

    # Step 1: 提取视频帧（异步）
    frame_paths = await extract_frames_async(video_path, source_video_frame_dir)

    # Step 2: 推理第一帧获取初始掩码和框
    first_frame_path = frame_paths[0]
    image_source, image = await load_image_async(first_frame_path)

    image_predictor.set_image(image_source)
    boxes, confidences, labels = await detect_objects_async(
        grounding_model, image, text_prompt, box_threshold, text_threshold
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    if len(input_boxes) == 0:
        print("未检测到目标")
        results = {
            "client_id": client_id,
            "video_path": video_path,
            "text_prompt": text_prompt,
            "output_video_path": None,
            "json_result_path": json_output_path if dump_json_results else None,
            "annotations": [],
            "img_width": w,
            "img_height": h,
        }
        if dump_json_results:
            with open(json_output_path, "w") as f:
                json.dump(results, f, indent=4)
        return results

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Step 3: 初始化视频追踪器
    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = ModelManager.get_sam2_checkpoint()
    model_cfg = ModelManager.get_sam2_config()
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    inference_state = video_predictor.init_state(video_path=str(source_video_frame_dir))

    # Step 4: 添加提示信息到视频追踪器
    ann_frame_idx = 0
    class_names = labels

    for object_id, (label, box) in enumerate(zip(class_names, input_boxes), start=1):
        video_predictor.add_new_points_or_box(inference_state, ann_frame_idx, object_id, box=box)

    # Step 5: 异步执行视频传播
    video_segments = {}
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    def propagate_fn():
        segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return segments

    video_segments = await loop.run_in_executor(executor, propagate_fn)

    # Step 6: 可视化 & 保存结果
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(class_names, start=1)}
    visualized_frames = []

    for frame_idx, segments in tqdm.tqdm(video_segments.items(), desc="生成可视化帧"):
        img_path = os.path.join(source_video_frame_dir, os.listdir(source_video_frame_dir)[frame_idx])
        img = cv2.imread(img_path)

        object_ids = list(segments.keys())
        masks_list = list(segments.values())
        masks = np.concatenate(masks_list, axis=0)

        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks), mask=masks, class_id=np.array(object_ids, dtype=np.int32))

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        output_path = os.path.join(tracking_result_dir, f"annotated_frame_{frame_idx:05d}.jpg")
        cv2.imwrite(output_path, annotated_frame)
        visualized_frames.append(output_path)

    # Step 7: 合成视频
    output_video_path = os.path.join(output_dir, f"{file_stem}_tracked.mp4")
    await create_video_from_images_async(visualized_frames, output_video_path)

    # 构建返回结果
    url_video_path = convert_local_to_url_path(output_video_path)
    url_json_path = convert_local_to_url_path(json_output_path)

    annotations = [
        {
            "class_name": class_name,
            "bbox": box.tolist(),
            "score": score if isinstance(score, list) else [score]
        }
        for class_name, box, score in zip(class_names, input_boxes, scores.tolist())
    ]

    results = {
        "client_id": client_id,
        "video_path": video_path,
        "text_prompt": text_prompt,
        "output_video_path": url_video_path,
        "json_result_path": url_json_path,
        "annotations": annotations,
        "img_width": w,
        "img_height": h
    }

    # 保存 JSON
    if dump_json_results:
        with open(json_output_path, "w") as f:
            json.dump(results, f, indent=4)

    return results


# 异步帧提取
async def extract_frames_async(video_path: str, output_dir: Path) -> List[str]:
    loop = asyncio.get_event_loop()
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"{idx:05d}.jpg")
        await loop.run_in_executor(None, cv2.imwrite, frame_path, frame)
        frame_paths.append(frame_path)
        idx += 1

    cap.release()
    return frame_paths


# 异步加载图像
async def load_image_async(image_path: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_image, image_path)


# 异步检测对象
async def detect_objects_async(model, image, caption, box_threshold, text_threshold):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        predict,
        model,
        image,
        caption,
        box_threshold,
        text_threshold
    )


# 异步生成视频
async def create_video_from_images_async(image_paths: List[str], output_path: str):
    loop = asyncio.get_event_loop()
    from utils.video_utils import create_video_from_images
    return await loop.run_in_executor(None, create_video_from_images, image_paths, output_path)
