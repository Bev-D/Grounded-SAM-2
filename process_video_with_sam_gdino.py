import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from win32con import FILE_NAME_NORMALIZED

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import argparse

def process_video_with_sam_gdino(
    video_path: str,
    text_prompt: str = "car.",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    output_dir: str = "./outputs/videotest",
    prompt_type: str = "box",  # point, box, mask
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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

    """ Step 1: 设置路径 """
    file_name = os.path.basename(video_path)
    file_stem = os.path.splitext(file_name)[0]
    
    output_file_path = os.path.join(output_dir, file_stem)
    output_video_path = os.path.join(output_file_path, file_name)
    source_video_frame_dir = os.path.join(output_file_path, "video_frames")
    save_tracking_results_dir = os.path.join(output_file_path, "tracking_results")

    os.makedirs(source_video_frame_dir, exist_ok=True)
    os.makedirs(save_tracking_results_dir, exist_ok=True)
    
    """ Step 2: 加载模型 """
    grounding_model = load_model(
        model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="gdino_checkpoints/groundingdino_swint_ogc.pth",
        device=device
    )

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)



        
        
    """ Step 3: 视频帧提取 """
    frame_generator = sv.get_video_frames_generator(video_path, stride=1)
    with sv.ImageSink(target_dir_path=source_video_frame_dir, overwrite=True, image_name_pattern="{:05d}.jpg") as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    frame_names = sorted(
        [p for p in os.listdir(source_video_frame_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    inference_state = video_predictor.init_state(video_path=source_video_frame_dir)

    ann_frame_idx = 0
    img_path = os.path.join(source_video_frame_dir, frame_names[ann_frame_idx])
    image_source, image = load_image(img_path)



    """ Step 4: Grounding DINO 检测 """
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    class_names = labels

    """ Step 5: SAM 图像预测 """
    image_predictor.set_image(image_source)

    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    """ Step 6: 添加提示信息到视频追踪器 """
    assert prompt_type in ["point", "box", "mask"], "prompt_type 必须是 ['point', 'box', 'mask']"

    if prompt_type == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(class_names, all_sample_points), start=1):
            labels_np = np.ones((points.shape[0]), dtype=np.int32)
            video_predictor.add_new_points_or_box(inference_state, ann_frame_idx, object_id, points, labels_np)

    elif prompt_type == "box":
        for object_id, (label, box) in enumerate(zip(class_names, input_boxes), start=1):
            video_predictor.add_new_points_or_box(inference_state, ann_frame_idx, object_id, box=box)

    elif prompt_type == "mask":
        for object_id, (label, mask) in enumerate(zip(class_names, masks), start=1):
            labels_np = np.ones((1), dtype=np.int32)
            video_predictor.add_new_mask(inference_state, ann_frame_idx, object_id, mask)

    """ Step 7: 跟踪传播 """
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """ Step 8: 可视化 & 保存结果 """
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(class_names, start=1)}
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(source_video_frame_dir, frame_names[frame_idx]))

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        cv2.imwrite(os.path.join(save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    """ Step 9: 生成输出视频 """
    create_video_from_images(save_tracking_results_dir, output_video_path)

    return output_video_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Grounding DINO 和 SAM 2 进行视频对象跟踪")

    parser.add_argument("--video-path", type=str, required=True,
                        help="输入视频路径（例如: ./assets/Rec_0007.mp4）")
    
    parser.add_argument("--text-prompt", type=str, default="car.",
                        help="文本提示语句（默认: 'car.'）")
    
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Grounding DINO 的 box 置信度阈值（默认: 0.35）")
    
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding DINO 的 text 置信度阈值（默认: 0.25）")
    
    parser.add_argument("--output-dir", type=str, default="./outputs/videotest",
                        help="输出目录根路径（默认: ./outputs/videotest）")
    
    parser.add_argument("--prompt-type", type=str, choices=["point", "box", "mask"], default="box",
                        help="提示类型，支持 'point', 'box', 'mask'（默认: 'box'）")
    
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="运行设备，'cuda' 或 'cpu'（默认: 'cuda'）")

    args = parser.parse_args([
        "--video-path", "./assets/Rec_0007.mp4",
        "--text-prompt", "car.",
        "--prompt-type", "box"
    ])

    result_video = process_video_with_sam_gdino(
        video_path=args.video_path,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type,
        device=args.device
    )

    print(f"✅ 输出视频已保存至: {result_video}")
