# grounded_sam_processor.py

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
import utils.FileTool as filetool
from utils.path_utils import convert_local_to_url_path


class GroundedSAMProcessor:
    def __init__(self,
                 sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
                 model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                 grounding_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_ckpt="gdino_checkpoints/groundingdino_swint_ogc.pth",
                 device="cuda"):
        """
        初始化模型组件，只加载一次，提升性能。
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # 加载 SAM2 模型
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # 加载 Grounding DINO 模型
        self.grounding_model = load_model(
            model_config_path=grounding_config,
            model_checkpoint_path=grounding_ckpt,
            device=self.device
        )

    def single_mask_to_rle(self, mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def process_image(self, img_path, text_prompt, client_id,
                      box_threshold=0.35, text_threshold=0.25,
                      output_dir=None, save_visualizations=True, dump_json_results=True):
        """
        处理单张图像，返回结构化结果。
        """
        if filetool.is_url(img_path):
            print(f"检测到 URL，正在下载：{img_path}")
            img_path = filetool.download_file(img_path)  # 下载为本地文件
        else:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"本地文件不存在：{img_path}")

        # 构建输出目录
        if output_dir is None or output_dir == "":
            output_dir = "./results/" + str(client_id)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]

        # 定义输出文件路径
        mask_output_path = os.path.join(output_dir, img_stem + "_mask.jpg")
        annotated_output_path = os.path.join(output_dir, img_stem + "_annotated.jpg")
        json_output_path = os.path.join(output_dir, client_id + "_image.json")

        # 加载图像
        image_source, image = load_image(img_path)

        # 设置图像
        self.sam2_predictor.set_image(image_source)

        # 使用 Grounding DINO 预测检测框
        boxes, confidences, labels = predict(
            model=self.grounding_model,
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
                "mask_image_path": None,
                "annotated_image_path": None,
                "json_result_path": json_output_path if dump_json_results else None,
                "annotations": [],
                "img_width": w,
                "img_height": h,
            }
            return results

        # 使用 SAM2 进行图像分割
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            masks, scores, logits = self.sam2_predictor.predict(
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
        mask_rles = [self.single_mask_to_rle(mask) for mask in masks]

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
            "client_id": client_id,
            "image_path": img_path,
            "text_prompt": text_prompt,
            "mask_image_path": url_mask_path,
            "annotated_image_path": url_annotated_path,
            "json_result_path": url_json_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": mask_rle,
                    "score": score if isinstance(score, list) else [score]  # 统一转为列表
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

    def batch_process_images(self, image_paths, text_prompt, client_ids, **kwargs):
        """
        批量处理多张图像
        """
        results = []
        for img_path, client_id in zip(image_paths, client_ids):
            result = self.process_image(img_path, text_prompt, client_id, **kwargs)
            results.append(result)
        return results


