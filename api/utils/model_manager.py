
import torch
import threading

from sam2.build_sam import build_sam2
from grounding_dino.groundingdino.util.inference import load_model
from sam2.build_sam import build_sam2_video_predictor


class ModelManager:
    _sam2_model = None
    _grounding_model = None
    _sam2_predictor = None
    _lock = threading.Lock()
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_sam2_model(cls):
        with cls._lock:
            if cls._sam2_model is None:
                print("正在加载 SAM2 模型...")
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
                cls._sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=cls._device)
            return cls._sam2_model

    @classmethod
    def get_grounding_model(cls):
        with cls._lock:
            if cls._grounding_model is None:
                print("正在加载 Grounding DINO 模型...")
                cls._grounding_model = load_model(
                    model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    model_checkpoint_path="gdino_checkpoints/groundingdino_swint_ogc.pth",
                    device=cls._device
                )
            return cls._grounding_model

    @classmethod
    def get_sam2_predictor(cls):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(cls.get_sam2_model())
    @classmethod
    def get_sam2_video_predictor(cls):
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        return build_sam2_video_predictor(model_cfg, sam2_checkpoint)    
    
    @classmethod
    def reset_sam_predictor(cls):
        """重置 predictor（适用于图像变化时）"""
        cls._sam2_predictor = None

    @classmethod
    def unload_models(cls):
        """主动卸载模型"""
        with cls._lock:
            if cls._sam2_model is not None:
                del cls._sam2_model
                cls._sam2_model = None
            if cls._grounding_model is not None:
                del cls._grounding_model
                cls._grounding_model = None
            if cls._sam2_predictor is not None:
                del cls._sam2_predictor
                cls._sam2_predictor = None

            torch.cuda.empty_cache()
            print("模型已卸载")

