from pydantic import BaseModel
from typing import List, Optional

class ProcessRequest(BaseModel):
    img_path: str
    text_prompt: str
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cuda"
    output_dir: Optional[str] = "./outputs/picturetest"
    save_visualizations: bool = True
    dump_json_results: bool = True

class AnnotationResult(BaseModel):
    class_name: str
    bbox: List[float]
    segmentation: dict  # RLE 格式
    score: float

class ProcessResponse(BaseModel):
    image_path: str
    annotations: List[AnnotationResult]
    box_format: str
    img_width: int
    img_height: int


# 视频处理模型（新增）
class ProcessVideoRequest(BaseModel):
    video_path: str
    text_prompt: str = "car."
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    output_dir: Optional[str] = "./outputs/videotest"
    prompt_type: str = "box"  # point, box, mask
    device: str = "cuda"


class ProcessVideoResponse(BaseModel):
    output_video_path: str