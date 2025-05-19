from pydantic import BaseModel, field_validator
from typing import List, Optional,Dict, Any
from pydantic import BaseModel

from task_manager import generate_client_id


class ProcessImageRequest(BaseModel):
    img_path: str
    text_prompt: str
    client_id: Optional[str] = None  # 先设为 None，默认值由验证器处理
    box_threshold: Optional[float] = 0.35
    text_threshold: Optional[float] = 0.25
    save_visualizations: Optional[bool] = True
    dump_json_results: Optional[bool] = True
    device:Optional[str] = "cuda"
    @field_validator("client_id", mode="after")
    def set_default_client_id(cls, value):
        if value is None or value == "":
            return generate_client_id()
        return value

class AnnotationItem(BaseModel):
    class_name: str
    bbox: List[float]
    segmentation: Dict[str, Any]  # RLE 格式
    score: list

class ProcessImageResponse(BaseModel):
    client_id: str
    image_path: str
    text_prompt: str
    mask_image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    json_result_path: Optional[str] = None
    annotations: List[AnnotationItem]
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