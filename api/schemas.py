from pydantic import BaseModel, field_validator, model_validator
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
    @model_validator(mode='after')
    def ensure_client_id(self):
        if not self.client_id:
            self.client_id = generate_client_id()
        return self

class AnnotationItem(BaseModel):
    class_name: str
    bbox: List[float]
    segmentation: Dict[str, Any]  # RLE 格式
    score: list[float]

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
    text_prompt: str
    client_id: Optional[str] = None  # 先设为 None，默认值由验证器处理
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    prompt_type: Optional[str] = "box"  # point, box, mask
    device:Optional[str] = "cuda"
    @model_validator(mode='after')
    def ensure_client_id(self):
        if not self.client_id:
            self.client_id = generate_client_id()
        return self


class ProcessVideoResponse(BaseModel):
    output_video_path: str