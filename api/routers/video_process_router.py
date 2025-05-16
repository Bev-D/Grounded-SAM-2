from fastapi import APIRouter, HTTPException
from typing import Any

from api.schemas import ProcessVideoRequest, ProcessVideoResponse
from process_video_with_sam_gdino import process_video_with_sam_gdino

router = APIRouter()

@router.post("/process", response_model=ProcessVideoResponse)
def process_video(request: ProcessVideoRequest) -> Any:
    try:
        result = process_video_with_sam_gdino(
            video_path=request.video_path,
            text_prompt=request.text_prompt,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            output_dir=request.output_dir,
            prompt_type=request.prompt_type,
            device=request.device
        )
        return {"output_video_path": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
