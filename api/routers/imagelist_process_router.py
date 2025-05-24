# api/routers/imagelist_process_router.py
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import ProcessImageRequest, ProcessImageResponse
from image_inference_service import run_image_inference

router = APIRouter()

@router.post("/process", response_model=ProcessImageResponse)
def process_image(request: ProcessImageRequest) -> Any:
    try:
        result = run_image_inference(
            img_path=request.img_path,
            text_prompt=request.text_prompt,
            client_id=request.client_id,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            device=request.device,
            save_visualizations=request.save_visualizations,
            dump_json_results=request.dump_json_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
