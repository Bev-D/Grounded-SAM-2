from fastapi import APIRouter, HTTPException
from typing import Any
from api.schemas import ProcessRequest, ProcessResponse
from process_picture_with_gdino import process_picture_API

router = APIRouter()

@router.post("/process", response_model=ProcessResponse)
def process_image(request: ProcessRequest) -> Any:
    try:
        result = process_picture_API(
            img_path=request.img_path,
            text_prompt=request.text_prompt,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            device=request.device,
            output_dir=request.output_dir,
            save_visualizations=request.save_visualizations,
            dump_json_results=request.dump_json_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
