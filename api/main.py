from fastapi import FastAPI
import uvicorn
from api.routers import image_process_router,video_process_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
app = FastAPI(title="Grounded SAM2 Image Processing API")

# 挂载静态资源目录
app.mount("/static", StaticFiles(directory="api/static", html=True), name="static")


# 注册图像处理路由
app.include_router(image_process_router.router, prefix="/api/v1/image", tags=["Image Processing"])

# 注册视频处理路由
app.include_router(video_process_router.router, prefix="/api/v1/video", tags=["Video Processing"])


# 根路径跳转到测试页面
@app.get("/")
def read_root():
    return RedirectResponse(url="/static/test.html")
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to Grounded SAM2 API"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

