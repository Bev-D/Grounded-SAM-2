import uvicorn
import threading
from fastapi import FastAPI,Request, HTTPException
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.utils.model_manager import ModelManager
from grounded_sam_processor import GroundedSAMProcessor
from api.routers import image_process_router,video_process_router,imagelist_process_router
app = FastAPI(title="Grounded SAM2 Image Processing API")


# # 全局锁 + 存储模型实例的字典
# processor_lock = threading.Lock()
# session_processors = {}  # {session_id: processor}
# def get_session_processor(request: Request):
#     session_id = request.cookies.get("session_id")
#     if not session_id:
#         raise HTTPException(status_code=400, detail="Missing session_id cookie")
# 
#     with processor_lock:
#         if session_id not in session_processors:
#             session_processors[session_id] = GroundedSAMProcessor(device="cuda")
#         return session_processors[session_id]

# # 初始化模型处理器（只加载一次）
# processor = GroundedSAMProcessor(device="cuda")  # 或 "cpu"
# 
# # 挂载到 app.state，供路由访问
# app.state.processor = processor

# 挂载静态资源目录
app.mount("/static", StaticFiles(directory="api/static", html=True), name="static")

# 挂载图像结果目录
app.mount("/results", StaticFiles(directory="results"), name="results")


# 注册图像处理路由
app.include_router(image_process_router.router, prefix="/api/v1/image", tags=["Image Processing"])

# 注册视频处理路由
app.include_router(video_process_router.router, prefix="/api/v1/video", tags=["Video Processing"])

# 注册图片集处理路由
app.include_router(imagelist_process_router.router, prefix="/api/v1/imagelist", tags=["Imagelist Processing"])
# 根路径跳转到测试页面
@app.get("/")
def read_root():
    return RedirectResponse(url="/static/mytest.html")

# 在服务启动时预加载模型（可选）
@app.on_event("startup")
async def startup_event():
    print("🚀 正在预加载模型...")
    try:
        ModelManager.get_sam2_model()
        ModelManager.get_grounding_model()
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")

# 提供卸载模型接口（用于调试或资源释放）
@app.post("/api/v1/unload-model", tags=["Model Management"])
def unload_models():
    """
    卸载当前加载的模型，释放显存。
    """
    try:
        ModelManager.unload_models()
        return {"status": "success", "message": "模型已成功卸载"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型卸载失败: {str(e)}")


# @app.get("/static/apitest.html", response_class=HTMLResponse)
# def load_model_on_apitest(request: Request):
#     # 生成或获取 session_id（这里简化处理）
#     session_id = "default_user"  # 可替换为 token 或 UUID
#     with processor_lock:
#         if session_id not in session_processors:
#             session_processors[session_id] = GroundedSAMProcessor(device="cuda")
# 
#     # 设置 session_id 到 Cookie
#     response = HTMLResponse(content=open("api/static/apitest.html").read())
#     response.set_cookie(key="session_id", value=session_id)
#     return response


# @app.post("/release-model")
# def release_model(request: Request):
#     session_id = request.cookies.get("session_id")
#     if not session_id:
#         raise HTTPException(status_code=400, detail="No session_id in cookies")
# 
#     with processor_lock:
#         if session_id in session_processors:
#             del session_processors[session_id]
#     return {"status": "Model released"}
# @app.on_event("shutdown")
# def shutdown_event():
#     with processor_lock:
#         session_processors.clear()
#     print("所有模型已释放")


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to Grounded SAM2 API"}






if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

# uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
# http://localhost:8000/static/mytest.html
