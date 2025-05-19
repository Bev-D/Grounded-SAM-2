# task_manager.py

import os
import uuid
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any

# 结果根目录
TASK_ROOT_DIR = "results"
# 默认清理时间（秒），例如：3600 秒 = 1 小时
DEFAULT_MAX_AGE = 3600

def generate_client_id() -> str:
    """生成唯一的 client_id"""
    return str(uuid.uuid4())

def create_task_dir(client_id: str) -> str:
    """为指定 client_id 创建任务目录"""
    task_dir = os.path.join(TASK_ROOT_DIR, client_id)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir

def save_result_files(task_dir: str, result: dict) -> None:
    """保存检测结果文件（图像 + JSON）"""
    # 复制图像
    if os.path.exists(result["mask_image_path"]):
        shutil.copy2(result["mask_image_path"], os.path.join(task_dir, "mask.jpg"))
    if os.path.exists(result["annotated_image_path"]):
        shutil.copy2(result["annotated_image_path"], os.path.join(task_dir, "annotated.jpg"))

    # 保存 JSON 结果
    json_path = os.path.join(task_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "result": result["annotations"],
            "image_width": result["img_width"],
            "image_height": result["img_height"]
        }, f, indent=2)

def get_result_urls(client_id: str) -> Dict[str, str]:
    """获取任务结果的 URL 路径"""
    base_url = f"/results/{client_id}"
    return {
        "mask_image_url": f"{base_url}/mask.jpg",
        "annotated_image_url": f"{base_url}/annotated.jpg",
        "json_result_url": f"{base_url}/results.json"
    }

def cleanup_old_tasks(max_age: int = DEFAULT_MAX_AGE) -> None:
    """清理超过 max_age 秒的旧任务"""
    now = datetime.now()
    for entry in os.listdir(TASK_ROOT_DIR):
        task_path = os.path.join(TASK_ROOT_DIR, entry)
        if not os.path.isdir(task_path):
            continue
        try:
            # 获取最后修改时间
            mtime = os.path.getmtime(task_path)
            last_modified = datetime.fromtimestamp(mtime)
            if (now - last_modified).total_seconds() > max_age:
                print(f"清理任务: {entry}")
                shutil.rmtree(task_path)
        except Exception as e:
            print(f"清理失败: {entry}, 错误: {e}")

def process_and_save_result(result: dict, client_id: str = None) -> Dict[str, Any]:
    """
    综合处理函数：
    1. 生成 client_id（如果未提供）
    2. 创建任务目录
    3. 保存图像和 JSON
    4. 返回可访问的 URL
    """
    if not os.path.exists(TASK_ROOT_DIR):
        os.makedirs(TASK_ROOT_DIR)

    if not client_id:
        client_id = generate_client_id()

    task_dir = create_task_dir(client_id)
    save_result_files(task_dir, result)
    # urls = get_result_urls(client_id)

    return {
        "client_id": client_id,
        # **urls
    }
