# file_utils.py

import os
import requests
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import tempfile
import shutil
from tqdm import tqdm
import hashlib
import mimetypes

# 默认下载目录
DEFAULT_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "file_utils_cache"
os.makedirs(DEFAULT_DOWNLOAD_DIR, exist_ok=True)






def get_url_hash(url: str) -> str:
    """
    对 URL 进行哈希计算，用于生成唯一的文件名。
    """
    return hashlib.md5(url.encode()).hexdigest()


def is_url(path_or_url: str) -> bool:
    """
    判断输入是否为 URL。
    """
    parsed = urlparse(path_or_url)
    return parsed.scheme in ("http", "https")


def download_file(url: str, filename: str = None, download_dir: str = None) -> str:
    """
    下载远程文件到本地目录，并显示进度条。

    :param url: 文件的URL地址
    :param filename: 自定义保存的文件名（含扩展名），默认从URL中提取
    :param download_dir: 本地存储目录，默认使用临时缓存目录
    :return: 本地文件路径
    """

    download_dir = Path(download_dir or DEFAULT_DOWNLOAD_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    content_type = response.headers.get("Content-Type")

    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1]  # 如 .jpg, .png

    # 根据 Content-Type 映射扩展名
    # ext = mimetypes.guess_extension(content_type) if content_type else None

    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp",".mp4",".avi"}

    if not ext or not ext.startswith(".") or len(ext) > 5 or ext.lower() not in allowed_extensions:
        print(f"非法扩展名: {ext}, 终止下载")
        return None  # 不合法时提前返回空字符串

    # 如果没有指定文件名，则用 URL 哈希 + 扩展名
    if not filename:
        hash_name = f"{get_url_hash(url)}{ext}"
        filename = hash_name


    # # 获取原始扩展名（可选）
    # parsed = urlparse(url)
    # ext = os.path.splitext(parsed.path)[1]  # 如 .jpg, .png
    # local_path = os.path.join(download_dir, f"{filename}{ext}")

    local_path = Path(os.path.join(download_dir, filename))  

    # 如果文件已存在，直接返回路径
    if local_path.exists():
        print(f"文件已存在，跳过下载：{local_path}")
        return str(local_path)

    # 开始下载
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))
    block_size = 1024

    with open(local_path, "wb") as f, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit="iB",
            unit_scale=True,
    ) as t:
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            t.update(len(chunk))

    return str(local_path)


def ensure_local_file(path_or_url: str, download_dir: str = None) -> str:
    """
    确保输入路径是本地有效文件路径。

    - 如果是 URL，则下载到本地并返回本地路径。
    - 如果是本地路径且存在，则直接返回该路径。
    """
    if is_url(path_or_url):
        return download_file(path_or_url, download_dir=download_dir)
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"本地文件不存在: {path_or_url}")
        return path_or_url


def load_image_from_url(url: str) -> Image.Image:
    """
    从 URL 加载图像并返回 PIL.Image 对象。
    """
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def save_output(data, output_path: str):
    """
    保存输出数据到指定路径。

    :param data: 要保存的数据（如图像、字节流或字符串）
    :param output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, Image.Image):
        data.save(output_path)
    elif isinstance(data, bytes):
        with open(output_path, "wb") as f:
            f.write(data)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(data))


def cleanup(download_dir: str = None):
    """
    清理指定目录中的所有文件。

    :param download_dir: 要清理的目录，默认使用默认缓存目录
    """
    dir_to_clean = Path(download_dir or DEFAULT_DOWNLOAD_DIR)
    if dir_to_clean.exists():
        shutil.rmtree(dir_to_clean)
        os.makedirs(dir_to_clean)  # 重建空目录


def get_temp_path(suffix: str = ".tmp") -> str:
    """
    获取一个临时文件路径。

    :param suffix: 文件后缀
    :return: 临时文件路径
    """
    return str(Path(tempfile.NamedTemporaryFile(suffix=suffix).name))


def process_input_image(image: str):
    # 去除前后空格并统一格式
    image = image.strip()

    # 判断是否为视频文件或单张图片文件
    if os.path.isfile(image):
        file_ext = os.path.splitext(image)[1].lower()
        if file_ext in ['.mp4', '.avi', '.mov']:  # 支持的视频格式
            print(f"✅ 输入为视频文件：{image}")
            file_name_list = None  # 视频无需预先处理文件名列表
            return "video", file_name_list
        elif file_ext in ['.jpg', '.jpeg', '.png']:  # 单张图片
            print(f"✅ 输入为单张图片：{image}")
            file_name_list = [os.path.basename(image)]
            return "image_list", file_name_list
        else:
            raise ValueError("❌ 不支持的文件格式")

    # 判断是否为多张图像文件的逗号分隔字符串
    elif ',' in image:
        file_list = [f.strip() for f in image.split(',')]
        supported_extensions = {'.jpg', '.jpeg', '.png'}

        for f in file_list:
            ext = os.path.splitext(f)[1].lower()
            if ext not in supported_extensions:
                raise ValueError(f"❌ 文件 {f} 格式不支持，仅支持 .jpg, .jpeg, .png")

        print(f"✅ 输入为多张图片列表：{file_list}")
        file_name_list = [os.path.basename(f) for f in file_list]
        return "image_list", file_name_list

    else:
        raise ValueError("❌ 无法识别的输入格式")