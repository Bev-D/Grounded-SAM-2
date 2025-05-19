import os

def convert_local_to_url_path(local_path: str) -> str:
    """
    将本地文件路径转换为 FastAPI 可访问的 URL 路径。
    
    示例：
        "results\\A01607\\A01607_mask.jpg" 
        → "/results/A01607/A01607_mask.jpg"
        
        "C:\\Users\\xxx\\results\\A01607\\A01607_mask.jpg"
        → "/results/A01607/A01607_mask.jpg"
    """
    # 分割路径各部分
    parts = []
    while True:
        head, tail = os.path.split(local_path)
        if tail:
            parts.append(tail)
        else:
            if head:
                parts.append(head)
            break
        local_path = head

    # 去掉盘符（如 C:\），只保留目录和文件名
    if ":" in parts[-1]:  # Windows 盘符检测
        parts = parts[:-1]

    # 反转以得到根到文件的顺序
    parts = parts[::-1]

    # 构建 URL 路径
    return "/" + "/".join(parts)


def get_relative_url_path(local_path: str, base_dir: str) -> str:
    """
    获取相对于某个基础目录的 URL 路径。
    
    示例：
        local_path = "C:\\project\\results\\A01607\\A01607_mask.jpg"
        base_dir = "C:\\project\\results"
        → "/A01607/A01607_mask.jpg"
    """
    try:
        relative = os.path.relpath(local_path, base_dir)
        return "/" + relative.replace("\\", "/")
    except ValueError:
        # Windows 下不同盘符时会报错
        return convert_local_to_url_path(local_path)


# 如果需要可以扩展更多函数，例如：
def ensure_url_starts_with(prefix="/results"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            path = func(*args, **kwargs)
            if not path.startswith(prefix):
                return f"{prefix}{path}"
            return path
        return wrapper
    return decorator


@ensure_url_starts_with("/results")
def normalize_mask_path(path: str) -> str:
    return convert_local_to_url_path(path)
