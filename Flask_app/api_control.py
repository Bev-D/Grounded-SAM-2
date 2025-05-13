import subprocess
import os

PID_FILE = "api_server.pid"

def start_api():
    if os.path.exists(PID_FILE):
        print("服务已经在运行中！")
        return

    # 启动 uvicorn 并记录 PID
    process = subprocess.Popen([
        "uvicorn", "process_video_APITest:app", "--host", "0.0.0.0", "--port", "8000"
    ])
    
    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))

    print(f"✅ 服务已启动，进程 PID: {process.pid}")


def stop_api():
    if not os.path.exists(PID_FILE):
        print("服务未运行。")
        return

    with open(PID_FILE, "r") as f:
        pid = f.read().strip()

    try:
        # 终止进程
        subprocess.run(["taskkill", "/F", "/PID", pid], check=True)
        print(f"🛑 服务已终止，PID: {pid}")
    except subprocess.CalledProcessError:
        print("❌ 无法终止服务，请手动检查任务管理器。")

    os.remove(PID_FILE)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ["start", "stop"]:
        print("用法: python api_control.py [start|stop]")
        sys.exit(1)

    if sys.argv[1] == "start":
        start_api()
    elif sys.argv[1] == "stop":
        stop_api()
