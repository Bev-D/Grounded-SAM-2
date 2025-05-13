# app.py
from flask import Flask, render_template_string, request, jsonify
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 临时存储上传文件的目录
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# API 地址（你的 FastAPI 后端）
API_URL = "http://localhost:8000/process-video"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>视频处理测试页面 - Flask</title>
  <style>
    body { font-family: Arial; padding: 20px; }
    input, select, button { margin: 10px 0; display: block; width: 300px; }
    pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
    .path-display {
      background-color: #eef6ff;
      padding: 10px;
      border-radius: 5px;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <h2>启动视频处理任务</h2>

  <form method="POST" enctype="multipart/form-data">
    <label>Video Path:</label>
    <input type="text" name="video_path" id="videoPathInput" value="./assets/Rec_0007.mp4" />

    <label>或选择本地视频文件:</label>
    <input type="file" name="video_file" id="videoFileInput" accept="video/*" />

    <label>Text Prompt:</label>
    <input type="text" name="text_prompt" value="car." />

    <label>Prompt Type:</label>
    <select name="prompt_type">
      <option value="point">Point</option>
      <option value="box" selected>Box</option>
      <option value="mask">Mask</option>
    </select>

    <button type="submit">开始处理</button>
  </form>

  {% if video_path_display %}
  <div class="path-display">
    <strong>当前处理的视频路径：</strong>
    <pre>{{ video_path_display }}</pre>
  </div>
  {% endif %}

  {% if result %}
  <h3>响应结果：</h3>
  <pre>{{ result }}</pre>
  {% endif %}

  <script>
    document.getElementById('videoFileInput').addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
          // 显示文件名而不是完整路径（更安全和兼容）
          document.getElementById('videoPathInput').value = file.name;
        }
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    video_path_display = ""  # 初始化为空字符串，确保所有路径都能访问到该变量

    if request.method == "POST":
        video_path = request.form.get("video_path")
        video_file = request.files.get("video_file")  # 获取上传的文件
        text_prompt = request.form.get("text_prompt")
        prompt_type = request.form.get("prompt_type")

        # 如果用户上传了视频文件，则保存并覆盖 video_path
        if video_file and video_file.filename:
            filename = secure_filename(video_file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            video_file.save(save_path)

            # 👇 使用绝对路径
            video_path = os.path.abspath(save_path)

        if not video_path:
            result = {"error": "未提供视频路径"}
        else:
            video_path_display = video_path  # 设置要传给前端显示的路径
            payload = {
                "video_path": video_path,
                "text_prompt": text_prompt,
                "prompt_type": prompt_type
            }

            try:
                response = requests.post(API_URL, json=payload)
                result = response.json()
            except Exception as e:
                result = {"error": str(e)}                

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        video_path_display=video_path_display
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
