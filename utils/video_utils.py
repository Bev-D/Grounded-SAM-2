import cv2
import os
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")

def create_video_from_image_array(images, output_video_path, frame_rate=25):
    """
    从传入的图像数组创建视频。

    参数:
        images (list of np.ndarray): 包含图像帧的数组，每个元素为OpenCV格式的图像。
        output_video_path (str): 输出视频文件的路径。
        frame_rate (int): 视频帧率，默认为25。
    """
    if not images:
        raise ValueError("图像数组为空，无法生成视频。")

    # 获取第一帧以确定视频尺寸
    height, width, _ = images[0].shape

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # 将每帧写入视频
    for image in tqdm(images):
        video_writer.write(image)

    # 释放资源
    video_writer.release()
    print(f"视频已保存至 {output_video_path}")
