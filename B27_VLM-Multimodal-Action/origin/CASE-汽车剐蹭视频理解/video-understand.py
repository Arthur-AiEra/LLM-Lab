"""
属于 InternVL 2.5系列
视频理解与生成：可以用于视频内容的分析、总结和生成相关的文本描述。
视觉问答：能够回答与图像或视频内容相关的问题。
多模态对话：支持与用户进行包含视觉信息的对话。
"""


# In[2]:


# 模型下载，需要下载3个大模型
from modelscope import snapshot_download

model_dir = snapshot_download('OpenGVLab/InternVideo2_5_Chat_8B', cache_dir='/root/autodl-tmp/models')
# model_dir = snapshot_download('internlm/internlm2_5-7b-chat', cache_dir='/root/autodl-tmp/models')
# model_dir = snapshot_download('LLM-Research/Mistral-7B-Instruct-v0.3', cache_dir='/root/autodl-tmp/models')
# model_dir = snapshot_download('AI-ModelScope/bert-base-uncased', cache_dir='/root/autodl-tmp/models')


# In[1]:


# 导入必要的库
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer


# 模型配置
model_path = '/root/autodl-tmp/models/OpenGVLab/InternVideo2_5_Chat_8B'

# 初始化分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)

# ImageNet 数据集的均值和标准差
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    构建图像转换pipeline
    
    参数:
        input_size: 输入图像大小
    
    返回:
        transform: 转换pipeline
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
        T.ToTensor(), 
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    寻找最接近原始图像宽高比的目标比例
    
    参数:
        aspect_ratio: 原始图像的宽高比
        target_ratios: 目标比例列表
        width: 原始图像宽度
        height: 原始图像高度
        image_size: 目标图像大小
        
    返回:
        best_ratio: 最佳比例
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    动态预处理图像，根据宽高比将图像分割成多个块
    
    参数:
        image: 原始图像
        min_num: 最小块数
        max_num: 最大块数
        image_size: 目标图像大小
        use_thumbnail: 是否使用缩略图
        
    返回:
        processed_images: 处理后的图像列表
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算现有图像宽高比
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 寻找最接近目标的宽高比
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, 
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # 分割图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    """
    加载并处理图像
    
    参数:
        image: 输入图像
        input_size: 输入大小
        max_num: 最大块数
        
    返回:
        pixel_values: 处理后的图像张量
    """
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    获取视频帧索引
    
    参数:
        bound: 时间边界 [开始时间, 结束时间]
        fps: 视频帧率
        max_frame: 最大帧数
        first_idx: 第一帧索引
        num_segments: 分段数量
        
    返回:
        frame_indices: 帧索引数组
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
    """
    根据视频时长计算帧数
    
    参数:
        duration: 视频时长（秒）
        
    返回:
        num_frames: 计算出的帧数
    """
    local_num_frames = 4        
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments
    
    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)

    return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    """
    加载并处理视频
    
    参数:
        video_path: 视频路径
        bound: 时间边界
        input_size: 输入大小
        max_num: 最大块数
        num_segments: 分段数量
        get_frame_by_duration: 是否根据时长获取帧数
        
    返回:
        pixel_values: 处理后的视频帧张量
        num_patches_list: 每帧的块数列表
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# 评估设置
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)
video_path = "car.mp4"
num_segments=128


with torch.no_grad():
  # 加载视频并处理
  pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
  pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
  video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
  
  # 单轮对话：视频详细描述
  question1 = "Describe this video in detail."
  question = video_prefix + question1
  output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
  print(output1)
  
  # 多轮对话：询问视频中的人数
  question2 = "How many people appear in the video?"
  output2, chat_history = model.chat(tokenizer, pixel_values, question2, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
  
  print(output2)


# In[4]:


# video_prefix


# In[3]:


with torch.no_grad():
  # 单轮对话：询问车辆损伤部位（中文）
  question1 = "车的哪个部位损伤了？"
  question = video_prefix + question1
  output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
  print(output1)
  
  # 多轮对话：询问车辆碰撞位置（中文）
  question2 = "车撞到哪里了？"
  output2, chat_history = model.chat(tokenizer, pixel_values, question2, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
  
  print(output2)

