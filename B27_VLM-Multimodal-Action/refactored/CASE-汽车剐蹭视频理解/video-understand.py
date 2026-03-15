"""
InternVideo2.5 视频理解脚本 - macOS MPS 深度优化版 (V3)
主要修复：
1. 针对 [mutex.cc : 452] RAW: Lock blocking 死锁的终极尝试。
2. 采用“CPU 加载 -> 转移至 MPS”的保守策略，避开加载时的 MPS 分配死锁。
3. 增加 MPS 容错环境变量 PYTORCH_ENABLE_MPS_FALLBACK=1。
4. 显式执行垃圾回收和 MPS 缓存清理。
"""

#t 98:24 99:39 https://gemini.google.com/app/a2010c5d9a45156a

import os

# ========== 阶段 0: 环境变量 (必须在任何 import 之前) ==========
# 1. 禁用 Tokenizer 并行
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. 限制底层线程
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 3. 允许重复库加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 4. MPS 内存管理优化
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# 5. 【新增】启用 MPS 算子回退到 CPU，防止某些算子挂起
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ========== 阶段 1: 导入库 ==========
import numpy as np
import torch
import torchvision.transforms as T
# 移动 cv2 导入到模型加载后，避免库冲突
# import cv2
# cv2.setNumThreads(0)

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
from transformers import PretrainedConfig

# ========== 阶段 2: 核心补丁 (Monkey Patch) ==========
_original_to_json_string = PretrainedConfig.to_json_string
def _safe_to_json_string(self, *args, **kwargs):
    try:
        return _original_to_json_string(self, *args, **kwargs)
    except Exception:
        return '{\n  "info": "Config logging bypassed to avoid serialization bug."\n}\n'
PretrainedConfig.to_json_string = _safe_to_json_string

# ========== 阶段 3: 模型加载策略 ==========
# 模型配置 https://www.modelscope.cn/models/OpenGVLab/InternVideo2_5_Chat_8B
model_path = '/private/var/ifc/app_data/autodl-tmp/models/OpenGVLab/InternVideo2_5_Chat_8B'

# 动态检测设备
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16  # Mac MPS 推荐使用 float16 (bfloat16支持不佳)
else:
    device = "cpu"
    dtype = torch.float32

print(f"🚀 运行设备: {device}, 精度: {dtype}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("开始加载模型权重 ...")


# V3 策略：先在 CPU 上加载模型，不使用 low_cpu_mem_usage 以简化加载逻辑
# 如果您的内存（RAM）小于 32GB，请注意可能会 OOM。
# 但为了解决死锁，这是目前最稳妥的尝试。
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,  # <--- 禁用此参数，防止 accelerate 引入的复杂线程锁
    device_map=None,          # <--- 强制先加载到 CPU
    attn_implementation="eager"  # <--- 禁用 flash_attention，防止在 CPU 上挂起
).eval()

print("模型权重已加载至 CPU，正在转移至 GPU/MPS (这通常是死锁的高发点，请观察)...")
try:
    if device == "mps":
        # 分步转移，尝试缓解压力
        model = model.to("mps")
        torch.mps.empty_cache()
    else:
        model = model.to(device)
    print("✅ 模型加载并转移成功！")
except Exception as e:
    print(f"❌ 转移至设备时出错: {e}")
    print("尝试保留在 CPU 运行（速度会很慢）...")
    device = "cpu"
    model = model.to("cpu")

# 现在导入 cv2，避免与 modelscope 的库冲突
import cv2
cv2.setNumThreads(0)

# ========== 阶段 4: 图像/视频处理工具 ==========
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    start, end = (bound[0], bound[1]) if bound else (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, num_segments=num_segments)

    pixel_values_list = []
    num_patches_list = []

    for idx, frame_index in enumerate(frame_indices):
        print(f"⏳ 正在处理第 {idx + 1}/{len(frame_indices)} 帧 (索引: {frame_index})...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])

        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    cap.release()
    if not pixel_values_list:
        raise ValueError("未能成功加载任何视频帧。")

    return torch.cat(pixel_values_list), num_patches_list

# ========== 阶段 5: 执行推理 ==========
video_path = "car.mp4"
num_segments = 128
generation_config = dict(do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1)

if not os.path.exists(video_path):
    print(f"❌ 错误: 未找到视频文件 '{video_path}'。")
else:
    with torch.no_grad():
        print("🎬 正在加载并处理视频...")
        pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1)
        pixel_values = pixel_values.to(dtype).to(device)

        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

        print("\n--- 任务 1: 详细描述 ---")
        question1 = video_prefix + "Describe this video in detail."
        output1, chat_history = model.chat(tokenizer, pixel_values, question1, generation_config,
                                         num_patches_list=num_patches_list, history=None, return_history=True)
        print(f"回答: {output1}")

        print("\n--- 任务 2: 损伤分析 ---")
        question2 = "车的哪个部位损伤了？"
        output2, _ = model.chat(tokenizer, pixel_values, question2, generation_config,
                               num_patches_list=num_patches_list, history=chat_history, return_history=True)
        print(f"回答: {output2}")
