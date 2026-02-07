#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# ## 使用 Qwen 大模型进行垃圾邮件分类（无需微调）
# 使用 ModelScope 下载模型，然后用 HuggingFace 加载和使用

# In[1]:


import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from modelscope import snapshot_download  # 使用 ModelScope 下载模型


# ## 使用 ModelScope 下载模型，然后用 HuggingFace 加载

# In[2]:


# Step 1: 使用 ModelScope 从魔搭社区下载 Qwen3-0.6B 模型
# ModelScope 是下载工具，HuggingFace 是加载工具
modelscope_model_id = "Qwen/Qwen2.5-7B-Instruct"  # ModelScope 上的模型 ID
cache_dir = "/root/autodl-tmp/models"  # 指定模型下载和缓存的文件夹

print(f"正在从 ModelScope 下载模型: {modelscope_model_id}")
print(f"模型将保存到: {cache_dir}")
print("提示：首次运行会下载模型，请耐心等待...")

try:
    # 从 ModelScope 下载模型文件到指定文件夹
    model_dir = snapshot_download(
        modelscope_model_id,
        cache_dir=cache_dir  # 指定下载文件夹
    )
    print(f"模型已下载到: {model_dir}")
except Exception as e:
    print(f"ModelScope 下载失败: {e}")
    print("提示：请确保已安装 modelscope: pip install modelscope")
    raise


# In[2]:


# Step 1: 使用 HuggingFace 加载本地模型
print("\n正在使用 HuggingFace 加载模型...")

try:
    # 使用本地路径加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # 检查是否有 GPU
    if torch.cuda.is_available():
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,  # 使用本地路径
            torch_dtype=torch.float16,  # 使用半精度节省显存
            device_map="auto",  # 自动分配设备
            trust_remote_code=True
        )
    else:
        print("未检测到 GPU，使用 CPU（速度较慢）")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,  # 使用本地路径
            trust_remote_code=True
        )
    print("模型加载完成！")
except Exception as e:
    print(f"模型加载失败: {e}")
    raise


# In[4]:


# ## Step2：定义分类函数（使用 Prompt）

def classify_spam_with_prompt(text, model, tokenizer):
    """
    使用 Prompt 让大模型进行分类
    """
    # 构建分类提示词（使用更清晰的格式）
    prompt = f"""你是一个垃圾邮件分类专家。请判断以下文本是否为垃圾邮件。

文本：{text}

请只回答"垃圾邮件"或"正常邮件"："""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,  # 生成少量token即可
            do_sample=False,    # 使用贪心解码，保证结果稳定
            pad_token_id=tokenizer.eos_token_id,  # 设置pad token
        )
    
    # 解码输出（只取新生成的部分）
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    # 解析结果
    generated_text_lower = generated_text.lower()
    #print('generated_text_lower=', generated_text_lower)
    if "垃圾" in generated_text or "spam" in generated_text_lower:
        return 1, "垃圾邮件"
    elif "正常" in generated_text or "normal" in generated_text_lower or "非垃圾" in generated_text:
        return 0, "正常邮件"
    else:
        # 如果模型输出不符合预期，返回原始输出用于调试
        return None, f"未识别: {generated_text}"


# ## Step3：使用 Pipeline 方式（更简单）
def classify_spam_with_pipeline(text, generator):
    """
    使用 Pipeline 进行分类（更简单的方式）
    """
    prompt = f"""你是一个垃圾邮件分类专家。请判断以下文本是否为垃圾邮件。

文本：{text}

请只回答"垃圾邮件"或"正常邮件"："""
    
    # 使用 pipeline 生成
    try:
        result = generator(
            prompt,
            max_new_tokens=15,
            do_sample=False,
            temperature=0.1,
            return_full_text=False,  # 只返回新生成的部分
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        
        generated_text = result[0]['generated_text'].strip()
        
        # 解析结果
        generated_text_lower = generated_text.lower()
        #print('generated_text_lower=', generated_text_lower)
        if "垃圾" in generated_text or "spam" in generated_text_lower:
            return 1, "垃圾邮件"
        elif "正常" in generated_text or "normal" in generated_text_lower or "非垃圾" in generated_text:
            return 0, "正常邮件"
        else:
            return None, f"未识别: {generated_text}"
    except Exception as e:
        return None, f"错误: {str(e)}"


# ## Step4：测试分类效果
# 准备测试数据
test_texts = [
    "今晚有空一起吃饭吗？",           # 正常
    "恭喜您获得500万大奖，点击领取",   # 垃圾
    "您的验证码是1234，请勿泄露",      # 正常
    "澳门首家线上赌场上线啦",          # 垃圾
    "项目进度怎么样了？需不需要开会",   # 正常
    "独家内幕消息，股票必涨，加群",     # 垃圾
    "低息贷款，无抵押，秒到账",        # 垃圾
]

print("=" * 60)
print("方法1：直接使用模型 + Prompt")
print("=" * 60)

for text in test_texts:
    label_id, label_name = classify_spam_with_prompt(text, model, tokenizer)
    print(f"文本: {text}")
    print(f"预测: {label_name} (ID: {label_id})")
    print("-" * 60)


# In[7]:


print("\n" + "=" * 60)
print("方法2：使用 Pipeline（推荐，更简单）")
print("=" * 60)

# 创建 text-generation pipeline
# 注意：如果模型使用了 device_map="auto"，模型已经通过 accelerate 分配到设备
# 此时不能指定 device 参数，Pipeline 会自动检测模型所在的设备
try:
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
        # 不指定 device，让 Pipeline 自动检测模型所在的设备
    )
    print("Pipeline 创建成功！")
except Exception as e:
    print(f"Pipeline 创建失败: {e}")
    generator = None

if generator is not None:
    for text in test_texts:
        label_id, label_name = classify_spam_with_pipeline(text, generator)
        print(f"文本: {text}")
        print(f"预测: {label_name} (ID: {label_id})")
        print("-" * 60)
else:
    print("Pipeline 未创建，跳过 Pipeline 方式测试")


# In[8]:


# ## Step5：批量处理示例

def batch_classify(texts, generator):
    """
    批量分类（使用 Pipeline）
    """
    if generator is None:
        print("Pipeline 未创建，无法批量处理")
        return []
    
    results = []
    for text in texts:
        label_id, label_name = classify_spam_with_pipeline(text, generator)
        results.append({
            "text": text,
            "label": label_id,
            "label_name": label_name
        })
    return results

# 批量处理
print("\n" + "=" * 60)
print("批量处理示例")
print("=" * 60)

if generator is not None:
    batch_results = batch_classify(test_texts, generator)
    for result in batch_results:
        print(f"{result['label_name']}: {result['text']}")
else:
    print("无法执行批量处理（Pipeline 未创建）")

