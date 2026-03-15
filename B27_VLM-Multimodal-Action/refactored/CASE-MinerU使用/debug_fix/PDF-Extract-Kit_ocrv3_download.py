import urllib.request, os

print('正在启动外科手术式精准下载...')

# 指向报错日志里明确说找不到的那个目录
target_dir = '/private/var/ifc/app_data/autodl-tmp/models/modelscope_models/models/OCR/paddleocr_torch'
os.makedirs(target_dir, exist_ok=True)

# 强制从 HuggingFace 的历史稳定 Commit 节点获取，不受官方删库影响
urls = {
    'ch_PP-OCRv3_det_infer.pth': 'https://hf-mirror.com/opendatalab/PDF-Extract-Kit-1.0/resolve/782e787d46ed9b52253af6c1f69cdfcc76583e8d/models/OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth',
    'ch_PP-OCRv4_rec_infer.pth': 'https://hf-mirror.com/opendatalab/PDF-Extract-Kit-1.0/resolve/782e787d46ed9b52253af6c1f69cdfcc76583e8d/models/OCR/paddleocr_torch/ch_PP-OCRv4_rec_infer.pth',
    'ch_PP-OCRv3_rec_infer.pth': 'https://hf-mirror.com/opendatalab/PDF-Extract-Kit-1.0/resolve/782e787d46ed9b52253af6c1f69cdfcc76583e8d/models/OCR/paddleocr_torch/ch_PP-OCRv3_rec_infer.pth'
}

# 伪装请求头以防被镜像站拦截
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')]
urllib.request.install_opener(opener)

for name, url in urls.items():
    path = os.path.join(target_dir, name)
    if not os.path.exists(path):
        print(f'正在精准植入 {name} ... (视网速可能需要几分钟)')
        try:
            urllib.request.urlretrieve(url, path)
            print(f'✅ {name} 植入成功！')
        except Exception as e:
            # 即使部分旧权重404也不影响核心的det文件获取
            print(f'⚠️ {name} 遇到问题: {e}')
    else:
        print(f'✅ {name} 已就位。')

print('所有前置地雷清扫完毕！')