import json
import shutil
import os

import requests
from modelscope import snapshot_download


def download_json(url):
    # 下载JSON文件
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        try:
            data = json.load(open(local_filename))
        except (json.JSONDecodeError, FileNotFoundError):
            data = download_json(url)

        config_version = data.get('config_version', '0.0.0')
        # 如果版本过低，则重新下载最新模板
        if config_version < '1.3.1':
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    mineru_patterns = [
        # "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
        # "models/TabRec/TableMaster/*",
        # "models/TabRec/StructEqTable/*",
    ]
    # 指定模型保存目录
    local_model_dir = "/private/var/ifc/app_data/autodl-tmp/models/modelscope_models"

    # 确保目录存在
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir, exist_ok=True)

    model_dir = snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0', allow_patterns=mineru_patterns, local_dir=local_model_dir)
    layoutreader_model_dir = snapshot_download('ppaanngggg/layoutreader', local_dir=local_model_dir)

    # 根据 MinerU 结构调整 model_dir
    # 注意：PDF-Extract-Kit-1.0 仓库内包含 models 文件夹
    model_dir_with_models = os.path.join(model_dir, 'models')
    print(f'model_dir is: {model_dir_with_models}')
    print(f'layoutreader_model_dir is: {layoutreader_model_dir}')

    # 更新为最新的 MinerU 模板 URL 和配置文件名
    # 原 magic-pdf.template.json 已更名为 mineru.template.json
    json_url = 'https://raw.githubusercontent.com/opendatalab/MinerU/master/mineru.template.json'
    config_file_name = 'magic-pdf.json'
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    # 适配新的配置结构 (models-dir 现在是字典)
    json_mods = {
        'models-dir': {
            'pipeline': model_dir_with_models,
            'vlm': model_dir_with_models
        },
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f'The configuration file has been configured successfully, the path is: {config_file}')
