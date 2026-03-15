import pandas as pd
import os

# ================= 配置区域 =================
# 1. 原始 Excel 文件路径
INPUT_EXCEL = './prompt_template_cn.xlsx'

# 2. 转换后生成的新 Excel 文件路径
OUTPUT_EXCEL = './prompt_template_cn_local.xlsx'

# 3. 你的本地图片存放的文件夹目录（请改成你电脑上实际放图片的路径）
# 例如: 'C:/Users/YourName/Desktop/images/' 或者相对路径 './images/'
BASE_IMAGE_DIR = './'


# ============================================

def format_local_path(img_name):
    """
    辅助函数：将单个文件名格式化为本地路径，并自动补全缺失的后缀
    """
    img_name = img_name.strip()
    # 如果原文件名没有图片后缀，自动补上 .jpg (根据你之前代码的逻辑)
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_name += '.jpg'

    # 拼接本地目录和文件名，统一使用正斜杠
    local_path = os.path.join(BASE_IMAGE_DIR, img_name).replace('\\', '/')
    return local_path


def update_image_column(cell_value):
    """
    处理 Excel 中 image 列的每一个单元格
    """
    if pd.isna(cell_value):
        return cell_value

    cell_str = str(cell_value).strip()

    # 场景1: 处理列表中有多个图片的情况，例如 "[car1, car2]"
    if cell_str.startswith('[') and ',' in cell_str:
        # 去掉前后的中括号，按逗号拆分
        names = cell_str[1:-1].split(',')
        updated_paths = [format_local_path(name) for name in names]
        # 重新组装回 "[path1, path2]" 的格式
        return f"[{', '.join(updated_paths)}]"

    # 场景2: 处理单个图片的情况
    else:
        return format_local_path(cell_str)


def main():
    print(f"⏳ 正在读取 Excel 文件: {INPUT_EXCEL} ...")
    try:
        df = pd.read_excel(INPUT_EXCEL)
    except Exception as e:
        print(f"❌ 读取文件失败，请检查文件是否被占用或路径错误: {e}")
        return

    if 'image' not in df.columns:
        print("❌ 错误：在 Excel 中没有找到名为 'image' 的列，请检查表头名！")
        return

    # 应用批量转换
    df['image'] = df['image'].apply(update_image_column)

    print("\n✅ 路径转换成功！前 5 条路径预览：")
    print(df['image'].head().to_string())

    # 保存为新文件
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n🎉 转换完成！新文件已保存为: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()