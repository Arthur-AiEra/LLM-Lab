import cv2
import numpy as np
from ultralytics import YOLO
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def annotate_image_with_yolov11(image_path, model_path='yolo11n.pt'):
    """
    使用YOLOv11对图片进行标注
    :param image_path: 输入图片路径
    :param model_path: YOLOv11模型路径，默认使用yolo11n.pt
    :return: 标注后的图片和检测结果
    """
    # 加载YOLOv11模型
    try:
        model = YOLO(model_path)
    except:
        # 如果没有预训练模型，则下载
        print(f"正在下载 {model_path} 模型...")
        model = YOLO(model_path)
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 进行预测
    results = model(image)
    
    # 在图片上绘制检测框
    annotated_image = results[0].plot()
    
    return annotated_image, results

def convert_to_voc_format(image_path, results):
    """
    将YOLOv11检测结果转换为Pascal VOC XML格式
    """
    # 获取图片信息
    image = cv2.imread(image_path)
    height, width, depth = image.shape
    
    # 创建XML根元素
    annotation = ET.Element('annotation')
    
    # 添加图片基本信息
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.dirname(image_path) or '.'
    
    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)
    
    path = ET.SubElement(annotation, 'path')
    path.text = os.path.abspath(image_path)
    
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    
    size = ET.SubElement(annotation, 'size')
    width_elem = ET.SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, 'height')
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, 'depth')
    depth_elem.text = str(depth)
    
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # 添加检测到的对象
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            
            # 获取边界框坐标 (xyxy格式)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 创建object元素
            obj = ET.SubElement(annotation, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = class_name
            
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(x1))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(y1))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(x2))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(y2))
            
            # 添加置信度信息 (非标准VOC字段，但可以保留用于参考)
            confidence_elem = ET.SubElement(obj, 'confidence')
            confidence_elem.text = f"{confidence:.2f}"
    
    # 返回格式化的XML字符串
    rough_string = ET.tostring(annotation, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
    # 图片路径
    image_path = '1.png'
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片 {image_path} 不存在")
        return
    
    try:
        # 对图片进行标注
        annotated_img, results = annotate_image_with_yolov11(image_path)
        
        # 保存标注后的图片
        output_path = 'annotated_1.png'
        cv2.imwrite(output_path, annotated_img)
        print(f"标注完成，结果保存至: {output_path}")
        
        # 生成XML标注文件
        xml_content = convert_to_voc_format(image_path, results)
        xml_path = image_path.replace('.png', '.xml')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        print(f"XML标注文件已生成: {xml_path}")
        
        # 打印检测结果
        print("\n检测到的对象:")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                # 获取边界框坐标用于显示
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                print(f"  - {class_name}: {confidence:.2f} [x1:{int(x1)}, y1:{int(y1)}, x2:{int(x2)}, y2:{int(y2)}]")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()