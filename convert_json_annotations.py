#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON标注格式转换工具

将labelme格式的JSON标注文件转换为SAM2训练所需的RLE编码格式
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import argparse
from pycocotools import mask as mask_utils


def polygon_to_mask(polygon, height, width):
    """
    将多边形坐标转换为二值掩码
    
    Args:
        polygon: 多边形点坐标列表 [[x1,y1], [x2,y2], ...]
        height: 图像高度
        width: 图像宽度
        
    Returns:
        numpy.ndarray: 二值掩码
    """
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 确保点坐标是整数
    polygon = [(int(x), int(y)) for x, y in polygon]
    
    # 绘制多边形
    draw.polygon(polygon, outline=1, fill=1)
    
    return np.array(mask)


def mask_to_rle(mask):
    """
    将二值掩码转换为RLE编码
    
    Args:
        mask: 二值掩码numpy数组
        
    Returns:
        dict: RLE编码字典，包含size和counts
    """
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')  # 转换为字符串格式
    return rle


def convert_labelme_to_sam2_format(input_json, output_dir):
    """
    将labelme格式的JSON转换为SAM2训练格式
    
    Args:
        input_json: 输入JSON文件路径
        output_dir: 输出目录路径
    """
    try:
        # 读取输入JSON
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像信息
        image_path = data.get('imagePath')
        image_width = data.get('imageWidth')
        image_height = data.get('imageHeight')
        
        if not all([image_width, image_height]):
            print(f"警告: {input_json} 中缺少图像尺寸信息")
            # 尝试从文件名匹配图像获取尺寸
            image_file = os.path.join(os.path.dirname(input_json), '..', 'images', image_path)
            if os.path.exists(image_file):
                with Image.open(image_file) as img:
                    image_width, image_height = img.size
                print(f"  从图像文件获取尺寸: {image_width}x{image_height}")
            else:
                print(f"  无法找到对应的图像文件: {image_file}")
                return False
        
        # 准备输出数据结构
        output_data = {
            "image": image_path,
            "width": image_width,
            "height": image_height,
            "annotations": []
        }
        
        # 处理每个形状
        for shape_idx, shape in enumerate(data.get('shapes', [])):
            label = shape.get('label', f'object_{shape_idx}')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'polygon')
            
            if shape_type != 'polygon':
                print(f"警告: {input_json} 中的形状类型 {shape_type} 不支持，跳过")
                continue
            
            if len(points) < 3:
                print(f"警告: {input_json} 中的多边形点数量不足，跳过")
                continue
            
            # 转换多边形为掩码
            mask = polygon_to_mask(points, image_height, image_width)
            
            # 转换掩码为RLE
            rle = mask_to_rle(mask)
            
            # 计算面积和边界框
            area = int(np.sum(mask))
            if area == 0:
                print(f"警告: {input_json} 中的多边形面积为0，跳过")
                continue
            
            # 计算边界框
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            
            # 添加到标注列表
            annotation = {
                "id": shape_idx + 1,
                "category_id": 1,  # 默认为类别1
                "category_name": label,
                "segmentation": rle,
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0
            }
            output_data["annotations"].append(annotation)
        
        # 保存输出JSON
        output_file = os.path.join(output_dir, os.path.basename(input_json))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功转换: {input_json} -> {output_file}")
        return True
        
    except Exception as e:
        print(f"转换失败 {input_json}: {str(e)}")
        return False


def batch_convert(input_dir, output_dir):
    """
    批量转换目录中的所有JSON文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    total = len(json_files)
    success_count = 0
    
    print(f"找到 {total} 个JSON文件")
    
    # 逐个转换
    for i, json_file in enumerate(json_files, 1):
        input_path = os.path.join(input_dir, json_file)
        print(f"[{i}/{total}] 处理 {json_file}")
        
        if convert_labelme_to_sam2_format(input_path, output_dir):
            success_count += 1
    
    print(f"转换完成！成功: {success_count}/{total}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='将labelme格式JSON标注转换为SAM2训练格式')
    parser.add_argument('--input', type=str, required=True, help='输入JSON文件或目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.input):
        # 批量处理目录
        batch_convert(args.input, args.output)
    else:
        # 处理单个文件
        convert_labelme_to_sam2_format(args.input, args.output)


if __name__ == '__main__':
    main()
