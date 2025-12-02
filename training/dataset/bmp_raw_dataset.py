# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import re
import json
from pycocotools import mask as mask_utils
from dataclasses import dataclass
from typing import List, Optional

import torch

from iopath.common.file_io import g_pathmgr

from training.dataset.vos_segment_loader import JSONSegmentLoader, LazySegments
from training.dataset.vos_raw_dataset import VOSFrame, VOSVideo, VOSRawDataset


class MultipleJSONSegmentLoader:
    """
    处理多个JSON标注文件的加载器。
    每个图像可能对应多个缺陷标注文件，需要合并这些标注。
    确保RLE编码格式符合pycocotools的要求。
    """
    def __init__(self, json_file_paths):
        """
        初始化多文件JSON加载器，专门处理转换后的COCO格式JSON文件。
        
        Args:
            json_file_paths: JSON文件路径列表
        """
        self.json_file_paths = json_file_paths
        self.segments = LazySegments()
        self.ann_every = 1  # 对于单帧图像，每1帧有一个标注
        self.annotation_map = {}  # 映射原始对象ID到内部ID
        
        # 合并所有JSON文件的标注
        obj_id_counter = 1  # 从1开始，避免使用0作为对象ID
        
        for i, json_path in enumerate(json_file_paths):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 专门处理COCO格式的JSON文件
                    if isinstance(data, dict) and 'annotations' in data and isinstance(data['annotations'], list):
                        #print(f"处理COCO格式JSON文件: {json_path}")
                        for j, annot in enumerate(data['annotations']):
                            if 'segmentation' in annot:
                                rle_data = annot['segmentation']
                                # 确保是字典格式的RLE编码
                                if isinstance(rle_data, dict) and 'counts' in rle_data and 'size' in rle_data:
                                    # 使用原始annotation中的id作为对象ID
                                    original_id = annot.get('id', obj_id_counter)
                                    
                                    # 存储映射关系
                                    self.annotation_map[original_id] = obj_id_counter
                                    
                                    # 处理counts字段
                                    if isinstance(rle_data['counts'], str):
                                        # 对于字符串形式的counts，转换为字节
                                        processed_rle = {
                                            'counts': rle_data['counts'].encode('utf-8'),
                                            'size': rle_data['size']
                                        }
                                        self.segments[obj_id_counter] = processed_rle
                                    else:
                                        # 其他格式直接使用
                                        self.segments[obj_id_counter] = rle_data
                                    
                                    #print(f"  加载对象ID {obj_id_counter}: 类别={annot.get('category_name', 'unknown')}, 面积={annot.get('area', 0)}")
                                    obj_id_counter += 1
                                else:
                                    print(f"  跳过无效的segmentation格式: {type(rle_data)}")
                    else:
                        print(f"警告: {json_path} 不是标准COCO格式JSON文件")
            except Exception as e:
                print(f"加载JSON文件 {json_path} 失败: {e}")
        
        #print(f"成功加载 {len(self.segments)} 个标注")
    
    def load(self, frame_idx, obj_ids=None):
        """
        加载指定帧的所有标注，返回字典格式而不是LazySegments对象。
        处理对象ID映射，确保正确的RLE格式。
        
        Args:
            frame_idx: 帧索引（对于单帧图像，总是0）
            obj_ids: 可选，指定要加载的对象ID
            
        Returns:
            字典格式的标注，键为对象ID，值为掩码，确保RLE格式符合pycocotools要求
        """
        # 返回字典格式而不是LazySegments对象
        result = {}
        
        if obj_ids is not None:
            # 只加载指定的对象ID
            for obj_id in obj_ids:
                # 检查是内部ID还是原始ID
                if obj_id in self.segments:
                    # 内部ID直接使用
                    result[obj_id] = self.segments[obj_id]
                elif obj_id in self.annotation_map:
                    # 原始ID需要映射到内部ID
                    mapped_id = self.annotation_map[obj_id]
                    result[mapped_id] = self.segments[mapped_id]
                else:
                    print(f"警告: 对象ID {obj_id} 未找到")
        else:
            # 加载所有对象
            for obj_id in self.segments.keys():
                result[obj_id] = self.segments[obj_id]
        
        #print(f"加载帧 {frame_idx} 的标注完成, 返回对象数量: {len(result)}")
        return result


class BMPRawDataset(VOSRawDataset):
    """
    数据集加载器，专门用于处理BMP图像格式和特殊的JSON标注文件格式。
    支持一个图像对应多个缺陷标注文件的情况。
    """
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
    ):
        """
        初始化数据集加载器。
        
        Args:
            img_folder: 图像文件夹路径
            gt_folder: 标注文件夹路径
            file_list_txt: 可选的文件列表文本文件路径
            excluded_videos_list_txt: 可选的排除文件列表文本文件路径
        """
        self.img_folder = img_folder
        self.gt_folder = gt_folder

        # 读取文件列表或获取所有图像文件
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            # 获取所有.bmp文件，不包含扩展名
            subset = [
                os.path.splitext(os.path.basename(path))[0]
                for path in glob.glob(os.path.join(self.img_folder, "*.bmp"))
            ]

        # 读取排除文件列表
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # 过滤出有效的视频名称（不包含排除文件）
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )
        
        print(f"找到 {len(self.video_names)} 个BMP图像文件")

    def get_video(self, idx):
        """
        获取指定索引的视频数据。
        
        Args:
            idx: 视频索引
            
        Returns:
            包含视频信息的VOSVideo对象和对应的分段加载器
        """
        if idx >= len(self.video_names):
            raise IndexError(f"索引 {idx} 超出范围，数据集长度为 {len(self.video_names)}")
            
        video_name = self.video_names[idx]
        
        # 构建图像文件路径
        video_frame_path = os.path.join(self.img_folder, video_name + ".bmp")
        
        # 找到所有匹配该图像的标注文件
        # 直接匹配同名的JSON文件，例如1_0-11.bmp对应1_0-11.json
        json_file = os.path.join(self.gt_folder, f"{video_name}.json")
        json_files = [json_file] if os.path.exists(json_file) else []
        
        if not json_files:
            print(f"警告：未找到图像 {video_name}.bmp 的标注文件")
            # 如果没有标注文件，创建一个空的JSON加载器
            # 这里可能需要根据实际情况调整处理方式
            pass
        
        # 创建VOSFrame对象
        frames = [VOSFrame(0, image_path=video_frame_path)]
        
        # 创建VOSVideo对象
        video = VOSVideo(video_name, idx, frames)
        
        # 创建自定义的多文件标注加载器
        if json_files:
            # 使用自定义的多文件加载逻辑
            segment_loader = MultipleJSONSegmentLoader(json_files)
        else:
            # 如果没有标注文件，返回None
            segment_loader = None
            
        return video, segment_loader

    def __len__(self):
        """
        返回数据集的长度（视频数量）。
        """
        return len(self.video_names)
