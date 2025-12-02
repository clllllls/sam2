import os
import numpy as np
import cv2
import json
import random
import time
import xml.etree.ElementTree as ET
from PyQt6.QtWidgets import QMessageBox
from .base import BaseBatchTransferComponent


class BatchGenerator(BaseBatchTransferComponent):
    """批量生成缺陷图像组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
    
    def generate_batch(self, target_image_path, selected_defect_paths, save_dir, generate_count, 
                      layer_masks, layer_checkboxes, hollow_layers, 
                      chk_random_layer, chk_random_defect, spin_max_defects, 
                      preview_defects, defect_selection_status):
        """批量生成缺陷图像
        
        Args:
            target_image_path: 目标图像路径
            selected_defect_paths: 选择的缺陷图像路径列表
            save_dir: 保存路径
            generate_count: 生成数量
            layer_masks: 层级掩码字典
            layer_checkboxes: 层级复选框字典
            hollow_layers: 中空层级字典
            chk_random_layer: 是否随机选择层
            chk_random_defect: 是否随机缺陷移植
            spin_max_defects: 每次最大缺陷数量
            preview_defects: 缺陷预览信息字典
            defect_selection_status: 缺陷选中状态字典
        """
        try:
            # 检查必要参数
            if not target_image_path:
                QMessageBox.warning(self.parent, "警告", "请先选择目标图像")
                return
                
            if not save_dir:
                QMessageBox.warning(self.parent, "警告", "请先选择保存路径")
                return
                
            if not selected_defect_paths:
                QMessageBox.warning(self.parent, "警告", "请先选择缺陷图像")
                return
            
            # 获取勾选的缺陷
            selected_defects = []
            for i in range(len(selected_defect_paths)):
                path = selected_defect_paths[i]
                defect_name = os.path.basename(path)
                if defect_name in defect_selection_status and defect_selection_status[defect_name]:
                    selected_defects.append({
                        'name': defect_name,
                        'path': path,
                        'info': preview_defects.get(defect_name, {})
                    })
            
            if not selected_defects:
                QMessageBox.warning(self.parent, "警告", "请至少选择一个缺陷")
                return
            
            # 获取勾选的层级
            selected_layers = []
            for layer_num, checkbox in layer_checkboxes.items():
                if checkbox.isChecked():
                    if layer_num in layer_masks:
                        selected_layers.append(layer_num)
            
            if not selected_layers:
                QMessageBox.warning(self.parent, "警告", "请至少选择一个层级")
                return
            
            # 加载目标图像
            target_image = cv2.imread(target_image_path)
            if target_image is None:
                raise Exception(f"无法加载目标图像: {target_image_path}")
            
            # 批量生成缺陷图像
            for i in range(generate_count):
                # 复制目标图像
                result_image = target_image.copy()
                # 初始化缺陷信息列表，用于后续生成JSON文件
                defect_info_list = []
                
                # 如果启用了随机缺陷移植，随机选择部分缺陷进行移植
                defects_to_transfer = []
                if chk_random_defect and selected_defects:
                    # 获取最大缺陷数量设置
                    max_defects = spin_max_defects
                    # 至少保留一个缺陷，但不超过最大缺陷数量
                    num_to_keep = random.randint(1, min(max_defects, len(selected_defects)))
                    defects_to_transfer = random.sample(selected_defects, num_to_keep)
                    print(f"第{i+1}张图随机保留{num_to_keep}个缺陷进行移植（最大允许{max_defects}个）")
                else:
                    # 保留所有缺陷
                    defects_to_transfer = selected_defects
                
                # 对选中的缺陷进行移植
                for defect in defects_to_transfer:
                    # 加载缺陷图像
                    defect_img = cv2.imread(defect['path'], cv2.IMREAD_UNCHANGED)
                    if defect_img is None:
                        raise Exception(f"无法加载缺陷图像: {defect['path']}")
                    
                    # 获取缺陷属性，如果没有则使用默认值
                    defect_info = defect['info']
                    scale_min = defect_info.get('scale_min', 100)
                    scale_max = defect_info.get('scale_max', 100)
                    alpha_min = defect_info.get('alpha_min', 100)
                    alpha_max = defect_info.get('alpha_max', 100)
                    angle_min = defect_info.get('angle_min', 0)
                    angle_max = defect_info.get('angle_max', 360)
                    
                    # 随机生成缩放比例和透明度
                    scale = random.randint(scale_min, scale_max) / 100.0
                    alpha = random.randint(alpha_min, alpha_max) / 100.0
                    
                    # 随机旋转角度，使用保存的参数范围
                    angle = random.randint(angle_min, angle_max)
                    print(f"随机生成的旋转角度: {angle}° (范围: {angle_min}-{angle_max}°)")
                    
                    # 选择层级区域
                    if chk_random_layer and selected_layers:
                        layer_num = random.choice(selected_layers)
                    else:
                        # 选择第一个选中的层级
                        layer_num = selected_layers[0]
                    
                    # 获取层级掩码
                    layer_mask = layer_masks[layer_num]
                    
                    # 调试信息
                    print(f"处理缺陷: {defect['name']}")
                    print(f"选择层级: {layer_num}")
                    
                    # 根据层级掩码确定有效的放置区域
                    img_height, img_width = result_image.shape[:2]
                    
                    # 初始化层级区域为整个图像（作为默认值）
                    min_y, max_y = 0, img_height - 1
                    min_x, max_x = 0, img_width - 1
                    
                    # 有效像素计数
                    valid_pixel_count = 0
                    
                    # 检查是否是中空结构
                    is_hollow = layer_num in hollow_layers and hollow_layers[layer_num]
                    print(f"层级是否为中空结构: {is_hollow}")
                    
                    if layer_mask is not None:
                        try:
                            # 对于实心结构（单个轮廓），根据轮廓点创建完整的掩码数组
                            if not is_hollow and isinstance(layer_mask, list):
                                print(f"检测到实心结构轮廓点列表，创建完整掩码数组")
                                # 创建与图像尺寸相同的掩码数组
                                layer_array = np.zeros((img_height, img_width), dtype=np.uint8)
                                # 将轮廓点转换为OpenCV格式并填充
                                points = np.array([[int(x), int(y)] for x, y in layer_mask], dtype=np.int32)
                                points = points.reshape((-1, 1, 2))
                                cv2.fillPoly(layer_array, [points], 255)
                                print(f"创建的实心层级掩码形状: {layer_array.shape}")
                            else:
                                # 对于中空结构或已经是数组的掩码，直接转换
                                layer_array = np.array(layer_mask)
                                print(f"层级掩码形状: {layer_array.shape}")
                            
                            # 标记是否已经设置了实心结构的层级区域
                            solid_layer_set = False
                            
                            # 确保层级掩码尺寸与原始图像一致
                            if layer_array.shape[:2] != (img_height, img_width):
                                print(f"层级掩码尺寸与原始图像不匹配，调整掩码尺寸")
                                # 对于实心结构，尝试调整掩码尺寸
                                if not is_hollow:
                                    from skimage.transform import resize
                                    try:
                                        layer_array = resize(layer_array, (img_height, img_width), \
                                                            preserve_range=True, anti_aliasing=False)
                                        print(f"调整后实心层级掩码形状: {layer_array.shape}")
                                    except:
                                        print("无法调整掩码尺寸，使用整个图像区域")
                                        min_y, max_y = 0, img_height - 1
                                        min_x, max_x = 0, img_width - 1
                                        valid_pixel_count = img_height * img_width
                                        solid_layer_set = True
                                else:
                                    # 对于中空结构，尝试调整掩码尺寸
                                    from skimage.transform import resize
                                    try:
                                        layer_array = resize(layer_array, (img_height, img_width), \
                                                            preserve_range=True, anti_aliasing=False)
                                        print(f"调整后中空层级掩码形状: {layer_array.shape}")
                                    except:
                                        print("无法调整掩码尺寸，使用默认值")
                            
                            # 找到所有大于0的像素（有效区域）
                            where = np.where(layer_array > 0)
                            valid_pixel_count = len(where[0])
                            print(f"层级区域有效像素数: {valid_pixel_count}")
                            
                            # 如果有有效像素，且不是实心结构或者还没有设置实心结构的层级区域，则计算有效的放置区域
                            if valid_pixel_count > 0 and not solid_layer_set:
                                if is_hollow:
                                    print(f"处理中空结构，将在有效像素点中随机选择位置")
                                    # 对于中空结构，我们不预先计算固定的矩形边界
                                    # 而是在放置缺陷时直接在有效像素中随机选择位置
                                    # 这里只计算整体边界用于后续逻辑
                                    min_y, max_y = np.min(where[0]), np.max(where[0])
                                    min_x, max_x = np.min(where[1]), np.max(where[1])
                                    print(f"中空结构整体边界: x({min_x}-{max_x}), y({min_y}-{max_y})")
                                    print(f"后续将在有效像素点中随机选择缺陷位置，避免放置在空心区域")
                                else:
                                    # 对于实心结构，正常计算矩形边界
                                    min_y, max_y = np.min(where[0]), np.max(where[0])
                                    min_x, max_x = np.min(where[1]), np.max(where[1])
                                    print(f"根据掩码计算的层级区域边界: x({min_x}-{max_x}), y({min_y}-{max_y})")
                            elif valid_pixel_count == 0 and not is_hollow:
                                # 如果没有有效像素，对于实心结构使用整个图像区域
                                min_y, max_y = 0, img_height - 1
                                min_x, max_x = 0, img_width - 1
                                valid_pixel_count = img_height * img_width
                                solid_layer_set = True
                                print(f"实心结构无有效像素，使用整个图像区域")
                        except Exception as e:
                            print(f"处理层级掩码时出错: {str(e)}")
                            # 出错时，对于实心结构使用整个图像区域
                            if not is_hollow:
                                min_y, max_y = 0, img_height - 1
                                min_x, max_x = 0, img_width - 1
                                valid_pixel_count = img_height * img_width
                                print(f"处理掩码出错，实心结构使用整个图像区域")
                            # 中空结构保持使用默认值
                    
                    print(f"最终使用的层级区域边界: x({min_x}-{max_x}), y({min_y}-{max_y})")
                    
                    # 缩放缺陷图像 - 按照设置的缩放比例
                    h, w = defect_img.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    print(f"缺陷原始尺寸: {w}x{h}, 缩放后尺寸: {new_w}x{new_h}, 缩放比例: {scale}")
                    
                    # 对于中空结构，特殊处理放置位置
                    if is_hollow and layer_mask is not None:
                        print(f"为中空结构选择缺陷放置位置")
                        # 确保我们有有效的像素点
                        if valid_pixel_count > 0:
                            # 保持缺陷原始比例，只设置最小尺寸限制
                            new_h = max(10, new_h)
                            new_w = max(10, new_w)
                            print(f"中空结构使用的缺陷尺寸: {new_w}x{new_h} (保持原始比例)")
                            
                            # 直接在有效像素点内随机生成坐标，不考虑缺陷大小
                            # 随机选择一个有效像素点作为缺陷中心
                            random_index = random.randint(0, valid_pixel_count - 1)
                            center_y = where[0][random_index]
                            center_x = where[1][random_index]
                            print(f"直接在有效像素点中随机选择位置: 中心({center_x}, {center_y})")
                        else:
                            # 如果没有有效像素，使用默认位置
                            center_y = (min_y + max_y) // 2
                            center_x = (min_x + max_x) // 2
                    else:
                        # 对于实心结构，正常处理
                        # 计算层级区域的实际大小
                        layer_height = max(1, max_y - min_y + 1)
                        layer_width = max(1, max_x - min_x + 1)
                        print(f"层级区域实际大小: {layer_width}x{layer_height}")
                        
                        # 保持缺陷原始比例，只设置最小尺寸限制
                        new_h = max(10, new_h)
                        new_w = max(10, new_w)
                        print(f"使用的缺陷尺寸: {new_w}x{new_h} (保持原始比例)")
                        
                        # 在层级区域内随机放置缺陷
                        # 计算有效放置区域（考虑缺陷大小）
                        available_height = max(0, layer_height - new_h)
                        available_width = max(0, layer_width - new_w)
                        
                        if available_height > 0 and available_width > 0:
                            # 在有效区域内随机选择位置
                            offset_y = random.randint(0, available_height)
                            offset_x = random.randint(0, available_width)
                            # 计算基于层级区域的中心位置
                            center_y = min_y + offset_y + new_h // 2
                            center_x = min_x + offset_x + new_w // 2
                        else:
                            # 如果可用区域太小，将缺陷放置在层级区域中心
                            center_y = (min_y + max_y) // 2
                            center_x = (min_x + max_x) // 2
                    
                    print(f"缺陷放置位置: ({center_x}, {center_y}) (层级区域内)")
                    
                    try:
                        # 调整图像大小
                        resized_defect = cv2.resize(defect_img, (new_w, new_h))
                        print(f"图像调整大小成功")
                        
                        # 应用随机旋转
                        if angle != 0:
                            # 获取图像中心
                            (h, w) = resized_defect.shape[:2]
                            center = (w // 2, h // 2)
                            
                            # 创建旋转矩阵
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            
                            # 计算旋转后的图像尺寸，确保不丢失任何部分
                            cos_theta = abs(M[0, 0])
                            sin_theta = abs(M[0, 1])
                            new_width = int(h * sin_theta + w * cos_theta)
                            new_height = int(h * cos_theta + w * sin_theta)
                            
                            # 调整旋转矩阵以考虑新的尺寸
                            M[0, 2] += (new_width / 2) - center[0]
                            M[1, 2] += (new_height / 2) - center[1]
                            
                            # 旋转图像
                            resized_defect = cv2.warpAffine(resized_defect, M, (new_width, new_height))
                            print(f"应用随机旋转: {angle}°，旋转后尺寸: {new_width}x{new_height}")
                            # 更新新的尺寸用于后续计算
                            new_h, new_w = new_height, new_width
                        
                        # 简化逻辑，直接以计算的中心点放置缺陷，不进行边界限制和尺寸调整
                        # 计算粘贴位置
                        top_y = center_y - new_h // 2
                        left_x = center_x - new_w // 2
                        bottom_y = top_y + new_h
                        right_x = left_x + new_w
                        
                        # 直接使用计算的区域，不进行边界检查
                        print(f"目标区域: x({left_x}-{right_x}), y({top_y}-{bottom_y}) (中心点位于({center_x}, {center_y}))")
                        
                        # 处理缺陷图像的通道
                        if len(resized_defect.shape) == 2:
                            # 灰度图像转换为3通道
                            resized_defect = cv2.cvtColor(resized_defect, cv2.COLOR_GRAY2BGR)
                            mask = None
                            print("转换灰度图像为3通道")
                        elif resized_defect.shape[2] == 4:
                            # 提取RGB通道和Alpha通道
                            bgr = resized_defect[:, :, :3]
                            mask = resized_defect[:, :, 3] / 255.0  # 归一化到0-1
                            resized_defect = bgr
                            print("提取RGB通道和Alpha通道作为掩码")
                        else:
                            mask = None
                            print("使用原始RGB通道")
                        
                        # 使用简单的覆盖方式，确保缺陷可见
                        # 设置较高的透明度确保可见
                        draw_alpha = max(0.5, alpha)  # 至少50%不透明度
                        
                        # 直接使用调整后的缺陷图像尺寸，不进行额外缩放
                        print(f"使用原始缺陷图像尺寸: {new_w}x{new_h}")
                        
                        # 混合图像 - 使用Alpha通道作为掩码
                        # 直接处理，不进行边界有效性检查
                        # 确保坐标在图像范围内（简单保护）
                        valid_top_y = max(0, top_y)
                        valid_left_x = max(0, left_x)
                        valid_bottom_y = min(result_image.shape[0], bottom_y)
                        valid_right_x = min(result_image.shape[1], right_x)
                        
                        # 计算缺陷图像的有效区域
                        defect_top = valid_top_y - top_y
                        defect_left = valid_left_x - left_x
                        defect_bottom = defect_top + (valid_bottom_y - valid_top_y)
                        defect_right = defect_left + (valid_right_x - valid_left_x)
                        
                        # 确保缺陷区域有效
                        if valid_bottom_y > valid_top_y and valid_right_x > valid_left_x:
                            result_roi = result_image[valid_top_y:valid_bottom_y, valid_left_x:valid_right_x]
                            
                            # 获取缺陷图像的有效部分
                            defect_roi = resized_defect[defect_top:defect_bottom, defect_left:defect_right]
                            
                            if mask is not None:
                                # 应用Alpha通道掩码
                                mask_roi = mask[defect_top:defect_bottom, defect_left:defect_right]
                                
                                # 确保mask是3通道
                                if len(mask_roi.shape) == 2:
                                    mask_roi = np.stack([mask_roi, mask_roi, mask_roi], axis=2)
                                
                                # 使用mask进行混合
                                blended = result_roi * (1 - mask_roi) + defect_roi * mask_roi
                                result_image[valid_top_y:valid_bottom_y, valid_left_x:valid_right_x] = blended.astype(np.uint8)
                                print("使用Alpha通道掩码成功混合缺陷到目标图像")
                                
                                # 保存缺陷信息到列表，用于生成JSON
                                # 这里保存更多信息，后续用于提取轮廓
                                defect_type = defect_info.get('defect_type', 'default_defect')
                                defect_info_list.append({
                                    'name': defect_type,
                                    'rect': {'left': valid_left_x, 'top': valid_top_y, 'right': valid_right_x, 'bottom': valid_bottom_y},
                                    'defect_roi': defect_roi,
                                    'mask_roi': mask_roi if mask is not None else None
                                })
                            else:
                                # 没有Alpha通道，直接覆盖
                                result_image[valid_top_y:valid_bottom_y, valid_left_x:valid_right_x] = defect_roi.astype(np.uint8)
                                print("直接覆盖目标区域以确保缺陷可见")
                                
                                # 保存缺陷信息到列表，用于生成JSON
                                # 这里保存更多信息，后续用于提取轮廓
                                defect_type = defect_info.get('defect_type', 'default_defect')
                                defect_info_list.append({
                                    'name': defect_type,
                                    'rect': {'left': valid_left_x, 'top': valid_top_y, 'right': valid_right_x, 'bottom': valid_bottom_y},
                                    'defect_roi': defect_roi,
                                    'mask_roi': None
                                })
                        else:
                            print("警告：目标区域超出图像边界")
                    except Exception as e:
                        print(f"处理缺陷时出错: {str(e)}")
                
                # 保存生成的图像
                # 使用时间戳+序号确保文件名唯一
                timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"  # 包含毫秒
                output_path = os.path.join(save_dir, f"defect_result_{timestamp}_{i+1}.bmp")
                cv2.imwrite(output_path, result_image)
                print(f"已保存图像: {output_path}")
                
                # 生成JSON文件，记录缺陷信息和轮廓坐标
                # 准备JSON数据结构
                # 将numpy类型转换为Python原生类型
                json_data = {
                    "version": "2.3.3",
                    "flags": {},
                    "shapes": [],
                    "imagePath": os.path.basename(output_path),
                    "imageData": None,
                    "imageHeight": int(result_image.shape[0]),
                    "imageWidth": int(result_image.shape[1]),
                    "text": ""
                }
                
                # 为每个移植的缺陷创建形状记录
                for defect_info in defect_info_list:
                    rect = defect_info['rect']
                    defect_roi = defect_info.get('defect_roi')
                    mask_roi = defect_info.get('mask_roi')
                    
                    # 提取缺陷的实际轮廓坐标
                    # 如果有mask，使用mask来提取轮廓
                    if mask_roi is not None:
                        # 将mask转换为二值图像（阈值处理）
                        mask_gray = (mask_roi * 255).astype(np.uint8)
                        # 确保mask是单通道
                        if len(mask_gray.shape) == 3:
                            mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
                        # 查找轮廓
                        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    else:
                        # 如果没有mask，尝试从缺陷图像中提取轮廓
                        # 将图像转换为灰度
                        if len(defect_roi.shape) == 3:
                            roi_gray = cv2.cvtColor(defect_roi, cv2.COLOR_BGR2GRAY)
                        else:
                            roi_gray = defect_roi.copy()
                        
                        # 二值化处理
                        _, binary = cv2.threshold(roi_gray, 1, 255, cv2.THRESH_BINARY)
                        # 查找轮廓
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    
                    # 合并所有轮廓点并转换为绝对坐标
                    points = []
                    # 按面积排序，选择最大的轮廓（如果有多个）
                    if contours:
                        # 计算每个轮廓的面积
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)
                        # 取最大的轮廓
                        largest_contour = contours[0]
                        
                        # 将轮廓点转换为绝对坐标
                        for point in largest_contour:
                            x = int(point[0][0] + rect['left'])
                            y = int(point[0][1] + rect['top'])
                            points.append([x, y])
                    
                    # 如果没有找到轮廓，回退到矩形轮廓
                    if not points:
                        points = [
                            [int(rect['left']), int(rect['top'])],
                            [int(rect['right']), int(rect['top'])],
                            [int(rect['right']), int(rect['bottom'])],
                            [int(rect['left']), int(rect['bottom'])]
                        ]
                    
                    # 添加到shapes列表
                    # 确保points中的坐标都是Python原生类型
                    json_points = [[int(x), int(y)] for x, y in points]
                    json_data['shapes'].append({
                        "text": "",
                        "label": defect_info['name'],
                        "points": json_points,
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "polygon",
                        "flags": {},
                        "attributes": {}
                    })
                
                # 保存JSON文件
                json_path = os.path.splitext(output_path)[0] + ".json"
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(json_data, json_file, indent=2, ensure_ascii=False)
                print(f"已保存JSON文件: {json_path}")
                
                # 生成XML检测矩形框文件
                # 创建根元素
                root = ET.Element('annotation')
                
                # 添加folder元素
                folder = ET.SubElement(root, 'folder')
                folder.text = os.path.basename(save_dir)
                
                # 添加filename元素
                filename = ET.SubElement(root, 'filename')
                filename.text = os.path.basename(output_path)
                
                # 添加path元素
                path_elem = ET.SubElement(root, 'path')
                path_elem.text = output_path
                
                # 添加source元素
                source = ET.SubElement(root, 'source')
                database = ET.SubElement(source, 'database')
                database.text = 'Unknown'
                
                # 添加size元素
                size = ET.SubElement(root, 'size')
                width = ET.SubElement(size, 'width')
                width.text = str(int(result_image.shape[1]))
                height = ET.SubElement(size, 'height')
                height.text = str(int(result_image.shape[0]))
                depth = ET.SubElement(size, 'depth')
                depth.text = str(3)  # 假设是RGB图像
                
                # 添加segmented元素
                segmented = ET.SubElement(root, 'segmented')
                segmented.text = '0'
                
                # 为每个缺陷添加object元素
                for defect_info in defect_info_list:
                    obj = ET.SubElement(root, 'object')
                    
                    # 添加name元素
                    name = ET.SubElement(obj, 'name')
                    name.text = defect_info['name']
                    
                    # 添加pose元素
                    pose = ET.SubElement(obj, 'pose')
                    pose.text = 'Unspecified'
                    
                    # 添加truncated元素
                    truncated = ET.SubElement(obj, 'truncated')
                    truncated.text = '0'
                    
                    # 添加difficult元素
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'
                    
                    # 添加bndbox元素
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin = ET.SubElement(bndbox, 'xmin')
                    xmin.text = str(int(defect_info['rect']['left']))
                    ymin = ET.SubElement(bndbox, 'ymin')
                    ymin.text = str(int(defect_info['rect']['top']))
                    xmax = ET.SubElement(bndbox, 'xmax')
                    xmax.text = str(int(defect_info['rect']['right']))
                    ymax = ET.SubElement(bndbox, 'ymax')
                    ymax.text = str(int(defect_info['rect']['bottom']))
                
                # 保存XML文件
                xml_path = os.path.splitext(output_path)[0] + ".xml"
                tree = ET.ElementTree(root)
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                print(f"已保存XML文件: {xml_path}")
            
            QMessageBox.information(self.parent, "成功", f"已成功生成 {generate_count} 张缺陷图像到 {save_dir}")
            
        except Exception as e:
            QMessageBox.critical(self.parent, "错误", f"批量移植过程中发生错误: {str(e)}")
