"""
缺陷移植工具 - 将分割出的缺陷从原图移植到新图片上
采用新的旋转逻辑：将移植后的缺陷视为单独图片进行旋转
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os
import time


class DefectTransferNew:
    """缺陷移植工具类（新版本）"""
    
    def __init__(self):
        self.original_image = None
        self.target_image = None
        self.defect_mask = None
        self.transferred_defect = None
        self.current_position = (0, 0)  # (x, y)
        self.current_scale = 1.0
        self.current_rotation = 0
        self.last_blended_result = None  # 保存上次融合的结果
        self.last_defect_region = None   # 保存上次的缺陷区域
        
    def load_images(self, original_image_path: str, target_image_path: str):
        """加载原始图像和目标图像"""
        try:
            # 加载原始图像（包含缺陷）
            self.original_image = cv2.imread(original_image_path)
            if self.original_image is None:
                raise ValueError(f"无法加载原始图像: {original_image_path}")
                
            # 加载目标图像（要移植到的背景）
            self.target_image = cv2.imread(target_image_path)
            if self.target_image is None:
                raise ValueError(f"无法加载目标图像: {target_image_path}")
                
            print(f"成功加载图像: 原始图像 {self.original_image.shape}, 目标图像 {self.target_image.shape}")
            return True
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            return False
    
    def set_defect_mask(self, mask: np.ndarray):
        """设置缺陷掩码"""
        if mask is None:
            raise ValueError("掩码不能为None")
            
        if len(mask.shape) != 2:
            raise ValueError("掩码必须是二维数组")
            
        self.defect_mask = mask.astype(np.uint8)
        print(f"缺陷掩码已设置，形状: {self.defect_mask.shape}")
        
        # 从原始图像中提取缺陷区域
        self._extract_defect()
        
    def _extract_defect(self):
        """从原始图像中提取缺陷区域"""
        if self.original_image is None or self.defect_mask is None:
            return
            
        # 确保掩码和图像尺寸匹配
        if self.defect_mask.shape[:2] != self.original_image.shape[:2]:
            # 调整掩码大小以匹配图像
            self.defect_mask = cv2.resize(
                self.defect_mask, 
                (self.original_image.shape[1], self.original_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # 提取缺陷区域（使用掩码）
        self.defect_region = cv2.bitwise_and(
            self.original_image, 
            self.original_image, 
            mask=self.defect_mask
        )
        
        # 获取缺陷的边界框
        y_indices, x_indices = np.where(self.defect_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            self.defect_bbox = (
                np.min(x_indices), np.min(y_indices),  # x_min, y_min
                np.max(x_indices), np.max(y_indices)   # x_max, y_max
            )
            
            # 裁剪缺陷区域
            x_min, y_min, x_max, y_max = self.defect_bbox
            self.cropped_defect = self.defect_region[y_min:y_max+1, x_min:x_max+1]
            self.cropped_mask = self.defect_mask[y_min:y_max+1, x_min:x_max+1]
            
            print(f"缺陷区域提取完成，尺寸: {self.cropped_defect.shape}")
            
            # 设置初始位置（目标图像中心）
            self.current_position = (
                (self.target_image.shape[1] - self.cropped_defect.shape[1]) // 2,
                (self.target_image.shape[0] - self.cropped_defect.shape[0]) // 2
            )
    
    def set_position(self, x: int, y: int):
        """设置缺陷位置"""
        self.current_position = (x, y)
        
    def set_scale(self, scale: float):
        """设置缺陷缩放比例"""
        self.current_scale = max(0.1, min(scale, 5.0))  # 限制缩放范围
        
    def set_rotation(self, angle):
        """设置缺陷旋转角度"""
        self.current_rotation = angle
    
    def apply_scale_only(self):
        """仅应用缩放变换"""
        if self.cropped_defect is None or self.cropped_mask is None:
            return None, None
            
        # 缩放
        if self.current_scale != 1.0:
            new_width = int(self.cropped_defect.shape[1] * self.current_scale)
            new_height = int(self.cropped_defect.shape[0] * self.current_scale)
            
            scaled_defect = cv2.resize(
                self.cropped_defect, 
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            scaled_mask = cv2.resize(
                self.cropped_mask,
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            scaled_defect = self.cropped_defect.copy()
            scaled_mask = self.cropped_mask.copy()
        
        return scaled_defect, scaled_mask
    
    def apply_rotation_to_image(self, image, mask, angle):
        """对图像应用旋转"""
        if image is None or mask is None:
            return None, None
            
        if angle == 0:
            return image.copy(), mask.copy()
        
        # 计算旋转中心（图像中心）
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算旋转后的图像尺寸，确保不丢失任何部分
        cos_theta = abs(rotation_matrix[0, 0])
        sin_theta = abs(rotation_matrix[0, 1])
        new_width = int(image.shape[1] * cos_theta + image.shape[0] * sin_theta)
        new_height = int(image.shape[1] * sin_theta + image.shape[0] * cos_theta)
        
        # 调整旋转矩阵以考虑新的尺寸
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # 旋转图像
        rotated_image = cv2.warpAffine(
            image, rotation_matrix, 
            (new_width, new_height)
        )
        
        # 旋转掩码
        rotated_mask = cv2.warpAffine(
            mask.astype(np.uint8), rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_NEAREST
        )
        
        # 优化掩码处理（保留边缘细节）
        rotated_mask = np.clip(rotated_mask, 0, 255).astype(np.uint8)
        _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)
        
        return rotated_image, rotated_mask
    
    def _apply_transformations(self):
        """应用缩放和旋转变换到缺陷区域"""
        if self.cropped_defect is None or self.cropped_mask is None:
            return None, None
            
        # 第一步：应用缩放
        scaled_defect, scaled_mask = self.apply_scale_only()
        
        # 第二步：应用旋转
        if self.current_rotation != 0:
            rotated_defect, rotated_mask = self.apply_rotation_to_image(
                scaled_defect, scaled_mask, self.current_rotation
            )
            return rotated_defect, rotated_mask
        else:
            return scaled_defect, scaled_mask

    def blend_defect(self, blend_strength=1.0):
        """融合缺陷到目标图像（仅支持缩放）"""
        print(f"[DEBUG] 开始融合缺陷，融合强度: {blend_strength}")
        
        if self.target_image is None or self.cropped_defect is None or self.cropped_mask is None:
            print("[ERROR] 无法融合缺陷：目标图像或缺陷区域未设置")
            return None
            
        try:
            # 应用缩放变换
            scaled_defect, scaled_mask = self.apply_scale_only()
            
            if scaled_defect is None or scaled_mask is None:
                print("[ERROR] 缩放变换失败")
                return None
                
            # 获取位置
            x, y = self.current_position
            
            # 确保位置在有效范围内
            x = max(0, min(x, self.target_image.shape[1] - scaled_defect.shape[1]))
            y = max(0, min(y, self.target_image.shape[0] - scaled_defect.shape[0]))
            
            print(f"[DEBUG] 融合位置: ({x}, {y}), 缺陷尺寸: {scaled_defect.shape}, 目标尺寸: {self.target_image.shape}")
            
            # 创建目标图像的副本
            result = self.target_image.copy()
            
            # 提取缺陷区域
            defect_region = scaled_defect
            mask_region = scaled_mask
            
            # 确保掩码是二值化的
            mask_binary = (mask_region > 0).astype(np.uint8)
            
            # 计算目标区域
            y_end = y + defect_region.shape[0]
            x_end = x + defect_region.shape[1]
            
            # 确保不超出边界
            if y_end > result.shape[0] or x_end > result.shape[1]:
                print("[ERROR] 缺陷超出目标图像边界")
                return None
                
            # 获取目标区域
            target_region = result[y:y_end, x:x_end]
            
            # 应用融合
            for c in range(3):  # 对每个颜色通道
                target_region[:, :, c] = np.where(
                    mask_binary > 0,
                    target_region[:, :, c] * (1 - blend_strength) + defect_region[:, :, c] * blend_strength,
                    target_region[:, :, c]
                )
            
            # 更新结果
            result[y:y_end, x:x_end] = target_region
            
            print("[DEBUG] 缺陷融合成功")
            return result
            
        except Exception as e:
            print(f"[ERROR] 融合缺陷时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def blend_defect_with_rotation(self, blend_strength=1.0):
        """融合缺陷到目标图像（支持缩放和旋转）"""
        print(f"[DEBUG] 开始融合缺陷（带旋转），融合强度: {blend_strength}")
        
        if self.target_image is None or self.cropped_defect is None or self.cropped_mask is None:
            print("[ERROR] 无法融合缺陷：目标图像或缺陷区域未设置")
            return None
            
        try:
            # 应用缩放和旋转变换
            transformed_defect, transformed_mask = self._apply_transformations()
            
            if transformed_defect is None or transformed_mask is None:
                print("[ERROR] 变换失败，尝试回退到仅缩放融合")
                return self.blend_defect(blend_strength)
                
            # 获取位置
            x, y = self.current_position
            
            # 确保位置在有效范围内
            x = max(0, min(x, self.target_image.shape[1] - transformed_defect.shape[1]))
            y = max(0, min(y, self.target_image.shape[0] - transformed_defect.shape[0]))
            
            print(f"[DEBUG] 融合位置: ({x}, {y}), 变换后缺陷尺寸: {transformed_defect.shape}, 目标尺寸: {self.target_image.shape}")
            
            # 创建目标图像的副本
            result = self.target_image.copy()
            
            # 确保掩码是二值化的
            mask_binary = (transformed_mask > 0).astype(np.uint8)
            
            # 计算目标区域
            y_end = y + transformed_defect.shape[0]
            x_end = x + transformed_defect.shape[1]
            
            # 确保不超出边界
            if y_end > result.shape[0] or x_end > result.shape[1]:
                print("[ERROR] 变换后的缺陷超出目标图像边界")
                return self.blend_defect(blend_strength)  # 回退到仅缩放
                
            # 获取目标区域
            target_region = result[y:y_end, x:x_end]
            
            # 应用融合
            for c in range(3):  # 对每个颜色通道
                target_region[:, :, c] = np.where(
                    mask_binary > 0,
                    target_region[:, :, c] * (1 - blend_strength) + transformed_defect[:, :, c] * blend_strength,
                    target_region[:, :, c]
                )
            
            # 更新结果
            result[y:y_end, x:x_end] = target_region
            cv2.imwrite("debug_rotated_result.png", result)
            print("[DEBUG] 带旋转的缺陷融合成功")
            return result
            
        except Exception as e:
            print(f"[ERROR] 带旋转融合缺陷时出错: {e}")
            import traceback
            traceback.print_exc()
            return self.blend_defect(blend_strength)  # 回退到仅缩放
            

    def save_result(self, output_path: str):
        """保存移植结果"""
        if self.transferred_defect is None:
            raise ValueError("没有可保存的结果，请先执行融合")
            
        cv2.imwrite(output_path, self.transferred_defect)
        print(f"结果已保存到: {output_path}")
        
    def get_preview(self) -> np.ndarray:
        """获取预览图像（显示缺陷位置）"""
        if self.target_image is None:
            return None
            
        preview = self.target_image.copy()
        
        if self.cropped_defect is not None:
            # 绘制缺陷边界框
            x, y = self.current_position
            scaled_defect, _ = self.apply_scale_only()
            
            if scaled_defect is not None:
                defect_h, defect_w = scaled_defect.shape[:2]
                
                # 绘制矩形框
                cv2.rectangle(preview, (x, y), (x + defect_w, y + defect_h), (0, 255, 0), 2)
                
                # 绘制中心点
                center_x = x + defect_w // 2
                center_y = y + defect_h // 2
                cv2.circle(preview, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return preview


def main():
    """独立运行示例"""
    # 创建缺陷移植器
    transfer = DefectTransferNew()
    
    # 加载图像
    original_path = "path/to/your/original_image.jpg"  # 替换为实际路径
    target_path = "path/to/your/target_image.jpg"      # 替换为实际路径
    
    if not transfer.load_images(original_path, target_path):
        return
    
    height, width = transfer.original_image.shape[:2]
    dummy_mask = np.zeros((height, width), dtype=np.uint8)
    # 在中心创建一个矩形区域作为示例缺陷
    center_x, center_y = width // 2, height // 2
    dummy_mask[center_y-50:center_y+50, center_x-50:center_x+50] = 255
    transfer.set_defect_mask(dummy_mask)
    
    # 设置位置和缩放
    transfer.set_position(100, 100)
    transfer.set_scale(1.5)
    transfer.set_rotation(45)
    
    # 保存结果
    output_dir = "transfer_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"transferred_defect_{timestamp}.jpg")
    transfer.save_result(output_path)


if __name__ == "__main__":
    main()