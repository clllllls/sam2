import os
import numpy as np
import cv2
import json
import random
import time
from PIL import Image, ImageQt
from PyQt6.QtWidgets import (
    QDialog, QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar,
    QMessageBox, QSplitter, QTabWidget, QGroupBox, QGridLayout, 
    QToolButton, QMenu, QSpinBox, QDoubleSpinBox, QSlider,
    QLineEdit, QTextEdit, QCheckBox, QFormLayout, QButtonGroup,
    QListWidget, QAbstractItemView, QListWidgetItem, QHBoxLayout, QVBoxLayout, QWidget
)
from PyQt6.QtGui import (
    QFont, QPixmap, QPainter, QPen, QColor, QBrush, QIcon, QCursor, 
    QImage, QPalette, QRegion, QFontDatabase, QBitmap, QIntValidator
)
from PyQt6.QtCore import (
    Qt, QPoint, QRect, QSize, pyqtSignal, pyqtSlot, QTimer
)
from image_viewer import ImageViewer


class BaseBatchTransferComponent:
    """批量移植组件的基础类，提供共享功能"""
    
    def __init__(self, parent=None):
        self.parent = parent
        
    def calculate_area(self, contour):
        """计算轮廓的面积
        
        Args:
            contour: 轮廓点列表
        
        Returns:
            float: 轮廓面积
        """
        if not contour or len(contour) < 3:
            return 0.0
        
        # 使用Shoelace公式计算多边形面积
        n = len(contour)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += contour[i][0] * contour[j][1]
            area -= contour[j][0] * contour[i][1]
        area = abs(area) / 2.0
        return area
    
    def _load_path_config(self, config_file):
        """从配置文件加载路径设置
        
        Args:
            config_file: 配置文件路径
        
        Returns:
            dict: 路径配置字典
        """
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config
        except Exception as e:
            print(f"加载路径配置失败: {e}")
        return {
            'last_defect_dir': '',
            'last_target_dir': '',
            'last_save_dir': ''
        }
    
    def _save_path_config(self, config_file, config):
        """保存路径设置到配置文件
        
        Args:
            config_file: 配置文件路径
            config: 路径配置字典
        """
        try:
            # 确保配置目录存在
            config_dir = os.path.dirname(config_file)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存路径配置失败: {e}")
    
    def get_layer_color(self, layer_num):
        """获取层级对应的颜色
        
        Args:
            layer_num: 层级编号
        
        Returns:
            list: RGBA颜色值
        """
        layer_colors = [
            [255, 0, 0, 100],    # 红色 - 第1层
            [0, 255, 0, 100],    # 绿色 - 第2层
            [0, 0, 255, 100],    # 蓝色 - 第3层
            [255, 255, 0, 100],  # 黄色 - 第4层
            [255, 0, 255, 100],  # 品红 - 第5层
            [0, 255, 255, 100],  # 青色 - 第6层
            [128, 0, 0, 100],    # 深红 - 第7层
            [0, 128, 0, 100],    # 深绿 - 第8层
            [0, 0, 128, 100]     # 深蓝 - 第9层
        ]
        return layer_colors[(layer_num - 1) % len(layer_colors)]
    
    def create_temp_preview(self, target_path, defect_path, scale, alpha, angle):
        """创建缺陷预览图像
        
        Args:
            target_path: 目标图像路径
            defect_path: 缺陷图像路径
            scale: 缩放比例
            alpha: 透明度
            angle: 旋转角度
        
        Returns:
            str: 临时预览图像路径
        """
        try:
            # 加载目标图像
            target_img = cv2.imread(target_path)
            if target_img is None:
                raise ValueError("无法加载目标图像")
            
            # 加载缺陷图像
            defect_img = cv2.imread(defect_path, cv2.IMREAD_UNCHANGED)
            if defect_img is None:
                raise ValueError("无法加载缺陷图像")
            
            # 缩放缺陷图像
            new_w = max(1, int(defect_img.shape[1] * scale))
            new_h = max(1, int(defect_img.shape[0] * scale))
            scaled_defect = cv2.resize(defect_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 应用旋转
            if angle != 0:
                (h, w) = scaled_defect.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w_rot = int((h * sin) + (w * cos))
                new_h_rot = int((h * cos) + (w * sin))
                
                M[0, 2] += (new_w_rot / 2) - center[0]
                M[1, 2] += (new_h_rot / 2) - center[1]
                
                if len(scaled_defect.shape) == 2:
                    scaled_defect = cv2.warpAffine(scaled_defect, M, (new_w_rot, new_h_rot), 
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                                borderValue=0)
                else:
                    orig_channels = scaled_defect.shape[2]
                    
                    if orig_channels == 4:
                        bgr = scaled_defect[:, :, :3]
                        alpha_channel = scaled_defect[:, :, 3]
                        
                        bgr_rotated = cv2.warpAffine(bgr, M, (new_w_rot, new_h_rot), 
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                                    borderValue=(0, 0, 0))
                        alpha_rotated = cv2.warpAffine(alpha_channel, M, (new_w_rot, new_h_rot), 
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                                    borderValue=0)
                        
                        scaled_defect = np.dstack((bgr_rotated, alpha_rotated))
                    else:
                        scaled_defect = cv2.warpAffine(scaled_defect, M, (new_w_rot, new_h_rot), 
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                                    borderValue=(0, 0, 0))
                
                new_h, new_w = scaled_defect.shape[:2]
            
            # 计算放置位置（图像中心）
            target_h, target_w = target_img.shape[:2]
            x_pos = max(0, min((target_w - new_w) // 2, target_w - new_w))
            y_pos = max(0, min((target_h - new_h) // 2, target_h - new_h))
            
            actual_new_w = min(new_w, target_w - x_pos)
            actual_new_h = min(new_h, target_h - y_pos)
            
            if actual_new_w > 0 and actual_new_h > 0:
                result_img = target_img.copy()
                scaled_defect_cropped = scaled_defect[:actual_new_h, :actual_new_w]
                
                if scaled_defect_cropped.shape[-1] == 4:
                    defect_bgr = scaled_defect_cropped[:, :, :3]
                    alpha_channel = scaled_defect_cropped[:, :, 3] / 255.0
                    alpha_channel = alpha_channel * alpha
                    
                    if len(alpha_channel.shape) == 2:
                        alpha_channel = alpha_channel[:, :, np.newaxis]
                    
                    result_region = result_img[y_pos:y_pos+actual_new_h, x_pos:x_pos+actual_new_w]
                    result_region = (defect_bgr * alpha_channel + result_region * (1 - alpha_channel)).astype(np.uint8)
                    result_img[y_pos:y_pos+actual_new_h, x_pos:x_pos+actual_new_w] = result_region
                elif scaled_defect_cropped.shape[-1] == 3:
                    result_region = result_img[y_pos:y_pos+actual_new_h, x_pos:x_pos+actual_new_w]
                    result_region = (scaled_defect_cropped * alpha + result_region * (1 - alpha)).astype(np.uint8)
                    result_img[y_pos:y_pos+actual_new_h, x_pos:x_pos+actual_new_w] = result_region
                else:
                    if len(scaled_defect_cropped.shape) == 2:
                        defect_bgr = cv2.cvtColor(scaled_defect_cropped, cv2.COLOR_GRAY2BGR)
                        result_region = result_img[y_pos:y_pos+actual_new_h, x_pos:x_pos+actual_new_w]
                        result_region = (defect_bgr * alpha + result_region * (1 - alpha)).astype(np.uint8)
                        result_img[y_pos:y_pos+actual_new_h, x_pos:x_pos+actual_new_w] = result_region
                    else:
                        result_img[y_pos:y_pos+actual_new_h, x_pos:x_pos+actual_new_w] = \
                            scaled_defect_cropped[:, :, 0:3]
                
                # 保存合成图像到临时路径
                temp_path = os.path.join(os.path.dirname(target_path), "temp_preview.png")
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(result_img_rgb)
                pil_image.save(temp_path)
                
                return temp_path
            
            return None
        except Exception as e:
            print(f"创建预览图像失败: {e}")
            return None
