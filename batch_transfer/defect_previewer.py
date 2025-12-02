import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QPushButton, QSlider, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from .base import BaseBatchTransferComponent


class DefectPreviewer(BaseBatchTransferComponent):
    """缺陷预览和参数调整组件"""
    
    def __init__(self, parent=None, layer_preview=None):
        super().__init__(parent)
        self.parent = parent
        self.layer_preview = layer_preview
        
        # 初始化变量
        self.is_defect_preview_enabled = False  # 缺陷预览是否启用
        self.current_preview_scale = 100  # 当前预览缩放比例（百分比）
        self.current_preview_alpha = 100  # 当前预览透明度（百分比）
        self.current_preview_angle = 0  # 当前预览旋转角度（度）
        self.current_selected_defect = None  # 当前选中的缺陷
        
        # 添加定时器用于控制滑块刷新率
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(2)  # 设置延迟，降低刷新率
        self.slider_timer.timeout.connect(self._update_preview_after_delay)
        
        # 初始化UI组件
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 添加预览缺陷按钮
        self.btn_preview_defect = QPushButton("预览缺陷")
        self.btn_preview_defect.clicked.connect(self.preview_defect_on_target)
        
        # 添加缺陷预览缩放滑块
        self.slider_defect_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_defect_scale.setRange(1, 500)
        self.slider_defect_scale.setValue(100)
        self.slider_defect_scale.setMinimumWidth(200)
        self.slider_defect_scale.valueChanged.connect(self.on_defect_scale_changed)
        self.slider_defect_scale.setEnabled(False)  # 初始禁用
        
        # 添加缩放百分比显示
        self.lbl_scale_value = QLabel("100%")
        
        # 添加缺陷预览透明度滑块
        self.slider_defect_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_defect_alpha.setRange(0, 100)
        self.slider_defect_alpha.setValue(100)
        self.slider_defect_alpha.setMinimumWidth(200)
        self.slider_defect_alpha.valueChanged.connect(self.on_defect_alpha_changed)
        self.slider_defect_alpha.setEnabled(False)  # 初始禁用
        
        # 添加透明度百分比显示
        self.lbl_alpha_value = QLabel("100%")
        
        # 添加缺陷预览旋转滑块
        self.slider_defect_angle = QSlider(Qt.Orientation.Horizontal)
        self.slider_defect_angle.setRange(0, 360)
        self.slider_defect_angle.setValue(0)
        self.slider_defect_angle.setMinimumWidth(200)
        self.slider_defect_angle.valueChanged.connect(self.on_defect_angle_changed)
        self.slider_defect_angle.setEnabled(False)  # 初始禁用
        
        # 添加旋转角度显示
        self.lbl_angle_value = QLabel("0°")
    
    def get_widgets(self):
        """获取组件的所有widgets，用于添加到布局中"""
        return [
            self.btn_preview_defect,
            QLabel("缩放:"),
            self.slider_defect_scale,
            self.lbl_scale_value,
            QLabel("透明度:"),
            self.slider_defect_alpha,
            self.lbl_alpha_value,
            QLabel("旋转:"),
            self.slider_defect_angle,
            self.lbl_angle_value
        ]
    
    def preview_defect_on_target(self):
        """在目标图像上预览缺陷（切换显示/隐藏）"""
        if not hasattr(self.parent, 'target_image_full_path') or not self.parent.target_image_full_path:
            QMessageBox.warning(self.parent, "警告", "请先加载目标图像")
            return
        
        if not hasattr(self.parent, 'selected_defect_paths') or not self.parent.selected_defect_paths:
            QMessageBox.warning(self.parent, "警告", "请先选择缺陷图像")
            return
        
        try:
            if self.is_defect_preview_enabled:
                # 隐藏预览
                self.is_defect_preview_enabled = False
                self.btn_preview_defect.setText("预览缺陷")
                self.slider_defect_scale.setEnabled(False)
                self.slider_defect_alpha.setEnabled(False)
                self.slider_defect_angle.setEnabled(False)
                # 禁用layer_editor中的滑块
                if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
                    self.parent.layer_editor.slider_defect_scale.setEnabled(False)
                    self.parent.layer_editor.slider_defect_alpha.setEnabled(False)
                    self.parent.layer_editor.slider_defect_angle.setEnabled(False)
                
                # 重新显示当前层级或原始图像
                if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
                    self.parent.layer_editor.on_layer_double_clicked(self.parent.layer_editor.current_layer)
                else:
                    self.layer_preview.set_image(self.parent.target_image_full_path, preserve_view_state=True)
            else:
                # 显示预览
                self.is_defect_preview_enabled = True
                self.btn_preview_defect.setText("隐藏预览")
                self.slider_defect_scale.setEnabled(True)
                self.slider_defect_alpha.setEnabled(True)
                self.slider_defect_angle.setEnabled(True)
                # 启用layer_editor中的滑块
                if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
                    self.parent.layer_editor.slider_defect_scale.setEnabled(True)
                    self.parent.layer_editor.slider_defect_alpha.setEnabled(True)
                    self.parent.layer_editor.slider_defect_angle.setEnabled(True)
                
                # 如果没有选中的缺陷，选择第一个
                if not self.current_selected_defect and hasattr(self.parent, 'selected_defect_paths') and self.parent.selected_defect_paths:
                    first_defect_name = os.path.basename(self.parent.selected_defect_paths[0])
                    self.current_selected_defect = first_defect_name
                    if hasattr(self.parent, 'defect_manager') and self.parent.defect_manager:
                        self.parent.defect_manager.update_defect_scale_ui(first_defect_name)
                        self.parent.defect_manager.btn_save_defect_scale.setEnabled(True)
                
                self._show_defect_preview()
                
        except Exception as e:
            QMessageBox.critical(self.parent, "预览失败", f"预览缺陷时出错: {str(e)}")
            # 失败后重新显示当前层级
            if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
                self.parent.layer_editor.on_layer_double_clicked(self.parent.layer_editor.current_layer)
    
    def _show_defect_preview(self):
        """显示缺陷预览"""
        if not self.is_defect_preview_enabled or not self.current_selected_defect:
            return
        
        try:
            # 重新加载目标图像
            self.layer_preview.set_image(self.parent.target_image_full_path, preserve_view_state=True)
            
            # 加载目标图像
            target_img = cv2.imread(self.parent.target_image_full_path)
            if target_img is None:
                raise ValueError("无法加载目标图像")
            
            # 确定要预览的缺陷
            defect_path = None
            defect_name = self.current_selected_defect
            
            if hasattr(self.parent, 'defect_manager') and self.parent.defect_manager:
                if defect_name in self.parent.defect_manager.preview_defects:
                    defect_info = self.parent.defect_manager.preview_defects[defect_name]
                    defect_path = defect_info['path']
            elif hasattr(self.parent, 'selected_defect_paths') and self.parent.selected_defect_paths:
                # 如果没有defect_manager，直接从路径列表中查找
                for path in self.parent.selected_defect_paths:
                    if os.path.basename(path) == defect_name:
                        defect_path = path
                        break
            
            if not defect_path:
                # 如果没有找到，使用第一个缺陷
                defect_path = self.parent.selected_defect_paths[0]
                defect_name = os.path.basename(defect_path)
                self.current_selected_defect = defect_name
                
            # 加载缺陷图像
            defect_img = cv2.imread(defect_path, cv2.IMREAD_UNCHANGED)
            if defect_img is None:
                raise ValueError("无法加载缺陷图像")
            
            # 使用当前缩放比例
            scale = self.current_preview_scale / 100.0
            
            # 确保缩放比例有效
            scale = max(0.01, min(scale, 10.0))  # 限制在1%-1000%之间
            
            # 使用base.py中的create_temp_preview方法创建预览图像
            temp_path = self.create_temp_preview(
                self.parent.target_image_full_path,
                defect_path,
                scale,
                self.current_preview_alpha / 100.0,
                self.current_preview_angle
            )
            
            if temp_path:
                # 使用ImageViewer显示临时图像
                self.layer_preview.set_image(temp_path, preserve_view_state=True)
                
                # 更新信息预览
                if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
                    self.parent.layer_editor.layer_info_preview.setPlainText(
                        f"当前编辑: 第{self.parent.layer_editor.current_layer}层\n" +
                        f"缺陷预览: {defect_name}\n" +
                        f"当前缩放: {self.current_preview_scale}%\n" +
                        f"当前透明度: {self.current_preview_alpha}%\n" +
                        f"当前旋转: {self.current_preview_angle}°\n" +
                        f"预览位置: 图像中心"
                    )
            else:
                # 缩放过大，无法显示
                if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
                    self.parent.layer_editor.layer_info_preview.setPlainText(
                        f"当前编辑: 第{self.parent.layer_editor.current_layer}层\n" +
                        f"警告: 缺陷图像缩放后过大，无法在目标图像中显示\n" +
                        f"请减小缩放比例"
                    )
        except Exception as e:
            # 捕获所有异常，确保程序不会崩溃
            if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
                self.parent.layer_editor.layer_info_preview.setPlainText(
                    f"当前编辑: 第{self.parent.layer_editor.current_layer}层\n" +
                    f"预览失败: {str(e)}"
                )
    
    def on_defect_scale_changed(self, value):
        """当缺陷预览缩放滑块值变化时的处理"""
        self.current_preview_scale = value
        self.lbl_scale_value.setText(f"{value}%")
        
        # 如果正在预览，延迟更新预览
        if self.is_defect_preview_enabled:
            self.slider_timer.start()
            
    def on_defect_alpha_changed(self, value):
        """当缺陷预览透明度滑块值变化时的处理"""
        self.current_preview_alpha = value
        self.lbl_alpha_value.setText(f"{value}%")
        
        # 如果正在预览，延迟更新预览
        if self.is_defect_preview_enabled:
            self.slider_timer.start()
            
    def on_defect_angle_changed(self, value):
        """当缺陷预览旋转角度滑块值变化时的处理"""
        self.current_preview_angle = value
        self.lbl_angle_value.setText(f"{value}°")
        
        # 如果正在预览，延迟更新预览
        if self.is_defect_preview_enabled:
            self.slider_timer.start()
            
    def _update_preview_after_delay(self):
        """延迟更新预览，用于降低滑块刷新率"""
        if self.is_defect_preview_enabled:
            self._show_defect_preview()
    
    def update_selected_defect(self, defect_name):
        """更新当前选中的缺陷"""
        self.current_selected_defect = defect_name
        # 当选中缺陷时，自动显示缺陷预览
        self.is_defect_preview_enabled = True
        self.btn_preview_defect.setText("隐藏预览")
        self.slider_defect_scale.setEnabled(True)
        self.slider_defect_alpha.setEnabled(True)
        self.slider_defect_angle.setEnabled(True)
        # 启用layer_editor中的滑块
        if hasattr(self.parent, 'layer_editor') and self.parent.layer_editor:
            self.parent.layer_editor.slider_defect_scale.setEnabled(True)
            self.parent.layer_editor.slider_defect_alpha.setEnabled(True)
            self.parent.layer_editor.slider_defect_angle.setEnabled(True)
        # 显示缺陷预览
        self._show_defect_preview()
    
    def get_preview_params(self):
        """获取当前预览参数"""
        return {
            'scale': self.current_preview_scale,
            'alpha': self.current_preview_alpha,
            'angle': self.current_preview_angle
        }
    
    def set_preview_params(self, scale, alpha, angle):
        """设置预览参数"""
        self.current_preview_scale = scale
        self.current_preview_alpha = alpha
        self.current_preview_angle = angle
        
        # 更新UI
        self.slider_defect_scale.setValue(scale)
        self.slider_defect_alpha.setValue(alpha)
        self.slider_defect_angle.setValue(angle)
        
        self.lbl_scale_value.setText(f"{scale}%")
        self.lbl_alpha_value.setText(f"{alpha}%")
        self.lbl_angle_value.setText(f"{angle}°")
    
    def reset_preview(self):
        """重置预览状态"""
        self.is_defect_preview_enabled = False
        self.btn_preview_defect.setText("预览缺陷")
        self.slider_defect_scale.setEnabled(False)
        self.slider_defect_alpha.setEnabled(False)
        self.slider_defect_angle.setEnabled(False)
        
        # 重置预览参数
        self.current_preview_scale = 100
        self.current_preview_alpha = 100
        self.current_preview_angle = 0
        
        # 更新UI
        self.slider_defect_scale.setValue(100)
        self.slider_defect_alpha.setValue(100)
        self.slider_defect_angle.setValue(0)
        
        self.lbl_scale_value.setText("100%")
        self.lbl_alpha_value.setText("100%")
        self.lbl_angle_value.setText("0°")
