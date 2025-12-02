import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QCheckBox, QSpinBox, QWidget, QTextEdit, QButtonGroup, QSlider
)
from PyQt6.QtCore import Qt
from .base import BaseBatchTransferComponent
from image_viewer import ImageViewer


class LayerEditor(BaseBatchTransferComponent):
    """层级编辑和掩码绘制组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # 初始化变量
        self.current_layer = 1  # 当前编辑的层级
        self.layer_masks = {}  # 存储各层级的掩码信息
        self.layer_checkboxes = {}  # 层级复选框
        self.hollow_layers = {}  # 存储中空结构的层
        self.layer_contours = {}  # 存储每层的轮廓列表（可能有多个，用于中空结构）
        self.current_contour_index = 0  # 当前正在绘制的轮廓索引
        
        # 初始化UI组件
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建目标图像分层操作区域
        self.layer_group = QGroupBox("目标图像分层操作")
        self.layer_layout = QVBoxLayout()
        
        # 分层参数设置
        self.layer_options_layout = QVBoxLayout()
        
        # 创建水平布局容纳启用分层处理、分层数量、随机选择层和层级选择
        self.layer_options_row_layout = QHBoxLayout()
        
        self.chk_enable_layering = QCheckBox("启用分层处理")
        self.layer_options_row_layout.addWidget(self.chk_enable_layering)
        
        layer_count_label = QLabel("分层数量:")
        self.layer_options_row_layout.addWidget(layer_count_label)
        
        self.spin_layer_count = QSpinBox()
        self.spin_layer_count.setRange(1, 10)
        self.spin_layer_count.setValue(3)
        # 连接信号，当分层数量变化时更新层级选择复选框
        self.spin_layer_count.valueChanged.connect(self._update_layer_checkboxes)
        self.layer_options_row_layout.addWidget(self.spin_layer_count)
        
        # 将随机选择层复选框添加到同一行，放在分层数量的右边
        self.chk_random_layer = QCheckBox("随机选择层")
        self.layer_options_row_layout.addWidget(self.chk_random_layer)
        
        # 添加随机缺陷移植复选框
        self.chk_random_defect = QCheckBox("随机缺陷移植")
        self.layer_options_row_layout.addWidget(self.chk_random_defect)
        
        # 添加最大缺陷数量控件
        self.layer_options_row_layout.addWidget(QLabel("每次最大缺陷数量:"))
        self.spin_max_defects = QSpinBox()
        self.spin_max_defects.setRange(1, 100)
        self.spin_max_defects.setValue(3)  # 默认值为3
        self.spin_max_defects.setFixedWidth(60)
        self.layer_options_row_layout.addWidget(self.spin_max_defects)
        
        # 添加层级选择区域，放在随机选择层的右边
        self.layer_options_row_layout.addWidget(QLabel("选择层级:"))
        self.layer_selection_widget = QWidget()
        self.layer_selection_layout = QHBoxLayout()
        self.layer_selection_widget.setLayout(self.layer_selection_layout)
        self.layer_options_row_layout.addWidget(self.layer_selection_widget)
        
        self.layer_options_row_layout.addStretch()  # 添加伸缩空间使控件靠左
        
        # 创建一个容器部件来容纳水平布局
        self.layer_options_widget = QWidget()
        self.layer_options_widget.setLayout(self.layer_options_row_layout)
        
        # 将容器部件添加到表单布局
        self.layer_options_layout.addWidget(self.layer_options_widget)
        
        self.layer_layout.addLayout(self.layer_options_layout)
        
        # 存储层级复选框的字典
        self.layer_checkboxes = {}
        # 初始化层级复选框
        self._update_layer_checkboxes(self.spin_layer_count.value())
        
        # 添加绘制工具选择区域
        self.drawing_tools_layout = QHBoxLayout()
        self.drawing_tools_layout.addWidget(QLabel("绘制工具:"))
        
        self.btn_pen = QPushButton("钢笔")
        self.btn_pen.setCheckable(True)
        self.btn_pen.setChecked(True)
        self.btn_pen.clicked.connect(lambda: self.set_drawing_tool(0))
        self.drawing_tools_layout.addWidget(self.btn_pen)
        
        self.btn_rect = QPushButton("矩形框")
        self.btn_rect.setCheckable(True)
        self.btn_rect.clicked.connect(lambda: self.set_drawing_tool(1))
        self.drawing_tools_layout.addWidget(self.btn_rect)
        
        self.btn_circle = QPushButton("圆形")
        self.btn_circle.setCheckable(True)
        self.btn_circle.clicked.connect(lambda: self.set_drawing_tool(2))
        self.drawing_tools_layout.addWidget(self.btn_circle)
        
        # 添加绘制工具按钮组
        self.drawing_tool_group = QButtonGroup()
        self.drawing_tool_group.addButton(self.btn_pen)
        self.drawing_tool_group.addButton(self.btn_rect)
        self.drawing_tool_group.addButton(self.btn_circle)
        
        # 添加重置按钮
        self.btn_reset_drawing = QPushButton("重置绘制")
        self.btn_reset_drawing.clicked.connect(self.reset_current_drawing)
        self.drawing_tools_layout.addWidget(self.btn_reset_drawing)
        
        # 添加内部轮廓按钮
        self.btn_add_inner_contour = QPushButton("添加内部轮廓")
        self.btn_add_inner_contour.clicked.connect(self.add_inner_contour)
        self.drawing_tools_layout.addWidget(self.btn_add_inner_contour)
        
        # 添加保存层级按钮
        self.btn_save_layer = QPushButton("保存层级")
        self.btn_save_layer.clicked.connect(self.save_current_layer)
        self.drawing_tools_layout.addWidget(self.btn_save_layer)
        
        # 添加缺陷预览缩放滑块
        self.drawing_tools_layout.addWidget(QLabel("缩放:"))
        self.slider_defect_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_defect_scale.setRange(1, 500)
        self.slider_defect_scale.setValue(100)
        self.slider_defect_scale.setMinimumWidth(200)
        self.slider_defect_scale.setEnabled(False)  # 初始禁用
        self.drawing_tools_layout.addWidget(self.slider_defect_scale)
        
        # 添加缩放百分比显示
        self.lbl_scale_value = QLabel("100%")
        self.drawing_tools_layout.addWidget(self.lbl_scale_value)
        
        # 添加缺陷预览透明度滑块
        self.drawing_tools_layout.addWidget(QLabel("透明度:"))
        self.slider_defect_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_defect_alpha.setRange(0, 100)
        self.slider_defect_alpha.setValue(100)
        self.slider_defect_alpha.setMinimumWidth(200)
        self.slider_defect_alpha.setEnabled(False)  # 初始禁用
        self.drawing_tools_layout.addWidget(self.slider_defect_alpha)
        
        # 添加透明度百分比显示
        self.lbl_alpha_value = QLabel("100%")
        self.drawing_tools_layout.addWidget(self.lbl_alpha_value)
        
        # 添加缺陷预览旋转滑块
        self.drawing_tools_layout.addWidget(QLabel("旋转:"))
        self.slider_defect_angle = QSlider(Qt.Orientation.Horizontal)
        self.slider_defect_angle.setRange(0, 360)
        self.slider_defect_angle.setValue(0)
        self.slider_defect_angle.setMinimumWidth(200)
        self.slider_defect_angle.setEnabled(False)  # 初始禁用
        self.drawing_tools_layout.addWidget(self.slider_defect_angle)
        
        # 添加旋转角度显示
        self.lbl_angle_value = QLabel("0°")
        self.drawing_tools_layout.addWidget(self.lbl_angle_value)
        
        # 连接滑块信号到defect_previewer的处理函数
        self.slider_defect_scale.valueChanged.connect(self._on_defect_scale_changed)
        self.slider_defect_alpha.valueChanged.connect(self._on_defect_alpha_changed)
        self.slider_defect_angle.valueChanged.connect(self._on_defect_angle_changed)
        
        self.drawing_tools_layout.addStretch()
        self.layer_layout.addLayout(self.drawing_tools_layout)
        
        # 添加层级信息预览文本框
        self.layer_info_preview = QTextEdit(self.parent)
        self.layer_info_preview.setReadOnly(True)
        self.layer_info_preview.setMaximumHeight(80)
        self.layer_layout.addWidget(self.layer_info_preview)
        
        # 缺陷图像视图（单个视图）
        self.layer_preview = ImageViewer(self.parent)
        self.layer_preview.setMinimumHeight(400)
        self.layer_preview.set_mode(3)  # 设置为轮廓绘制模式
        
        self.layer_layout.addWidget(self.layer_preview)
        
        self.layer_group.setLayout(self.layer_layout)
    
    def get_widget(self):
        """获取组件的主widget"""
        return self.layer_group
    
    def get_image_viewer(self):
        """获取图像查看器组件"""
        return self.layer_preview
    
    def _update_layer_checkboxes(self, count):
        """更新层级选择复选框
        
        Args:
            count: 分层数量
        """
        # 清除现有的复选框
        for i in range(len(self.layer_checkboxes) + 1):
            if i in self.layer_checkboxes:
                checkbox = self.layer_checkboxes[i]
                self.layer_selection_layout.removeWidget(checkbox)
                checkbox.deleteLater()
        
        self.layer_checkboxes.clear()
        
        # 创建新的复选框
        for i in range(1, count + 1):
            checkbox = QCheckBox(f"第{i}层")
            checkbox.setChecked(True)  # 默认选中所有层
            checkbox.mouseDoubleClickEvent = lambda event, layer=i: self.on_layer_double_clicked(layer)
            self.layer_checkboxes[i] = checkbox
            self.layer_selection_layout.addWidget(checkbox)
        
        # 添加伸缩空间使复选框靠左
        self.layer_selection_layout.addStretch()
        
        # 初始化新添加层级的掩码
        for i in range(1, count + 1):
            if i not in self.layer_masks:
                self.layer_masks[i] = None
            if i not in self.layer_contours:
                self.layer_contours[i] = []
    
    def on_layer_double_clicked(self, layer):
        """处理层级复选框双击事件
        
        Args:
            layer: 双击的层级
        """
        self.current_layer = layer
        
        # 清除当前显示
        self.layer_preview.clear_all()
        if hasattr(self.parent, 'target_image_full_path') and self.parent.target_image_full_path:
            self.layer_preview.set_image(self.parent.target_image_full_path, preserve_view_state=True)
        
        # 显示当前层级的掩码（如果有）
        if layer in self.layer_masks and self.layer_masks[layer] is not None:
            # 根据层级选择颜色
            color = self.get_layer_color(layer)
            
            # 检查是否是中空结构
            is_hollow = layer in self.hollow_layers and self.hollow_layers[layer]
            
            if is_hollow:
                # 对于中空结构，直接设置掩码
                self.layer_preview.set_mask(self.layer_masks[layer], color=color)
                # 计算并显示面积
                area = np.sum(self.layer_masks[layer]) / 255.0
            else:
                # 对于普通轮廓，设置轮廓点和掩码
                self.layer_preview.contour_points = self.layer_masks[layer].copy()
                self.layer_preview.set_mask(self.layer_masks[layer], color=color)
                # 计算并显示面积
                area = self.calculate_area(self.layer_masks[layer])
            
            self.layer_info_preview.setPlainText(f"当前编辑: 第{layer}层\n已保存区域面积: {area:.2f} 像素")
        else:
            # 初始化当前层级的轮廓列表
            if layer not in self.layer_contours:
                self.layer_contours[layer] = []
            self.layer_info_preview.setPlainText(f"当前编辑: 第{layer}层\n尚未保存区域")
    
    def set_drawing_tool(self, tool):
        """设置绘制工具
        
        Args:
            tool: 0: 钢笔, 1: 矩形框, 2: 圆形
        """
        self.layer_preview.set_drawing_tool(tool)
    
    def reset_current_drawing(self):
        """重置当前层级的绘制内容"""
        # 清除当前绘制内容
        self.layer_preview.clear_all()
        # 如果有目标图像，重新加载
        if hasattr(self.parent, 'target_image_full_path') and self.parent.target_image_full_path:
            self.layer_preview.set_image(self.parent.target_image_full_path, preserve_view_state=True)
        # 清除当前层级的掩码和轮廓列表（如果有）
        if self.current_layer in self.layer_masks:
            self.layer_masks[self.current_layer] = None
        if self.current_layer in self.layer_contours:
            self.layer_contours[self.current_layer] = []
        # 更新信息预览
        self.layer_info_preview.setPlainText(f"当前编辑: 第{self.current_layer}层\n尚未保存区域")
    
    def add_inner_contour(self):
        """添加内部轮廓"""
        # 获取当前绘制的轮廓
        contour = self.layer_preview.get_contour()
        if not contour:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self.parent, "警告", "请先绘制外部轮廓")
            return
        
        # 保存当前轮廓到轮廓列表
        if self.current_layer not in self.layer_contours:
            self.layer_contours[self.current_layer] = []
        
        self.layer_contours[self.current_layer].append(contour)
        
        # 清除当前绘制，准备绘制内部轮廓
        self.layer_preview.clear_all()
        if hasattr(self.parent, 'target_image_full_path') and self.parent.target_image_full_path:
            self.layer_preview.set_image(self.parent.target_image_full_path, preserve_view_state=True)
        
        # 显示当前轮廓数量
        contour_count = len(self.layer_contours[self.current_layer])
        self.layer_info_preview.setPlainText(
            f"当前编辑: 第{self.current_layer}层\n" +
            f"已绘制轮廓数量: {contour_count}\n" +
            f"当前正在绘制: 第{contour_count + 1}个轮廓（内部轮廓）"
        )
    
    def save_current_layer(self):
        """保存当前层级"""
        from PyQt6.QtWidgets import QMessageBox
        
        # 获取当前绘制的轮廓
        contour = self.layer_preview.get_contour()
        if not contour:
            QMessageBox.warning(self.parent, "警告", "请先绘制轮廓")
            return
        
        # 保存当前轮廓到轮廓列表
        if self.current_layer not in self.layer_contours:
            self.layer_contours[self.current_layer] = []
        
        self.layer_contours[self.current_layer].append(contour)
        
        # 计算面积
        area = 0.0
        
        # 处理不同类型的轮廓
        if len(self.layer_contours[self.current_layer]) == 1:
            # 只有一个轮廓，作为实心结构
            self.layer_masks[self.current_layer] = contour
            self.hollow_layers[self.current_layer] = False
            area = self.calculate_area(contour)
        else:
            # 多个轮廓，作为中空结构处理
            if hasattr(self.parent, 'target_image_full_path') and self.parent.target_image_full_path:
                # 加载目标图像获取尺寸
                target_img = cv2.imread(self.parent.target_image_full_path)
                if target_img is not None:
                    height, width = target_img.shape[:2]
                    
                    # 创建基础掩码
                    hollow_mask = np.zeros((height, width), dtype=np.uint8)
                    
                    # 填充外部轮廓
                    outer_contour = self.layer_contours[self.current_layer][0]
                    points = np.array([[int(x), int(y)] for x, y in outer_contour], dtype=np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.fillPoly(hollow_mask, [points], 255)
                    
                    # 依次减去每个内部轮廓
                    for inner_contour in self.layer_contours[self.current_layer][1:]:
                        inner_mask = np.zeros((height, width), dtype=np.uint8)
                        inner_points = np.array([[int(x), int(y)] for x, y in inner_contour], dtype=np.int32)
                        inner_points = inner_points.reshape((-1, 1, 2))
                        cv2.fillPoly(inner_mask, [inner_points], 255)
                        hollow_mask = cv2.subtract(hollow_mask, inner_mask)
                    
                    # 保存中空结构掩码
                    self.layer_masks[self.current_layer] = hollow_mask
                    self.hollow_layers[self.current_layer] = True
                    
                    # 计算中空区域的面积
                    area = np.sum(hollow_mask) / 255.0  # 255是白色像素值
        
        # 显示保存成功信息，包含面积
        QMessageBox.information(self.parent, "保存成功", f"第{self.current_layer}层已保存\n区域面积: {area:.2f} 像素")
        
        # 更新信息预览
        self.layer_info_preview.setPlainText(f"当前编辑: 第{self.current_layer}层\n已保存区域面积: {area:.2f} 像素")
        
        # 清除当前层的轮廓列表，准备下一次绘制
        self.layer_contours[self.current_layer] = []
        self.layer_preview.clear_all()
        
        # 重新加载图像以显示保存的掩码
        if hasattr(self.parent, 'target_image_full_path') and self.parent.target_image_full_path:
            self.layer_preview.set_image(self.parent.target_image_full_path, preserve_view_state=True)
            
            # 根据层级选择颜色
            color = self.get_layer_color(self.current_layer)
            
            # 设置掩码显示中空结构
            self.layer_preview.set_mask(self.layer_masks[self.current_layer], color=color)
    
    def _update_layer_info_preview(self):
        """更新层级信息预览"""
        info = "层级信息:\n"
        for layer_num, mask in self.layer_masks.items():
            is_hollow = layer_num in self.hollow_layers and self.hollow_layers[layer_num]
            info += f"层级 {layer_num}: {'中空' if is_hollow else '实心'} 结构\n"
        self.layer_info_preview.setText(info)
    
    def update_layer_info(self):
        """更新层级信息预览"""
        self._update_layer_info_preview()
    
    def _on_defect_scale_changed(self, value):
        """处理缺陷缩放滑块变化事件"""
        # 更新显示
        self.lbl_scale_value.setText(f"{value}%")
        # 调用defect_previewer的处理函数
        if hasattr(self.parent, 'defect_previewer') and self.parent.defect_previewer:
            self.parent.defect_previewer.slider_defect_scale.setValue(value)
            self.parent.defect_previewer.on_defect_scale_changed(value)
    
    def _on_defect_alpha_changed(self, value):
        """处理缺陷透明度滑块变化事件"""
        # 更新显示
        self.lbl_alpha_value.setText(f"{value}%")
        # 调用defect_previewer的处理函数
        if hasattr(self.parent, 'defect_previewer') and self.parent.defect_previewer:
            self.parent.defect_previewer.slider_defect_alpha.setValue(value)
            self.parent.defect_previewer.on_defect_alpha_changed(value)
    
    def _on_defect_angle_changed(self, value):
        """处理缺陷旋转滑块变化事件"""
        # 更新显示
        self.lbl_angle_value.setText(f"{value}°")
        # 调用defect_previewer的处理函数
        if hasattr(self.parent, 'defect_previewer') and self.parent.defect_previewer:
            self.parent.defect_previewer.slider_defect_angle.setValue(value)
            self.parent.defect_previewer.on_defect_angle_changed(value)
