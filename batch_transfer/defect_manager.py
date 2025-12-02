import os
import json
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QListWidget, QAbstractItemView, QLineEdit, QGridLayout,
    QMessageBox, QListWidgetItem, QApplication
)
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import Qt
from .base import BaseBatchTransferComponent


class DefectManager(BaseBatchTransferComponent):
    """缺陷选择和管理组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # 初始化变量
        self.selected_defect_paths = []  # 选择的缺陷图像路径
        self.preview_defects = {}  # 存储每个缺陷的预览信息
        self.defect_selection_status = {}  # 存储缺陷选中状态的字典
        self.last_clicked_index = -1  # 上次点击的项目索引，用于Shift+点击功能
        
        # JSON配置文件保存目录
        self.defect_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "defect_configs")
        # 确保配置目录存在
        if not os.path.exists(self.defect_config_dir):
            os.makedirs(self.defect_config_dir)
        
        # 路径配置文件
        self.path_config_file = os.path.join(self.defect_config_dir, "path_config.json")
        
        # 上次文件夹位置记忆
        self.last_defect_dir = ""  # 上次选择缺陷图像的文件夹
        
        # 加载路径配置
        self._load_path_config()
        
        # 初始化UI组件
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建选择缺陷区域
        self.defect_group = QGroupBox("选择缺陷")
        self.defect_layout = QVBoxLayout()
        
        # 选择缺陷按钮
        self.btn_select_defect = QPushButton("选择缺陷图像")
        self.btn_select_defect.clicked.connect(self.select_defect)
        self.defect_layout.addWidget(self.btn_select_defect)
        
        # 创建缺陷列表和控制按钮的水平布局
        self.defect_control_layout = QHBoxLayout()
        
        # 缺陷类型筛选区域
        self.filter_group = QGroupBox("缺陷类型筛选")
        self.filter_layout = QVBoxLayout()
        
        # 缺陷类型下拉框
        self.filter_layout.addWidget(QLabel("选择缺陷类型:"))
        self.defect_type_filter = QComboBox()
        self.defect_type_filter.setMinimumWidth(150)
        self.defect_type_filter.addItem("所有缺陷")  # 默认选项
        self.defect_type_filter.addItem("未分类")    # 未分类选项
        self.defect_type_filter.currentIndexChanged.connect(self.on_defect_type_filter_changed)
        self.filter_layout.addWidget(self.defect_type_filter)
        
        self.filter_group.setLayout(self.filter_layout)
        self.defect_control_layout.addWidget(self.filter_group, 1)
        
        # 缺陷列表视图
        self.defect_list_group = QGroupBox("缺陷列表")
        self.defect_list_layout = QVBoxLayout()
        self.defect_list_view = QListWidget()
        self.defect_list_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.defect_list_view.itemDoubleClicked.connect(self.on_defect_double_clicked)
        self.defect_list_view.itemClicked.connect(self.on_defect_item_clicked)
        self.defect_list_layout.addWidget(self.defect_list_view)
        self.defect_list_group.setLayout(self.defect_list_layout)
        self.defect_control_layout.addWidget(self.defect_list_group, 2)
        
        # 缩放与透明度参数控制区域
        self.scale_control_group = QGroupBox("缩放与透明度参数")
        self.scale_control_layout = QVBoxLayout()
        
        # 当前缺陷信息
        self.lbl_current_defect = QLabel("当前缺陷: 无")
        self.scale_control_layout.addWidget(self.lbl_current_defect)
        
        # 创建3列2行的参数控制网格布局
        self.params_grid_layout = QGridLayout()
        self.params_grid_layout.setHorizontalSpacing(20)  # 设置水平间距
        self.params_grid_layout.setVerticalSpacing(10)    # 设置垂直间距
        
        # 第一列：缩放相关（左列）
        # 最小缩放
        scale_min_layout = QHBoxLayout()
        scale_min_layout.addWidget(QLabel("最小缩放:"))
        self.txt_defect_scale_min = QLineEdit("100")
        self.txt_defect_scale_min.setFixedWidth(60)
        self.txt_defect_scale_min.setValidator(QIntValidator(1, 500))
        scale_min_layout.addWidget(self.txt_defect_scale_min)
        scale_min_layout.addWidget(QLabel("%"))
        self.params_grid_layout.addLayout(scale_min_layout, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # 最大缩放
        scale_max_layout = QHBoxLayout()
        scale_max_layout.addWidget(QLabel("最大缩放:"))
        self.txt_defect_scale_max = QLineEdit("100")
        self.txt_defect_scale_max.setFixedWidth(60)
        self.txt_defect_scale_max.setValidator(QIntValidator(1, 500))
        scale_max_layout.addWidget(self.txt_defect_scale_max)
        scale_max_layout.addWidget(QLabel("%"))
        self.params_grid_layout.addLayout(scale_max_layout, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # 第二列：旋转相关（中间列）
        # 最小旋转
        angle_min_layout = QHBoxLayout()
        angle_min_layout.addWidget(QLabel("最小旋转:"))
        self.txt_defect_angle_min = QLineEdit("0")
        self.txt_defect_angle_min.setFixedWidth(60)
        self.txt_defect_angle_min.setValidator(QIntValidator(0, 360))
        angle_min_layout.addWidget(self.txt_defect_angle_min)
        angle_min_layout.addWidget(QLabel("°"))
        self.params_grid_layout.addLayout(angle_min_layout, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # 最大旋转
        angle_max_layout = QHBoxLayout()
        angle_max_layout.addWidget(QLabel("最大旋转:"))
        self.txt_defect_angle_max = QLineEdit("360")
        self.txt_defect_angle_max.setFixedWidth(60)
        self.txt_defect_angle_max.setValidator(QIntValidator(0, 360))
        angle_max_layout.addWidget(self.txt_defect_angle_max)
        angle_max_layout.addWidget(QLabel("°"))
        self.params_grid_layout.addLayout(angle_max_layout, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # 第三列：透明度相关（右列）
        # 最小透明度
        alpha_min_layout = QHBoxLayout()
        alpha_min_layout.addWidget(QLabel("最小透明度:"))
        self.txt_defect_alpha_min = QLineEdit("100")
        self.txt_defect_alpha_min.setFixedWidth(60)
        self.txt_defect_alpha_min.setValidator(QIntValidator(0, 100))
        alpha_min_layout.addWidget(self.txt_defect_alpha_min)
        alpha_min_layout.addWidget(QLabel("%"))
        self.params_grid_layout.addLayout(alpha_min_layout, 0, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # 最大透明度
        alpha_max_layout = QHBoxLayout()
        alpha_max_layout.addWidget(QLabel("最大透明度:"))
        self.txt_defect_alpha_max = QLineEdit("100")
        self.txt_defect_alpha_max.setFixedWidth(60)
        self.txt_defect_alpha_max.setValidator(QIntValidator(0, 100))
        alpha_max_layout.addWidget(self.txt_defect_alpha_max)
        alpha_max_layout.addWidget(QLabel("%"))
        self.params_grid_layout.addLayout(alpha_max_layout, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # 添加缺陷类型名称文本框（跨3列）
        defect_type_layout = QHBoxLayout()
        defect_type_layout.addWidget(QLabel("缺陷类型:"))
        self.txt_defect_type = QLineEdit("")
        self.txt_defect_type.setPlaceholderText("请输入缺陷类型名称")
        self.txt_defect_type.setMinimumWidth(200)
        defect_type_layout.addWidget(self.txt_defect_type)
        self.params_grid_layout.addLayout(defect_type_layout, 2, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignLeft)
        
        self.scale_control_layout.addLayout(self.params_grid_layout)
        
        # 保存按钮
        self.btn_save_defect_scale = QPushButton("保存缩放设置")
        self.btn_save_defect_scale.clicked.connect(self.save_defect_scale_settings)
        self.btn_save_defect_scale.setEnabled(False)
        self.scale_control_layout.addWidget(self.btn_save_defect_scale)
        
        self.scale_control_layout.addStretch()
        self.scale_control_group.setLayout(self.scale_control_layout)
        self.defect_control_layout.addWidget(self.scale_control_group, 1)
        
        self.defect_layout.addLayout(self.defect_control_layout)
        self.defect_group.setLayout(self.defect_layout)
    
    def get_widget(self):
        """获取组件的主widget"""
        return self.defect_group
    
    def select_defect(self):
        """选择缺陷图像并在目标图像上预览"""
        from PyQt6.QtWidgets import QFileDialog
        
        # 允许用户选择多张图像（PyQt6版本）
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent, 
            "选择缺陷图像", 
            self.last_defect_dir if self.last_defect_dir else ".", 
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        
        if file_paths:
            # 存储选择的缺陷图像路径
            self.selected_defect_paths = file_paths
            # 更新上次缺陷文件夹位置
            self.last_defect_dir = os.path.dirname(file_paths[0])
            # 保存路径配置
            self._save_path_config()
            
            # 初始化存储缺陷选中状态的字典（如果不存在）
            if not hasattr(self, 'defect_selection_status'):
                self.defect_selection_status = {}
            
            # 更新缺陷列表
            self.defect_list_view.clear()
            for path in file_paths:
                defect_name = os.path.basename(path)
                item = QListWidgetItem(defect_name)

                # 设置项为可勾选
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # 设置初始勾选状态
                if defect_name in self.defect_selection_status:
                    # 如果之前有保存的状态，使用保存的状态
                    if self.defect_selection_status[defect_name]:
                        item.setCheckState(Qt.CheckState.Checked)
                    else:
                        item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)  # 默认选中
                    self.defect_selection_status[defect_name] = True
                
                # 初始化预览信息
                if defect_name not in self.preview_defects:
                    self.preview_defects[defect_name] = {
                        'path': path,
                        'scale_min': 100,  # 默认最小缩放100%
                        'scale_max': 100,  # 默认最大缩放100%
                        'alpha_min': 100,  # 默认最小透明度100%
                        'alpha_max': 100,  # 默认最大透明度100%
                        'angle_min': 0,    # 默认最小旋转0°
                        'angle_max': 360,  # 默认最大旋转360°
                        'selected': True    # 默认选中该缺陷用于批量生成
                    }
                
                # 尝试加载缺陷的JSON配置文件
                self._load_defect_config(defect_name)
                
                # 添加到列表
                self.defect_list_view.addItem(item)
            
            # 为缺陷列表添加事件处理
            self.defect_list_view.itemClicked.connect(self.on_defect_item_clicked)
            # 为缺陷列表添加键盘事件处理
            self.defect_list_view.keyPressEvent = self.on_defect_list_key_press
            
            # 更新缺陷类型筛选下拉框
            self.update_defect_type_filter()
            # 默认显示所有缺陷
            self.filter_defect_list("所有缺陷")
        else:
            # 清除选择
            if hasattr(self, 'selected_defect_paths'):
                delattr(self, 'selected_defect_paths')
            # 清除缺陷列表，但保留选中状态字典
            self.defect_list_view.clear()
            self.parent.current_selected_defect = None
            self.lbl_current_defect.setText("当前缺陷: 无")
            self.btn_save_defect_scale.setEnabled(False)
    
    def on_defect_list_key_press(self, event):
        """处理缺陷列表的键盘事件，支持Ctrl+A全选"""
        from PyQt6.QtCore import QApplication, Qt
        
        # 处理Ctrl+A全选
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_A:
            for i in range(self.defect_list_view.count()):
                item = self.defect_list_view.item(i)
                item.setCheckState(Qt.CheckState.Checked)
                # 更新选中状态字典
                defect_name = item.text()
                self.defect_selection_status[defect_name] = True
            event.accept()
        else:
            # 调用默认的键盘事件处理
            super(QListWidget, self.defect_list_view).keyPressEvent(event)
    
    def on_defect_item_clicked(self, item):
        """处理缺陷列表项的点击事件，支持Shift+点击连选和Ctrl+点击多选"""
        
        # 获取当前点击的项目索引
        current_index = self.defect_list_view.row(item)
        
        # 检查是否按下了Shift键（连选）
        if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier and self.last_clicked_index != -1:
            # 确定选择范围
            start_index = min(self.last_clicked_index, current_index)
            end_index = max(self.last_clicked_index, current_index)
            
            # 获取最后一次点击项的勾选状态作为目标状态
            last_item = self.defect_list_view.item(self.last_clicked_index)
            target_state = last_item.checkState()
            
            # 更新范围内所有项目的勾选状态
            for i in range(start_index, end_index + 1):
                list_item = self.defect_list_view.item(i)
                list_item.setCheckState(target_state)
                # 更新选中状态字典
                defect_name = list_item.text()
                self.defect_selection_status[defect_name] = (target_state == Qt.CheckState.Checked)
        
        # 记录当前点击的索引
        self.last_clicked_index = current_index
        
        # 更新选中状态字典
        defect_name = item.text()
        self.defect_selection_status[defect_name] = (item.checkState() == Qt.CheckState.Checked)
        
        # 如果不是Shift或Ctrl组合键，则正常处理预览和UI更新
        if QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier:
            # 更新当前选中的缺陷
            self.parent.current_selected_defect = defect_name
            self.update_defect_scale_ui(defect_name)
            # 通知缺陷预览器更新选中的缺陷
            if hasattr(self.parent, 'defect_previewer') and self.parent.defect_previewer:
                self.parent.defect_previewer.update_selected_defect(defect_name)
            # 启用保存按钮
            self.btn_save_defect_scale.setEnabled(True)
    
    def update_defect_scale_ui(self, defect_name):
        """更新UI显示当前缺陷的缩放、透明度和旋转参数以及缺陷类型"""
        self.lbl_current_defect.setText(f"当前缺陷: {defect_name}")
        
        if defect_name in self.preview_defects:
            scale_min = self.preview_defects[defect_name]['scale_min']
            scale_max = self.preview_defects[defect_name]['scale_max']
            alpha_min = self.preview_defects[defect_name].get('alpha_min', 100)
            alpha_max = self.preview_defects[defect_name].get('alpha_max', 100)
            angle_min = self.preview_defects[defect_name].get('angle_min', 0)
            angle_max = self.preview_defects[defect_name].get('angle_max', 360)
            defect_type = self.preview_defects[defect_name].get('defect_type', '')
            
            self.txt_defect_scale_min.setText(str(scale_min))
            self.txt_defect_scale_max.setText(str(scale_max))
            self.txt_defect_alpha_min.setText(str(alpha_min))
            self.txt_defect_alpha_max.setText(str(alpha_max))
            self.txt_defect_angle_min.setText(str(angle_min))
            self.txt_defect_angle_max.setText(str(angle_max))
            self.txt_defect_type.setText(defect_type)
    
    def on_defect_double_clicked(self, item):
        """处理缺陷列表项双击事件"""
        if not item or not hasattr(self.parent, 'target_image_full_path') or not self.parent.target_image_full_path:
            return
        
        # 获取双击的缺陷名称
        defect_name = item.text()
        self.parent.current_selected_defect = defect_name
        
        # 保存当前选中状态
        if hasattr(self, 'defect_selection_status'):
            self.defect_selection_status[defect_name] = item.checkState() == Qt.CheckState.Checked
        
        # 更新UI显示当前缺陷的缩放参数
        self.update_defect_scale_ui(defect_name)
        
        # 通知缺陷预览器更新选中的缺陷
        if hasattr(self.parent, 'defect_previewer') and self.parent.defect_previewer:
            self.parent.defect_previewer.update_selected_defect(defect_name)
        
        # 启用保存按钮
        self.btn_save_defect_scale.setEnabled(True)
    
    def update_defect_type_filter(self):
        """更新缺陷类型筛选下拉框"""
        # 保存当前选择的类型
        current_type = self.defect_type_filter.currentText()
        
        # 清空下拉框，保留前两个选项
        while self.defect_type_filter.count() > 2:
            self.defect_type_filter.removeItem(2)
        
        # 收集所有唯一的缺陷类型
        defect_types = set()
        for defect_name, defect_info in self.preview_defects.items():
            if 'defect_type' in defect_info and defect_info['defect_type']:
                defect_types.add(defect_info['defect_type'])
        
        # 添加收集到的缺陷类型
        for defect_type in sorted(defect_types):
            self.defect_type_filter.addItem(defect_type)
        
        # 恢复之前的选择，如果不存在则选择"所有缺陷"
        index = self.defect_type_filter.findText(current_type)
        if index >= 0:
            self.defect_type_filter.setCurrentIndex(index)
        else:
            self.defect_type_filter.setCurrentIndex(0)
    
    def on_defect_type_filter_changed(self, index):
        """处理缺陷类型筛选下拉框变化事件"""
        selected_type = self.defect_type_filter.currentText()
        self.filter_defect_list(selected_type)
    
    def filter_defect_list(self, defect_type):
        """根据缺陷类型筛选缺陷列表"""
        # 保存所有项的勾选状态
        selection_status = {}
        for i in range(self.defect_list_view.count()):
            item = self.defect_list_view.item(i)
            selection_status[item.text()] = item.checkState() == Qt.CheckState.Checked
        
        # 清空列表
        self.defect_list_view.clear()
        
        # 根据选择的类型添加缺陷
        if hasattr(self, 'selected_defect_paths'):
            for path in self.selected_defect_paths:
                defect_name = os.path.basename(path)
                
                # 判断是否符合筛选条件
                if defect_type == "所有缺陷":
                    # 显示所有缺陷
                    include = True
                elif defect_type == "未分类":
                    # 显示未分类的缺陷
                    defect_info = self.preview_defects.get(defect_name, {})
                    include = 'defect_type' not in defect_info or not defect_info['defect_type']
                else:
                    # 显示特定类型的缺陷
                    defect_info = self.preview_defects.get(defect_name, {})
                    include = 'defect_type' in defect_info and defect_info['defect_type'] == defect_type
                
                if include:
                    item = QListWidgetItem(defect_name)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    
                    # 恢复勾选状态
                    if defect_name in selection_status:
                        if selection_status[defect_name]:
                            item.setCheckState(Qt.CheckState.Checked)
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)
                    elif hasattr(self, 'defect_selection_status') and defect_name in self.defect_selection_status:
                        if self.defect_selection_status[defect_name]:
                            item.setCheckState(Qt.CheckState.Checked)
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)
                    else:
                        item.setCheckState(Qt.CheckState.Checked)
                    
                    self.defect_list_view.addItem(item)
    
    def save_defect_scale_settings(self):
        """保存当前缺陷的缩放、透明度和旋转设置"""
        if not self.parent.current_selected_defect:
            return
        
        try:
            # 获取当前输入的缩放、透明度和旋转参数
            scale_min = int(self.txt_defect_scale_min.text())
            scale_max = int(self.txt_defect_scale_max.text())
            alpha_min = int(self.txt_defect_alpha_min.text())
            alpha_max = int(self.txt_defect_alpha_max.text())
            angle_min = int(self.txt_defect_angle_min.text())
            angle_max = int(self.txt_defect_angle_max.text())
            
            # 确保参数有效
            scale_min = max(1, min(scale_min, 500))
            scale_max = max(1, min(scale_max, 500))
            alpha_min = max(0, min(alpha_min, 100))
            alpha_max = max(0, min(alpha_max, 100))
            angle_min = max(0, min(angle_min, 360))
            angle_max = max(0, min(angle_max, 360))
            
            # 确保最小值不大于最大值
            if scale_min > scale_max:
                scale_max = scale_min
                self.txt_defect_scale_max.setText(str(scale_max))
            if alpha_min > alpha_max:
                alpha_max = alpha_min
                self.txt_defect_alpha_max.setText(str(alpha_max))
            if angle_min > angle_max:
                angle_max = angle_min
                self.txt_defect_angle_max.setText(str(angle_max))
            
            # 更新缺陷信息
            if self.parent.current_selected_defect in self.preview_defects:
                self.preview_defects[self.parent.current_selected_defect]['scale_min'] = scale_min
                self.preview_defects[self.parent.current_selected_defect]['scale_max'] = scale_max
                self.preview_defects[self.parent.current_selected_defect]['alpha_min'] = alpha_min
                self.preview_defects[self.parent.current_selected_defect]['alpha_max'] = alpha_max
                self.preview_defects[self.parent.current_selected_defect]['angle_min'] = angle_min
                self.preview_defects[self.parent.current_selected_defect]['angle_max'] = angle_max
                
                # 获取缺陷类型名称
                defect_type = self.txt_defect_type.text().strip()
                # 同时更新预览缺陷字典中的缺陷类型
                self.preview_defects[self.parent.current_selected_defect]['defect_type'] = defect_type
                
                # 保存配置到JSON文件
                config = {
                    'scale_min': scale_min,
                    'scale_max': scale_max,
                    'alpha_min': alpha_min,
                    'alpha_max': alpha_max,
                    'angle_min': angle_min,
                    'angle_max': angle_max,
                    'defect_type': defect_type
                }
                self._save_defect_config(self.parent.current_selected_defect, config)
                
                # 显示保存成功消息
                QMessageBox.information(self.parent, "保存成功", \
                                       f"已保存缺陷'{self.parent.current_selected_defect}' 的设置\n" +
                                       f"最小缩放: {scale_min}%，最大缩放: {scale_max}%\n" +
                                       f"最小透明度: {alpha_min}%，最大透明度: {alpha_max}%\n" +
                                       f"最小旋转: {angle_min}°，最大旋转: {angle_max}°\n" +
                                       f"配置已保存到: {os.path.join(self.defect_config_dir, f'{self.parent.current_selected_defect}.json')}")
                
                # 更新缺陷类型筛选下拉框
                self.update_defect_type_filter()
                # 重新应用筛选
                current_type = self.defect_type_filter.currentText()
                self.filter_defect_list(current_type)
        except ValueError:
            QMessageBox.warning(self.parent, "输入错误", "请输入有效的百分比数值")
    
    def _load_defect_config(self, defect_name):
        """加载缺陷的JSON配置文件"""
        # 创建配置文件路径
        config_file = os.path.join(self.defect_config_dir, f"{defect_name}.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新预览缺陷信息
                if 'scale_min' in config:
                    self.preview_defects[defect_name]['scale_min'] = config['scale_min']
                if 'scale_max' in config:
                    self.preview_defects[defect_name]['scale_max'] = config['scale_max']
                if 'alpha_min' in config:
                    self.preview_defects[defect_name]['alpha_min'] = config['alpha_min']
                if 'alpha_max' in config:
                    self.preview_defects[defect_name]['alpha_max'] = config['alpha_max']
                if 'angle_min' in config:
                    self.preview_defects[defect_name]['angle_min'] = config['angle_min']
                if 'angle_max' in config:
                    self.preview_defects[defect_name]['angle_max'] = config['angle_max']
                if 'defect_type' in config:
                    self.preview_defects[defect_name]['defect_type'] = config['defect_type']
                
                # 如果当前选中的是这个缺陷，更新UI
                if self.parent.current_selected_defect == defect_name:
                    self.update_defect_scale_ui(defect_name)
                    
            except Exception as e:
                print(f"加载缺陷 {defect_name} 配置失败: {str(e)}")
    
    def _save_defect_config(self, defect_name, config):
        """保存缺陷配置到JSON文件"""
        # 创建配置文件路径
        config_file = os.path.join(self.defect_config_dir, f"{defect_name}.json")
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缺陷 {defect_name} 配置失败: {str(e)}")
            QMessageBox.warning(self.parent, "保存配置失败", f"无法保存缺陷配置到文件: {str(e)}")
    
    def _load_path_config(self):
        """从配置文件加载路径设置"""
        try:
            if os.path.exists(self.path_config_file):
                with open(self.path_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.last_defect_dir = config.get('last_defect_dir', '')
        except Exception as e:
            print(f"加载路径配置失败: {e}")
            # 如果加载失败，保持默认值
            pass
    
    def _save_path_config(self):
        """保存路径设置到配置文件"""
        try:
            config = {
                'last_defect_dir': self.last_defect_dir,
                'last_target_dir': getattr(self.parent, 'last_target_dir', ''),
                'last_save_dir': getattr(self.parent, 'last_save_dir', '')
            }
            # 确保配置目录存在
            if not os.path.exists(self.defect_config_dir):
                os.makedirs(self.defect_config_dir)
            # 保存配置
            with open(self.path_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存路径配置失败: {e}")
