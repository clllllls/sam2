import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QFileDialog, QMessageBox, QGroupBox, QSpinBox
)
from PyQt6.QtCore import Qt

from .defect_manager import DefectManager
from .layer_editor import LayerEditor
from .defect_previewer import DefectPreviewer
from .batch_generator import BatchGenerator
from .base import BaseBatchTransferComponent


class BatchTransferDialog(QDialog):
    """批量随机移植缺陷参数设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量随机移植缺陷")
        self.setGeometry(350, 60, 1200, 850)
        
        # 初始化变量
        self.current_selected_defect = None  # 当前选中的缺陷
        self.target_image_full_path = ""  # 目标图像完整路径
        self.save_path = ""  # 保存路径
        
        # 上次文件夹位置记忆
        self.last_target_dir = ""  # 上次选择目标图像的文件夹
        self.last_save_dir = ""  # 上次选择保存路径的文件夹
        
        # 初始化子模块
        self._init_modules()
        
        # 创建主布局
        self._init_main_layout()
        
        # 加载路径配置
        self._load_path_config()
    
    def _init_modules(self):
        """初始化子模块"""
        # 初始化层级编辑器
        self.layer_editor = LayerEditor(self)
        
        # 初始化缺陷管理器
        self.defect_manager = DefectManager(self)
        
        # 初始化缺陷预览器
        self.defect_previewer = DefectPreviewer(self, self.layer_editor.get_image_viewer())
        
        # 初始化批量生成器
        self.batch_generator = BatchGenerator(self)
    
    def _init_main_layout(self):
        """初始化主布局"""
        main_layout = QVBoxLayout(self)
        
        # 添加缺陷管理器组件
        main_layout.addWidget(self.defect_manager.get_widget())
        
        # 创建包含目标图像、生成参数和保存路径的水平布局
        combined_layout = QHBoxLayout()
        
        # 目标图像选择区域（左侧）
        target_group = QGroupBox("目标图像")
        target_layout = QHBoxLayout()
        
        self.target_image_path = QLineEdit()
        self.target_image_path.setReadOnly(True)
        target_layout.addWidget(self.target_image_path, 1)
        
        self.btn_select_target_image = QPushButton("加载图像")
        self.btn_select_target_image.clicked.connect(self.select_target_image)
        target_layout.addWidget(self.btn_select_target_image)
        
        target_group.setLayout(target_layout)
        combined_layout.addWidget(target_group, 1)
        
        # 生成参数区域（中间）
        params_group = QGroupBox("生成参数")
        params_layout = QVBoxLayout()
        
        # 生成数量数字框
        self.spin_generate_count = QSpinBox()
        self.spin_generate_count.setRange(1, 100)
        self.spin_generate_count.setValue(10)
        params_layout.addWidget(QLabel("生成数量:"))
        params_layout.addWidget(self.spin_generate_count)
        
        params_group.setLayout(params_layout)
        combined_layout.addWidget(params_group, 1)
        
        # 移植结果保存路径区域（右侧）
        save_group = QGroupBox("保存路径")
        save_layout = QHBoxLayout()
        
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setReadOnly(False)  # 设置为可编辑，允许用户直接输入路径
        self.save_path_edit.textChanged.connect(self._on_save_path_text_changed)
        save_layout.addWidget(self.save_path_edit, 1)
        
        self.btn_select_save_path = QPushButton("选择路径")
        self.btn_select_save_path.clicked.connect(self.select_save_path)
        save_layout.addWidget(self.btn_select_save_path)
        
        save_group.setLayout(save_layout)
        combined_layout.addWidget(save_group, 2)
        
        # 将组合布局添加到主布局
        main_layout.addLayout(combined_layout)
        
        # 添加层级编辑器组件
        main_layout.addWidget(self.layer_editor.get_widget())
        
        # 创建按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        self.btn_start = QPushButton("开始批量移植")
        self.btn_start.clicked.connect(self.on_start_batch_transfer)
        button_layout.addWidget(self.btn_start)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
    
    def select_target_image(self):
        """选择目标图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择目标图像", 
            self.last_target_dir if self.last_target_dir else ".", 
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        
        if file_path:
            # 存储选择的目标图像路径
            self.target_image_full_path = file_path
            # 更新上次目标文件夹位置
            self.last_target_dir = os.path.dirname(file_path)
            # 保存路径配置
            self._save_path_config()
            
            # 在路径输入框中显示选中的文件路径
            self.target_image_path.setText(file_path)
            
            # 在缺陷预览视图中加载目标图像
            self.layer_editor.get_image_viewer().set_image(file_path)
            
            # 显示当前层级（默认为第1层）
            self.layer_editor.on_layer_double_clicked(1)
        else:
            # 清除选择
            self.target_image_path.clear()
            self.target_image_full_path = ""
            self.layer_editor.get_image_viewer().clear_all()
    
    def select_save_path(self):
        """选择保存路径"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择保存路径", 
            self.save_path_edit.text() if self.save_path_edit.text() else ".",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        
        if dir_path:
            self.save_path_edit.setText(dir_path)
            self.last_save_dir = dir_path
            self._save_path_config()
    
    def _on_save_path_text_changed(self, text):
        """保存路径文本框内容改变时的处理"""
        # 当用户手动输入路径时，更新last_save_dir
        if text and os.path.isdir(text):
            self.last_save_dir = text
            self._save_path_config()
    
    def on_start_batch_transfer(self):
        """开始批量移植缺陷功能"""
        # 调用批量生成器生成缺陷图像
        self.batch_generator.generate_batch(
            self.target_image_full_path,
            self.defect_manager.selected_defect_paths,
            self.save_path_edit.text(),
            self.spin_generate_count.value(),
            self.layer_editor.layer_masks,
            self.layer_editor.layer_checkboxes,
            self.layer_editor.hollow_layers,
            self.layer_editor.chk_random_layer.isChecked(),
            self.layer_editor.chk_random_defect.isChecked(),
            self.layer_editor.spin_max_defects.value(),
            self.defect_manager.preview_defects,
            self.defect_manager.defect_selection_status
        )
    
    def _load_path_config(self):
        """从配置文件加载路径设置"""
        try:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "defect_configs", "path_config.json")
            if os.path.exists(config_file):
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.last_target_dir = config.get('last_target_dir', '')
                    self.last_save_dir = config.get('last_save_dir', '')
                    
                    # 如果有上次保存路径，显示在文本框中
                    if self.last_save_dir:
                        self.save_path_edit.setText(self.last_save_dir)
        except Exception as e:
            print(f"加载路径配置失败: {e}")
            # 如果加载失败，保持默认值
            pass
    
    def _save_path_config(self):
        """保存路径设置到配置文件"""
        try:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "defect_configs", "path_config.json")
            import json
            config = {
                'last_defect_dir': self.defect_manager.last_defect_dir if hasattr(self, 'defect_manager') else '',
                'last_target_dir': self.last_target_dir,
                'last_save_dir': self.last_save_dir
            }
            # 确保配置目录存在
            config_dir = os.path.dirname(config_file)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存路径配置失败: {e}")


# 导出类
export = BatchTransferDialog
