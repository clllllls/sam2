import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image, ImageQt
import time
import platform

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar,
    QMessageBox, QSplitter, QTabWidget, QGroupBox, QGridLayout, 
    QToolButton, QMenu, QSpinBox, QDoubleSpinBox, QSlider,
    QLineEdit, QTextEdit, QCheckBox, QFormLayout, QDialog, QButtonGroup,
    QListWidget, QAbstractItemView, QListWidgetItem
)
from PyQt6.QtGui import (
    QFont, QPixmap, QPainter, QPen, QColor, QBrush, QIcon, QCursor, 
    QImage, QPalette, QRegion, QFontDatabase, QBitmap, QIntValidator
)
from PyQt6.QtCore import (
    Qt, QPoint, QRect, QSize, pyqtSignal, pyqtSlot, QTimer
)

# SAM2 imports
sys.path.append('..')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


try:
    from defect_transfer import DefectTransferNew as DefectTransfer
    print("成功导入缺陷移植模块")
except ImportError as e3:
    print(f"导入缺陷移植模块失败: {e3}")
    # 创建虚拟类
    class DefectTransfer:
        def __init__(self):
            pass
        def load_images(self, *args):
            return False
        def set_defect_mask(self, *args):
            pass
        def blend_defect(self, *args):
            return None

# 导入批量移植对话框
from batch_transfer import BatchTransferDialog
print("成功导入批量移植对话框模块")

from image_viewer import ImageViewer
print("成功导入图像画布模块")

class SAM2PyQtApp(QMainWindow):
    """SAM2 交互式缺陷分割工具主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 2 交互式缺陷分割工具")
        self.setGeometry(250, 150, 1000, 800)
        
        # 设置中文字体支持
        self._setup_fonts()
        
        # 初始化变量
        self.image_path = ""
        self.device = None
        self.model = None
        self.predictor = None
        self.current_model_size = "base_plus"
        self.trained_model_path = ""
        self.masks = []
        self.scores = []
        self.current_mask_idx = 0
        
        # 使用单一颜色表示分割区域
        self.mask_color = [255, 0, 0, 100]  # 统一使用红色
        
        # 缺陷移植相关变量
        self.target_image_path = ""  # 目标图像路径
        self.defect_transfer = DefectTransfer()  # 缺陷移植工具
        self.transfer_result = None  # 移植结果图像
        self.transfer_mode = False   # 是否处于移植模式
        self.defect_image_path = ""  # 透明背景缺陷图像路径
        self.preview_defect_image = None  # 预览用的缺陷图像（带透明度）
        self.preview_composite = None  # 预览合成图像
        
        # 上次文件夹位置记忆
        self.last_image_dir = "."  # 上次加载图像的文件夹
        self.last_target_image_dir = "."  # 上次加载目标图像的文件夹
        self.last_model_dir = "."  # 上次加载模型的文件夹
        self.last_defect_dir = "defects"  # 上次选择缺陷图像的文件夹
        
        # 防抖机制相关变量
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._on_debounce_timeout)
        self.pending_transfer = False  # 是否有待处理的移植操作
        self.last_slider_values = {
            'x': 0,
            'y': 0, 
            'scale': 100,
            'rotate': 0
        }
        
        # 创建UI
        self._init_ui()
        
        # 初始化设备
        self._init_device()
    
    def show_batch_transfer_dialog(self):
        """显示批量随机移植缺陷对话框"""
        dialog = BatchTransferDialog(self)
        if dialog.exec() == QDialog.accepted:
            # 批量移植操作将在后续实现
            pass
        
    def _setup_fonts(self):
        """设置字体支持中文"""
        # 在PyQt中，默认字体通常已经支持中文，但为了确保兼容性，我们可以设置字体策略
        font = self.font()
        font_families = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
        
        # 尝试设置一个支持中文的字体
        for family in font_families:
            font.setFamily(family)
            if self._is_font_available(family):
                self.setFont(font)
                break
        
    def _is_font_available(self, font_name):
        """检查字体是否可用"""
        available_fonts = QFontDatabase.families()
        return font_name in available_fonts
    
    def _init_ui(self):
        """初始化用户界面"""
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建顶部工具栏
        self._create_toolbar(main_layout)
        
        # 创建水平分割器，用于分隔两个图像查看器
        image_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 创建左侧图像查看器（缺陷图像）
        self.image_viewer = ImageViewer(self)
        self.image_viewer.mouse_clicked.connect(self.on_mouse_clicked)
        self.image_viewer.mouse_dragged.connect(self.on_mouse_dragged)
        self.image_viewer.contour_updated.connect(self.on_contour_updated)
        image_splitter.addWidget(self.image_viewer)
        
        # 创建右侧图像查看器（目标图像）
        self.target_image_viewer = ImageViewer(self)
        self.target_image_viewer.set_mode(0)  # 设置为平移模式
        image_splitter.addWidget(self.target_image_viewer)
        
        # 设置图像分割器比例，左右各占50%
        image_splitter.setSizes([500, 500])
        
        # 创建垂直分割器，用于分隔图像查看器和控制面板
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(image_splitter)
        
        # 创建右侧控制面板
        self._create_control_panel(splitter)
        
        # 设置分割器比例，增大图像查看区域高度
        splitter.setSizes([600, 300])
        
        # 设置主窗口最小尺寸，确保图像查看区域有足够高度
        self.setMinimumSize(1400, 800)
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
        
    def _create_toolbar(self, parent_layout):
        """创建顶部工具栏"""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar_layout.setSpacing(10)
        
        # 加载图像按钮
        self.btn_load_image = QPushButton("加载图像")
        self.btn_load_image.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.btn_load_image)
        
        # 加载模型按钮
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        toolbar_layout.addWidget(self.btn_load_model)
        
        # 批量随机移植缺陷按钮
        self.btn_batch_transfer = QPushButton("批量移植缺陷")
        self.btn_batch_transfer.clicked.connect(self.show_batch_transfer_dialog)
        toolbar_layout.addWidget(self.btn_batch_transfer)
        
        # 模型大小选择
        toolbar_layout.addWidget(QLabel("模型大小:"))
        self.cb_model_size = QComboBox()
        self.cb_model_size.addItems(["tiny", "small", "base_plus", "large"])
        self.cb_model_size.setCurrentText("base_plus")
        self.cb_model_size.currentTextChanged.connect(self.on_model_size_changed)
        toolbar_layout.addWidget(self.cb_model_size)
        
        # 当前加载的模型信息
        self.lbl_model_info = QLabel("未加载模型")
        self.lbl_model_info.setStyleSheet("color: gray;")
        toolbar_layout.addWidget(self.lbl_model_info)
        
        # 帮助按钮
        self.btn_help = QPushButton("帮助")
        self.btn_help.clicked.connect(self.show_help)
        toolbar_layout.addWidget(self.btn_help)

        # 填充空间
        toolbar_layout.addStretch()
        
        # 添加工具栏到主布局
        parent_layout.addWidget(toolbar_widget)
        
    def show_help(self):
        """显示帮助信息"""
        help_text = (
            "SAM2 交互式缺陷分割工具使用说明\n\n"
            "基本操作:\n"
            "1. 加载图像: 点击'加载图像'按钮选择要分割的图像\n"
            "2. 加载模型: 点击'加载模型'按钮选择模型文件\n"
            "3. 选择操作模式:\n"
            "   - 平移: 拖动图像查看不同区域\n"
            "   - 点选: 左键点击添加前景点，右键点击添加背景点\n"
            "   - 框选: 拖动鼠标创建矩形框选区域\n"
            "   - 轮廓: 点击添加轮廓点，右键闭合轮廓\n"
            "4. 执行分割: 点击'执行分割'按钮\n"
            "5. 清楚最后点：点击'清楚最后点'按钮清除最后添加的点\n"
            "6. 切换Mask: 使用'前一个掩码'和'后一个掩码'按钮切换不同掩码\n"
            "7. 重置所有：点击'重置所有'重置所有提示区域和提示点\n\n"
            "8. 缺损移植：\n"
            "   - 点击'加载目标'按钮，加载想要移植的目标图像\n"
            "   - 点击'选择缺陷'按钮，加载进行移植的对象缺陷\n"
            "   - 点击'保存缺陷'按钮，保存当前分割完成的缺陷为单独png图像\n"
            "   - 点击'缺陷融合'按钮，融合当前缺陷与目标图像为完整的一张图像并保存\n\n"
            "9. 位置调整与变换调整：\n"
            "   - 拖动'X''Y'滑块，调整当前缺陷的位置\n"
            "   - 拖动'缩放''旋转'滑块，调整当前缺陷的变换属性（旋转、缩放）\n"
            "   - 拖动'透明度'滑块，调整当前缺陷的透明度属性\n\n"
            "点选叠加功能:\n"
            "- 在使用框选或轮廓选后，可以切换到点选模式\n"
            "- 在框选或轮廓区域内点击添加点提示，以优化分割结果\n"
            "- 分割时会同时使用框提示和点提示\n\n"
            "提示:\n"
            "- 轮廓会自动转换为框提示\n"
            "- 状态栏会显示当前使用的提示类型和数量"
        )
        QMessageBox.information(self, "使用帮助", help_text)

    def _create_control_panel(self, splitter):
        """创建右侧控制面板"""
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 创建水平布局容器，左边显示操作模式，右边显示操作按钮
        main_horizontal_layout = QHBoxLayout()
        
        # 左侧：操作模式选择（宽度减少一半）
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        mode_group = QGroupBox("操作模式")
        mode_layout = QGridLayout()
        mode_layout.setSpacing(5)  # 减少间距
        
        self.btn_mode_pan = QPushButton("平移")
        self.btn_mode_pan.setCheckable(True)
        self.btn_mode_pan.setFixedHeight(30)  # 固定高度
        self.btn_mode_pan.clicked.connect(lambda: self.set_mode(0))
        mode_layout.addWidget(self.btn_mode_pan, 0, 0)
        
        self.btn_mode_point = QPushButton("点选")
        self.btn_mode_point.setCheckable(True)
        self.btn_mode_point.setChecked(True)
        self.btn_mode_point.setFixedHeight(30)
        self.btn_mode_point.clicked.connect(lambda: self.set_mode(1))
        mode_layout.addWidget(self.btn_mode_point, 0, 1)
        
        self.btn_mode_box = QPushButton("框选")
        self.btn_mode_box.setCheckable(True)
        self.btn_mode_box.setFixedHeight(30)
        self.btn_mode_box.clicked.connect(lambda: self.set_mode(2))
        mode_layout.addWidget(self.btn_mode_box, 1, 0)
        
        self.btn_mode_contour = QPushButton("轮廓")
        self.btn_mode_contour.setCheckable(True)
        self.btn_mode_contour.setFixedHeight(30)
        self.btn_mode_contour.clicked.connect(lambda: self.set_mode(3))
        mode_layout.addWidget(self.btn_mode_contour, 1, 1)
        
        # 创建按钮组，确保只能选择一个模式
        self.mode_buttons = [
            self.btn_mode_pan, 
            self.btn_mode_point, 
            self.btn_mode_box, 
            self.btn_mode_contour
        ]
        
        mode_group.setLayout(mode_layout)
        left_layout.addWidget(mode_group)
        
        # 提示信息
        info_label = QLabel("提示: 直接框选区域进行缺陷分割")
        info_label.setStyleSheet("color: blue; font-weight: bold; font-size: 9px;")
        info_label.setWordWrap(True)
        left_layout.addWidget(info_label)
        
        # 右侧：操作按钮
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 操作按钮组
        action_group = QGroupBox("操作")
        action_layout = QGridLayout()
        action_layout.setSpacing(5)
        
        self.btn_segment = QPushButton("执行分割")
        self.btn_segment.setFixedHeight(30)
        self.btn_segment.clicked.connect(self.segment_image)
        action_layout.addWidget(self.btn_segment, 0, 0)
        
        self.btn_clear_last = QPushButton("清除最后点")
        self.btn_clear_last.setFixedHeight(30)
        self.btn_clear_last.clicked.connect(self.clear_last_point)
        action_layout.addWidget(self.btn_clear_last, 0, 1)
        
        self.btn_toggle_mask = QPushButton("切换Mask")
        self.btn_toggle_mask.setFixedHeight(30)
        self.btn_toggle_mask.clicked.connect(self.toggle_mask)
        action_layout.addWidget(self.btn_toggle_mask, 1, 0)
        
        self.btn_reset = QPushButton("重置所有")
        self.btn_reset.setFixedHeight(30)
        self.btn_reset.clicked.connect(self.reset_all)
        action_layout.addWidget(self.btn_reset, 1, 1)
        
        action_group.setLayout(action_layout)
        right_layout.addWidget(action_group)
        
        # 将左右两部分添加到水平布局
        main_horizontal_layout.addWidget(left_widget, 1)  # 左边占1份
        main_horizontal_layout.addWidget(right_widget, 1)  # 右边占1份
        
        # 将水平布局添加到主垂直布局
        control_layout.addLayout(main_horizontal_layout)
        
        # 缺陷移植功能组（滑块布局）
        transfer_group = QGroupBox("缺陷移植")
        transfer_layout = QGridLayout()
        transfer_layout.setSpacing(8)
        
        # 第一行：加载目标和选择缺陷按钮
        self.btn_load_target = QPushButton("加载目标")
        self.btn_load_target.clicked.connect(self.load_target_image)
        transfer_layout.addWidget(self.btn_load_target, 0, 0, 1, 2)
        
        self.btn_select_defect = QPushButton("选择缺陷")
        self.btn_select_defect.clicked.connect(self.select_defect_image)
        self.btn_select_defect.setEnabled(True)
        transfer_layout.addWidget(self.btn_select_defect, 0, 2, 1, 2)
        
        # 第二行：保存缺陷和融合按钮
        self.btn_save_defect = QPushButton("保存缺陷")
        self.btn_save_defect.clicked.connect(self.save_defect_image)
        self.btn_save_defect.setEnabled(False)  # 初始禁用
        transfer_layout.addWidget(self.btn_save_defect, 1, 0, 1, 2)
        
        self.btn_final_fusion = QPushButton("缺陷融合")
        self.btn_final_fusion.clicked.connect(self.final_fusion)
        self.btn_final_fusion.setEnabled(False)
        transfer_layout.addWidget(self.btn_final_fusion, 1, 2, 1, 2)
        
        # 第三行：位置调整（左侧）- 使用滑块
        pos_group = QGroupBox("位置调整")
        pos_layout = QVBoxLayout()
        
        # X轴滑块
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.setRange(-1000, 1000)  # 扩展范围以支持0.1精度
        self.slider_x.setValue(0)
        self.slider_x.setSingleStep(1)  # 单个步长
        self.slider_x.setPageStep(10)   # 页步长（鼠标点击滑块轨道时）
        self.slider_x.valueChanged.connect(self.on_position_slider_changed)
        x_layout.addWidget(self.slider_x)
        self.lbl_x = QLabel("0.0")
        self.lbl_x.setFixedWidth(40)
        x_layout.addWidget(self.lbl_x)
        pos_layout.addLayout(x_layout)
        
        # Y轴滑块
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setRange(-1000, 1000)  # 扩展范围以支持0.1精度
        self.slider_y.setValue(0)
        self.slider_y.setSingleStep(1)  # 单个步长
        self.slider_y.setPageStep(10)   # 页步长
        self.slider_y.valueChanged.connect(self.on_position_slider_changed)
        y_layout.addWidget(self.slider_y)
        self.lbl_y = QLabel("0.0")
        self.lbl_y.setFixedWidth(40)
        y_layout.addWidget(self.lbl_y)
        pos_layout.addLayout(y_layout)
        
        pos_group.setLayout(pos_layout)
        transfer_layout.addWidget(pos_group, 3, 0, 2, 2)
        
        # 第五行：缩放和旋转（右侧）- 使用滑块
        transform_group = QGroupBox("变换调整")
        transform_layout = QVBoxLayout()
        
        # 缩放滑块
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("缩放:"))
        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setRange(1, 1000)   # 0.01-5.0 映射为 1-500
        self.slider_scale.setValue(100)      # 1.0
        self.slider_scale.setSingleStep(1)   # 单个步长
        self.slider_scale.setPageStep(10)    # 页步长
        self.slider_scale.valueChanged.connect(self.on_transform_slider_changed)
        scale_layout.addWidget(self.slider_scale)
        self.lbl_scale = QLabel("1.0")
        self.lbl_scale.setFixedWidth(40)
        scale_layout.addWidget(self.lbl_scale)
        transform_layout.addLayout(scale_layout)
        
        # 旋转滑块
        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(QLabel("旋转:"))
        self.slider_rotate = QSlider(Qt.Orientation.Horizontal)
        self.slider_rotate.setRange(-180, 180)  # -180.0到180.0度，步长0.1度
        self.slider_rotate.setValue(0)
        self.slider_rotate.setSingleStep(1)   # 单个步长
        self.slider_rotate.setPageStep(10)    # 页步长
        self.slider_rotate.valueChanged.connect(self.on_transform_slider_changed)
        rotate_layout.addWidget(self.slider_rotate)
        self.lbl_rotate = QLabel("0.0°")
        self.lbl_rotate.setFixedWidth(50)
        rotate_layout.addWidget(self.lbl_rotate)
        transform_layout.addLayout(rotate_layout)
        

        
        transform_group.setLayout(transform_layout)
        transfer_layout.addWidget(transform_group, 3, 2, 2, 2)
        
        # 第六行：透明度
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(100)
        self.slider_alpha.valueChanged.connect(self.update_transfer_alpha)
        alpha_layout.addWidget(self.slider_alpha)
        self.lbl_alpha = QLabel("100%")
        self.lbl_alpha.setFixedWidth(40)
        alpha_layout.addWidget(self.lbl_alpha)
        transfer_layout.addLayout(alpha_layout, 5, 0, 1, 4)
        
        transfer_group.setLayout(transfer_layout)
        control_layout.addWidget(transfer_group)
        
        # 常驻进度条 - 显示各种变换的执行进度
        progress_group = QGroupBox("执行进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.update_progress(0, "准备就绪")
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        control_layout.addWidget(progress_group)
        
        # 填充空间
        control_layout.addStretch()
        
        # 添加控制面板到分割器
        splitter.addWidget(control_panel)
        
    def _init_device(self):
        """初始化设备（CPU/CUDA）"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.update_status(f"使用GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.update_status("使用MPS设备")
        else:
            self.device = torch.device("cpu")
            self.update_status("使用CPU")
        
    def load_image(self):
        """加载图像"""
        logger.info("开始加载图像")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开图像文件", self.last_image_dir, 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            logger.info(f"选择的图像文件: {file_path}")
            self.image_path = file_path
            # 更新上次图像文件夹位置
            self.last_image_dir = os.path.dirname(file_path)
            success = self.image_viewer.set_image(file_path)
            if success:
                self.update_status(f"已加载图像: {os.path.basename(file_path)}")
                logger.info(f"已加载图像: {os.path.basename(file_path)}")
                # 如果已经加载了模型，设置图像到预测器
                if self.predictor is not None:
                    try:
                        image = np.array(Image.open(file_path).convert("RGB"))
                        logger.info(f"图像类型: {type(image)}, 形状: {image.shape}")
                        logger.info(f"预测器类型: {type(self.predictor)}")
                        # 检查predictor是否有set_image属性
                        if hasattr(self.predictor, 'set_image'):
                            logger.info(f"set_image属性类型: {type(self.predictor.set_image)}")
                            # 确保set_image是可调用的
                            if callable(self.predictor.set_image):
                                logger.info("调用predictor.set_image")
                                self.predictor.set_image(image)
                                logger.info("predictor.set_image调用成功")
                            else:
                                logger.error(f"set_image不是可调用对象: {self.predictor.set_image}")
                                print(f"set_image不是可调用对象: {self.predictor.set_image}")
                        else:
                            logger.error("predictor对象没有set_image属性")
                            print("predictor对象没有set_image属性")
                    except Exception as e:
                        logger.error(f"调用predictor.set_image时出错: {type(e).__name__} - {str(e)}", exc_info=True)
                        print(f"调用predictor.set_image时出错: {type(e).__name__} - {str(e)}")
                    self.update_status(f"已加载图像: {os.path.basename(file_path)}，并设置到预测器")
            else:
                logger.error("加载图像失败")
                self.update_status("加载图像失败")
        else:
            logger.info("用户取消了图像选择")
    
    def load_target_image(self):
        """加载目标图像（用于缺陷移植）"""
        logger.info("开始加载目标图像")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开目标图像文件", self.last_target_image_dir, 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            logger.info(f"选择的目标图像文件: {file_path}")
            self.target_image_path = file_path
            # 更新上次目标图像文件夹位置
            self.last_target_image_dir = os.path.dirname(file_path)
            
            # 在目标图像查看器中显示目标图像
            try:
                success = self.target_image_viewer.set_image(file_path)
                if success:
                    self.update_status(f"已加载并显示目标图像: {os.path.basename(file_path)}")
                    logger.info(f"目标图像显示成功")
                else:
                    self.update_status(f"无法加载目标图像: {os.path.basename(file_path)}")
                    logger.error(f"图像加载失败: {file_path}")
            except Exception as e:
                error_msg = f"加载目标图像失败: {str(e)}"
                self.update_status(error_msg)
                logger.error(error_msg)
            
            # 加载缺陷移植工具
            if self.image_path and self.target_image_path:
                success = self.defect_transfer.load_images(self.image_path, self.target_image_path)
                if success:
                    self.update_status(f"缺陷移植工具加载图像成功")
                    logger.info(f"缺陷移植工具加载图像成功")
                    
                    # 如果已经有分割的缺陷掩码，设置到移植工具
                    if len(self.masks) > 0:
                        mask = self.masks[self.current_mask_idx]
                        self.defect_transfer.set_defect_mask(mask)
                        self.update_status(f"已设置缺陷掩码到移植工具")
                else:
                    self.update_status("缺陷移植工具加载图像失败")
                    logger.error("缺陷移植工具加载图像失败")
            else:
                logger.info(f"目标图像已加载，等待源图像和分割结果")
        else:
            logger.info("用户取消了目标图像选择")
        
    def load_model_dialog(self):
        """打开对话框选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开模型文件", self.last_model_dir, 
            "模型文件 (*.pth *.pt)"
        )
        
        if file_path:
            self.trained_model_path = file_path
            # 更新上次模型文件夹位置
            self.last_model_dir = os.path.dirname(file_path)
            self.load_model(self.current_model_size, file_path)
        
    def load_model(self, model_size="base_plus", trained_model_path=None):
        """加载SAM2模型"""
        try:
            self.update_status(f"正在加载模型...")
            
            # 模型配置 - 支持sam2和sam2.1版本
            model_configs = {
                "tiny": ("sam2.1_hiera_tiny.pt", "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"),
                "small": ("sam2.1_hiera_small.pt", "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"),
                "base_plus": ("sam2.1_hiera_base_plus.pt", "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"),
                "large": ("sam2.1_hiera_large.pt", "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),
                "base_plus_sam2": ("sam2_hiera_base_plus.pt", "sam2/configs/sam2/sam2_hiera_b+.yaml")  # sam2版本
            }
            
            if model_size not in model_configs:
                self.update_status(f"模型大小 {model_size} 不可用，使用 base_plus")
                model_size = "base_plus"
                self.cb_model_size.setCurrentText("base_plus")
            
            self.current_model_size = model_size
            
            # 获取模型文件路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_name, config_name = model_configs[model_size]
            checkpoint_path = os.path.join(base_dir, "checkpoints", checkpoint_name)
            config_path = os.path.join(base_dir, config_name)
            
            # 优先尝试加载训练好的模型
            if trained_model_path and os.path.exists(trained_model_path):
                self.update_status(f"尝试直接加载训练好的模型: {os.path.basename(trained_model_path)}")
                
                # 直接使用build_sam2函数加载模型，它会处理权重格式和严格匹配
                try:
                    # 加载用户训练的模型权重
                    user_checkpoint = torch.load(trained_model_path, map_location="cpu", weights_only=False)
                    
                    # 提取模型权重
                    if 'model' in user_checkpoint:
                        model_weights = user_checkpoint['model']
                    elif 'model_state_dict' in user_checkpoint:
                        model_weights = user_checkpoint['model_state_dict']
                    else:
                        model_weights = user_checkpoint
                    
                    # 创建SAM2期望的权重格式
                    sam2_checkpoint = {'model': model_weights}
                    
                    # 保存临时权重文件
                    temp_ckpt_path = os.path.join(os.path.dirname(trained_model_path), "temp_sam2_checkpoint.pt")
                    torch.save(sam2_checkpoint, temp_ckpt_path)
                    
                    # 使用build_sam2函数直接加载模型
                    self.model = build_sam2(config_path, temp_ckpt_path, device=self.device)
                    
                    # 删除临时文件
                    os.remove(temp_ckpt_path)
                    
                    self.update_status(f"成功加载训练模型: {os.path.basename(trained_model_path)}")
                except Exception as e:
                    self.update_status(f"直接加载失败，尝试传统方式加载: {str(e)[:50]}...")
                    # 先加载预训练模型
                    self.model = build_sam2(config_path, checkpoint_path, device=self.device)
                    
                    # 加载训练好的权重
                    checkpoint = torch.load(trained_model_path, map_location=self.device, weights_only=False)
                    
                    # 处理不同格式的模型权重
                    model_weights = None
                    if 'model' in checkpoint:
                        model_weights = checkpoint['model']
                    elif 'model_state_dict' in checkpoint:
                        model_weights = checkpoint['model_state_dict']
                    else:
                        model_weights = checkpoint
                    
                    # 图像训练模型已知的缺失键（这些键在图像训练中被忽略）
                    known_missing_keys = [
                        'no_obj_ptr', 'no_obj_embed_spatial', 'mask_downsample.weight', 'mask_downsample.bias',
                        'memory_attention.layers.0.self_attn.q_proj.weight', 'memory_attention.layers.0.self_attn.q_proj.bias',
                        'memory_attention.layers.0.self_attn.k_proj.weight', 'memory_attention.layers.0.self_attn.k_proj.bias',
                        'memory_attention.layers.0.self_attn.v_proj.weight', 'memory_attention.layers.0.self_attn.v_proj.bias',
                        'memory_attention.layers.0.self_attn.out_proj.weight', 'memory_attention.layers.0.self_attn.out_proj.bias',
                        'memory_attention.layers.0.cross_attn_image.q_proj.weight', 'memory_attention.layers.0.cross_attn_image.q_proj.bias',
                        'memory_attention.layers.0.cross_attn_image.k_proj.weight', 'memory_attention.layers.0.cross_attn_image.k_proj.bias',
                        'memory_attention.layers.0.cross_attn_image.v_proj.weight', 'memory_attention.layers.0.cross_attn_image.v_proj.bias',
                        'memory_attention.layers.0.cross_attn_image.out_proj.weight', 'memory_attention.layers.0.cross_attn_image.out_proj.bias',
                        'memory_attention.layers.0.linear1.weight', 'memory_attention.layers.0.linear1.bias',
                        'memory_attention.layers.0.linear2.weight', 'memory_attention.layers.0.linear2.bias',
                        'memory_attention.layers.0.norm1.weight', 'memory_attention.layers.0.norm1.bias',
                        'memory_attention.layers.0.norm2.weight', 'memory_attention.layers.0.norm2.bias',
                        'memory_attention.layers.0.norm3.weight', 'memory_attention.layers.0.norm3.bias',
                        'memory_attention.layers.1.self_attn.q_proj.weight', 'memory_attention.layers.1.self_attn.q_proj.bias',
                        'memory_attention.layers.1.self_attn.k_proj.weight', 'memory_attention.layers.1.self_attn.k_proj.bias',
                        'memory_attention.layers.1.self_attn.v_proj.weight', 'memory_attention.layers.1.self_attn.v_proj.bias',
                        'memory_attention.layers.1.self_attn.out_proj.weight', 'memory_attention.layers.1.self_attn.out_proj.bias',
                        'memory_attention.layers.1.cross_attn_image.q_proj.weight', 'memory_attention.layers.1.cross_attn_image.q_proj.bias',
                        'memory_attention.layers.1.cross_attn_image.k_proj.weight', 'memory_attention.layers.1.cross_attn_image.k_proj.bias',
                        'memory_attention.layers.1.cross_attn_image.v_proj.weight', 'memory_attention.layers.1.cross_attn_image.v_proj.bias',
                        'memory_attention.layers.1.cross_attn_image.out_proj.weight', 'memory_attention.layers.1.cross_attn_image.out_proj.bias',
                        'memory_attention.layers.1.linear1.weight', 'memory_attention.layers.1.linear1.bias',
                        'memory_attention.layers.1.linear2.weight', 'memory_attention.layers.1.linear2.bias',
                        'memory_attention.layers.1.norm1.weight', 'memory_attention.layers.1.norm1.bias',
                        'memory_attention.layers.1.norm2.weight', 'memory_attention.layers.1.norm2.bias',
                        'memory_attention.layers.1.norm3.weight', 'memory_attention.layers.1.norm3.bias',
                        'memory_attention.layers.2.self_attn.q_proj.weight', 'memory_attention.layers.2.self_attn.q_proj.bias',
                        'memory_attention.layers.2.self_attn.k_proj.weight', 'memory_attention.layers.2.self_attn.k_proj.bias',
                        'memory_attention.layers.2.self_attn.v_proj.weight', 'memory_attention.layers.2.self_attn.v_proj.bias',
                        'memory_attention.layers.2.self_attn.out_proj.weight', 'memory_attention.layers.2.self_attn.out_proj.bias',
                        'memory_attention.layers.2.cross_attn_image.q_proj.weight', 'memory_attention.layers.2.cross_attn_image.q_proj.bias',
                        'memory_attention.layers.2.cross_attn_image.k_proj.weight', 'memory_attention.layers.2.cross_attn_image.k_proj.bias',
                        'memory_attention.layers.2.cross_attn_image.v_proj.weight', 'memory_attention.layers.2.cross_attn_image.v_proj.bias',
                        'memory_attention.layers.2.cross_attn_image.out_proj.weight', 'memory_attention.layers.2.cross_attn_image.out_proj.bias',
                        'memory_attention.layers.2.linear1.weight', 'memory_attention.layers.2.linear1.bias',
                        'memory_attention.layers.2.linear2.weight', 'memory_attention.layers.2.linear2.bias',
                        'memory_attention.layers.2.norm1.weight', 'memory_attention.layers.2.norm1.bias',
                        'memory_attention.layers.2.norm2.weight', 'memory_attention.layers.2.norm2.bias',
                        'memory_attention.layers.2.norm3.weight', 'memory_attention.layers.2.norm3.bias',
                        'memory_attention.layers.3.self_attn.q_proj.weight', 'memory_attention.layers.3.self_attn.q_proj.bias',
                        'memory_attention.layers.3.self_attn.k_proj.weight', 'memory_attention.layers.3.self_attn.k_proj.bias',
                        'memory_attention.layers.3.self_attn.v_proj.weight', 'memory_attention.layers.3.self_attn.v_proj.bias',
                        'memory_attention.layers.3.self_attn.out_proj.weight', 'memory_attention.layers.3.self_attn.out_proj.bias',
                        'memory_attention.layers.3.cross_attn_image.q_proj.weight', 'memory_attention.layers.3.cross_attn_image.q_proj.bias',
                        'memory_attention.layers.3.cross_attn_image.k_proj.weight', 'memory_attention.layers.3.cross_attn_image.k_proj.bias',
                        'memory_attention.layers.3.cross_attn_image.v_proj.weight', 'memory_attention.layers.3.cross_attn_image.v_proj.bias',
                        'memory_attention.layers.3.cross_attn_image.out_proj.weight', 'memory_attention.layers.3.cross_attn_image.out_proj.bias',
                        'memory_attention.layers.3.linear1.weight', 'memory_attention.layers.3.linear1.bias',
                        'memory_attention.layers.3.linear2.weight', 'memory_attention.layers.3.linear2.bias',
                        'memory_attention.layers.3.norm1.weight', 'memory_attention.layers.3.norm1.bias',
                        'memory_attention.layers.3.norm2.weight', 'memory_attention.layers.3.norm2.bias',
                        'memory_attention.layers.3.norm3.weight', 'memory_attention.layers.3.norm3.bias',
                        'memory_attention.norm.weight', 'memory_attention.norm.bias',
                        'sam_mask_decoder.obj_score_token.weight',
                        'sam_mask_decoder.pred_obj_score_head.layers.0.weight', 'sam_mask_decoder.pred_obj_score_head.layers.0.bias',
                        'sam_mask_decoder.pred_obj_score_head.layers.1.weight', 'sam_mask_decoder.pred_obj_score_head.layers.1.bias',
                        'sam_mask_decoder.pred_obj_score_head.layers.2.weight', 'sam_mask_decoder.pred_obj_score_head.layers.2.bias',
                        'obj_ptr_proj.layers.0.weight', 'obj_ptr_proj.layers.0.bias',
                        'obj_ptr_proj.layers.1.weight', 'obj_ptr_proj.layers.1.bias',
                        'obj_ptr_proj.layers.2.weight', 'obj_ptr_proj.layers.2.bias',
                        'obj_ptr_tpos_proj.weight', 'obj_ptr_tpos_proj.bias'
                    ]
                    
                    # 加载模型权重，允许忽略不匹配的权重，但记录匹配情况
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_weights, strict=False)
                    
                    # 过滤掉已知的缺失键
                    filtered_missing_keys = [key for key in missing_keys if key not in known_missing_keys]
                    
                    # 记录匹配情况
                    if not filtered_missing_keys and not unexpected_keys:
                        self.update_status(f"成功加载训练模型，所有权重匹配: {os.path.basename(trained_model_path)}")
                        self.lbl_model_info.setText(f"已加载: {os.path.basename(trained_model_path)} (所有权重匹配)")
                    else:
                        # 记录匹配情况
                        match_status = f"成功加载训练模型: {os.path.basename(trained_model_path)}"
                        if filtered_missing_keys:
                            match_status += f"\n缺失键数量: {len(filtered_missing_keys)}"
                            logger.warning(f"模型加载时缺失键: {filtered_missing_keys}")
                        if unexpected_keys:
                            match_status += f"\n意外键数量: {len(unexpected_keys)}"
                            logger.warning(f"模型加载时意外键: {unexpected_keys}")
                        self.update_status(match_status)
                        self.lbl_model_info.setText(f"已加载: {os.path.basename(trained_model_path)}")
                
                # 获取训练元数据
                best_iou = None
                if 'checkpoint' in locals() or 'checkpoint' in globals():
                    best_iou = checkpoint.get('best_meter_values', {}).get('iou', None)
                    if best_iou is None:
                        best_iou = checkpoint.get('iou', None)
                elif 'user_checkpoint' in locals():
                    best_iou = user_checkpoint.get('best_meter_values', {}).get('iou', None)
                    if best_iou is None:
                        best_iou = user_checkpoint.get('iou', None)
                
                if best_iou is not None:
                    self.update_status(f"成功加载训练模型，训练时的最佳IoU: {best_iou:.4f}")
                    self.lbl_model_info.setText(f"已加载: {os.path.basename(trained_model_path)} (IoU: {best_iou:.4f})")
                else:
                    self.update_status(f"成功加载训练模型")
                    self.lbl_model_info.setText(f"已加载: {os.path.basename(trained_model_path)}")
            else:
                self.update_status(f"加载预训练模型: {checkpoint_name}")
                self.model = build_sam2(config_path, checkpoint_path, device=self.device)
                self.lbl_model_info.setText(f"已加载预训练模型: {model_size}")
                
                if trained_model_path:
                    self.update_status(f"警告: 未找到训练模型 {trained_model_path}，使用预训练模型")
            
            # 创建预测器
            logger.info(f"创建预测器，模型类型: {type(self.model)}")
            try:
                self.predictor = SAM2ImagePredictor(self.model)
                logger.info(f"预测器创建成功，类型: {type(self.predictor)}")
                
                # 检查set_image方法
                if not hasattr(self.predictor, 'set_image'):
                    logger.error("预测器对象没有set_image属性")
                    print("预测器对象没有set_image属性")
                    self.predictor = None
                    self.update_status("创建预测器失败: 预测器对象没有set_image属性")
                else:
                    logger.info(f"预测器的set_image方法类型: {type(self.predictor.set_image)}")
                    
                    if not callable(self.predictor.set_image):
                        logger.error(f"set_image不是可调用对象: {type(self.predictor.set_image)}")
                        print(f"set_image不是可调用对象: {type(self.predictor.set_image)}")
                        self.predictor = None
                        self.update_status("创建预测器失败: set_image不是可调用对象")
                    else:
                        # 如果已经加载了图像，设置图像到预测器
                        if self.image_path:
                            try:
                                image = np.array(Image.open(self.image_path).convert("RGB"))
                                logger.info(f"调用predictor.set_image，图像形状: {image.shape}")
                                self.predictor.set_image(image)
                                logger.info("predictor.set_image调用成功")
                                self.update_status(f"模型加载完成，并设置图像到预测器")
                            except Exception as e:
                                error_msg = f"设置图像到预测器失败: {type(e).__name__} - {str(e)}"
                                self.update_status(error_msg)
                                logger.error(error_msg, exc_info=True)
                                print(error_msg)
                        else:
                            self.update_status("模型加载完成")
            except Exception as e:
                error_msg = f"创建预测器失败: {type(e).__name__} - {str(e)}"
                self.update_status(error_msg)
                logger.error(error_msg, exc_info=True)
                print(error_msg)
                self.predictor = None
            
        except Exception as e:
                import traceback
                error_msg = f"加载模型出错: {str(e)}"
                self.update_status(error_msg)
                logger.error(f"加载模型出错: {str(e)}", exc_info=True)
                print(f"加载模型出错: {str(e)}")
                traceback.print_exc()
        
    def on_model_size_changed(self, new_size):
        """当模型大小选择改变时调用"""
        self.current_model_size = new_size
        self.load_model(new_size, self.trained_model_path)
        
    def set_mode(self, mode):
        """设置操作模式"""
        # 取消所有按钮的选中状态
        for btn in self.mode_buttons:
            btn.setChecked(False)
        
        # 选中当前模式按钮
        if 0 <= mode < len(self.mode_buttons):
            self.mode_buttons[mode].setChecked(True)
            self.image_viewer.set_mode(mode)
            
            # 更新状态信息
            mode_names = ["平移", "点选", "框选", "轮廓"]
            self.update_status(f"当前模式: {mode_names[mode]}")
        
    # 移除缺陷类型选择方法
    # def select_defect_type(self, idx):
    #     """选择缺陷类型"""
    #     ...
            if len(self.masks) > 0:
                mask = self.masks[self.current_mask_idx]
                score = self.scores[self.current_mask_idx] if len(self.scores) > 0 else 0
                label = f"分割区域 ({score:.2f})"
                self.image_viewer.set_mask(mask, self.mask_color, label)
        
    def on_mouse_clicked(self, x, y, label):
        """处理鼠标点击事件"""
        self.image_viewer.add_point(x, y, label)
        point_type = "包含点" if label == 1 else "排除点"
        self.update_status(f"添加{point_type}: ({x}, {y})")
        
    def on_mouse_dragged(self, x1, y1, x2, y2):
        """处理鼠标拖动事件（框选）"""
        self.image_viewer.add_box(x1, y1, x2, y2)
        self.update_status(f"添加框选: ({x1}, {y1}) 到 ({x2}, {y2})")
        
        # 自动执行分割
        self.segment_image()
        
    def on_contour_updated(self, points, is_closed=False):
        """处理轮廓更新事件"""
        status = f"轮廓点数量: {len(points)}"
        if is_closed:
            status += " (已闭合)"
        status += "\n提示: 轮廓已转换为框提示，可切换到点选模式添加点提示"
        self.update_status(status)
        
    def clear_last_point(self):
        """清除最后一个点标记"""
        self.image_viewer.clear_last_point()
        self.update_status("已清除最后一个点标记")
        
    def reset_all(self):
        """重置所有设置"""
        self.image_viewer.clear_all()
        self.masks = []
        self.scores = []
        self.current_mask_idx = 0
        self.update_status("已重置所有设置")
        
    def segment_image(self):
        """执行分割"""
        if not self.image_path:
            self.update_status("请先加载图像")
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
        
        if self.predictor is None:
            self.update_status("请先加载模型")
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        try:
            # 获取点和框
            points = self.image_viewer.get_points()
            boxes = self.image_viewer.get_boxes()
            contour = self.image_viewer.get_contour()
            
            if not points and not boxes and len(contour) <= 2:
                self.update_status("请先添加点、框选区域或绘制轮廓")
                QMessageBox.warning(self, "警告", "请先添加点、框选区域或绘制轮廓")
                return
            
            # 准备输入参数
            point_coords = None
            point_labels = None
            box = None
            
            if points:
                coords = np.array([(x, y) for x, y, _ in points])
                labels = np.array([label for _, _, label in points])
                point_coords = coords
                point_labels = labels
            
            if boxes:
                # 使用最后一个框
                box = np.array(boxes[-1])
            
            # 处理轮廓提示
            if len(contour) > 2:
                # 将轮廓转换为边界框
                contour_coords = np.array(contour)
                x_min = np.min(contour_coords[:, 0])
                y_min = np.min(contour_coords[:, 1])
                x_max = np.max(contour_coords[:, 0])
                y_max = np.max(contour_coords[:, 1])
                
                # 设置为框提示 (覆盖现有框提示)
                box = np.array([x_min, y_min, x_max, y_max])

            # 准备状态信息
            status_info = []
            if point_coords is not None:
                status_info.append(f"点提示: {len(point_coords)}个点")
            if box is not None:
                status_info.append(f"框提示: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
            
            if status_info:
                self.update_status(f"正在执行分割... 使用: {', '.join(status_info)}")
            else:
                self.update_status("正在执行分割...")
            self.update_progress(20, "正在执行分割...")
            
            # 确保图像已设置
            if not hasattr(self.predictor, '_is_image_set') or not self.predictor._is_image_set:
                if self.image_path:
                    image = np.array(Image.open(self.image_path).convert("RGB"))
                    self.predictor.set_image(image)
                    self.update_status(f"重新设置图像到预测器")
                else:
                    self.update_status("错误: 没有图像可设置")
                    QMessageBox.critical(self, "分割错误", "没有图像可设置，请先加载图像")
                    self.update_progress(0, "错误: 没有图像可设置")
                    return

            # 记录预测参数以调试
            self.update_status(f"预测参数: 点提示={point_coords is not None}, 框提示={box is not None}")
            
            # 执行预测
            start_time = time.time()
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
            
            self.update_progress(80, "分割完成，正在处理结果...")
            
            if len(masks) > 0:
                self.masks = masks
                self.scores = scores
                self.current_mask_idx = np.argmax(scores)
                
                # 显示最佳掩码
                label = f"分割区域 ({scores[self.current_mask_idx]:.2f})"
                self.image_viewer.set_mask(
                    masks[self.current_mask_idx], 
                    self.mask_color,
                    label
                )
                
                end_time = time.time()
                self.update_status(f"分割完成，最佳得分: {scores[self.current_mask_idx]:.3f}, 耗时: {end_time - start_time:.2f}秒")
            
            # 启用保存缺陷按钮
            self.btn_save_defect.setEnabled(True)
            
            # 如果已经加载了目标图像，更新状态
            if self.target_image_path:
                self.update_status("分割完成，可以执行缺陷移植")
                self.update_progress(100, "分割完成，可以执行缺陷移植")
            else:
                self.update_status("未生成有效的分割结果")
                self.update_progress(100, "分割完成")
            
        except Exception as e:
            error_msg = f"分割出错: {str(e)}"
            self.update_status(error_msg)
            self.update_progress(0, f"错误: {str(e)[:30]}...")
            QMessageBox.critical(self, "分割错误", error_msg)
            
    def toggle_mask(self):
        """切换显示不同的掩码"""
        if len(self.masks) == 0:
            self.update_status("没有可用的掩码")
            QMessageBox.information(self, "提示", "没有可用的掩码")
            return
        
        # 切换到下一个掩码
        self.current_mask_idx = (self.current_mask_idx + 1) % len(self.masks)
        
        # 显示当前掩码
        label = f"分割区域 ({self.scores[self.current_mask_idx]:.2f})"
        self.image_viewer.set_mask(
            self.masks[self.current_mask_idx], 
            self.mask_color,
            label
        )
        
        self.update_status(f"显示掩码 {self.current_mask_idx + 1}/{len(self.masks)}，得分: {self.scores[self.current_mask_idx]:.3f}")
    
    def update_progress(self, value, text=None):
        """更新进度条状态
        
        Args:
            value: 进度值 (0-100)
            text: 可选的状态文本，如果为None则保持当前格式
        """
        self.progress_bar.setValue(value)
        if text is not None:
            self.progress_bar.setFormat(text)
    
    def _on_debounce_timeout(self):
        """防抖超时处理，执行实际的移植操作"""
        if self.pending_transfer:
            self.pending_transfer = False
            self.transfer_defect()
    
    def on_position_slider_changed(self):
        """位置滑块值改变事件（带防抖）"""
        x_value = self.slider_x.value() / 10.0  # 转换为带小数点的位置值
        y_value = self.slider_y.value() / 10.0
        self.lbl_x.setText(f"{x_value:.1f}")
        self.lbl_y.setText(f"{y_value:.1f}")
        
        # 防抖处理：只有在有缺陷图像和目标图像时才处理        
        # 启动防抖计时器（50ms延迟）
        self.pending_transfer = True
        self.debounce_timer.start(50)
    
    def on_transform_slider_changed(self):
        """变换滑块值改变事件（带防抖）"""
        scale_value = self.slider_scale.value() / 100.0  # 转换为0.01-5.0范围
        rotate_value = self.slider_rotate.value() / 10.0  # 转换为带小数点的角度值
        self.lbl_scale.setText(f"{scale_value:.1f}")
        self.lbl_rotate.setText(f"{rotate_value:.1f}°")
        
        # 防抖处理：只有在有缺陷图像和目标图像时才处理
        if self.defect_image_path and self.target_image_path: 
            # 启动防抖计时器（50ms延迟）
            self.pending_transfer = True
            self.debounce_timer.start(50)
    
    def select_defect_image(self):
        """选择缺陷图像文件"""
        try:
            # 打开文件对话框选择缺陷图像
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择缺陷图像", self.last_defect_dir, "PNG Images (*.png);;All Files (*)"
            )
            
            if file_path:
                # 更新上次缺陷文件夹位置
                self.last_defect_dir = os.path.dirname(file_path)
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    self.update_status("选择的文件不存在")
                    QMessageBox.warning(self, "警告", "选择的文件不存在")
                    return
                    
                # 检查是否为PNG图像
                if not file_path.lower().endswith('.png'):
                    self.update_status("请选择PNG格式的图像文件")
                    QMessageBox.warning(self, "警告", "请选择PNG格式的图像文件")
                    return
                    
                # 更新缺陷图像路径
                self.defect_image_path = file_path
                
                self.update_status(f"已选择缺陷图像: {os.path.basename(file_path)}")
                
                # 启用缺陷融合按钮
                self.btn_final_fusion.setEnabled(True)
                
                # 自动进行预览
                if self.target_image_path:
                    self.transfer_defect()
                else:
                    QMessageBox.information(self, "提示", "缺陷图像已选择，请加载目标图像后进行预览")
                
                # 启用所有滑块控件，允许实时调整
                self.slider_x.setEnabled(True)
                self.slider_y.setEnabled(True)
                self.slider_scale.setEnabled(True)
                self.slider_rotate.setEnabled(True)
                self.slider_alpha.setEnabled(True)
            
        except Exception as e:
            error_msg = f"选择缺陷图像出错: {str(e)}"
            self.update_status(error_msg)
            QMessageBox.critical(self, "选择错误", error_msg)
    
    def save_defect_image(self):
        """保存透明背景的缺陷图像，将缺陷部分置于坐标中心"""
        if len(self.masks) == 0:
            self.update_status("没有可保存的分割结果")
            QMessageBox.information(self, "提示", "没有可保存的分割结果")
            return
        
        # 创建保存目录
        save_dir = "defects"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_name = os.path.basename(self.image_path)
        base_name = os.path.splitext(image_name)[0]
        
        # 获取保存文件路径
        defect_path = os.path.join(save_dir, f"{base_name}_defect_{timestamp}.png")
        
        try:
            # 加载原始图像（保持原始BGR颜色）
            original_image = cv2.imread(self.image_path)
            if original_image is None:
                raise ValueError("无法加载原始图像")
            
            # 获取当前掩码
            mask = self.masks[self.current_mask_idx]
            
            # 找到掩码的边界框
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            # 检查是否有有效区域
            if not np.any(rows) or not np.any(cols):
                raise ValueError("掩码中没有有效区域")
            
            # 计算边界坐标
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # 计算缺陷区域的尺寸
            defect_height = y_max - y_min + 1
            defect_width = x_max - x_min + 1
            
            # 确定新图像的尺寸（保证足够大以容纳缺陷并居中）
            # 为了美观，可以将图像尺寸设置为缺陷尺寸的1.2倍，并向上取整到最近的10的倍数
            new_height = max(100, int(defect_height * 1.2))
            new_width = max(100, int(defect_width * 1.2))
            
            # 向上取整到最近的10的倍数
            new_height = (new_height + 9) // 10 * 10
            new_width = (new_width + 9) // 10 * 10
            
            # 创建新的透明背景图像
            centered_rgba = np.zeros((new_height, new_width, 4), dtype=np.uint8)
            
            # 计算缺陷在新图像中的居中位置
            start_y = (new_height - defect_height) // 2
            start_x = (new_width - defect_width) // 2
            
            # 复制缺陷区域到新图像的中心位置
            centered_rgba[start_y:start_y+defect_height, start_x:start_x+defect_width, :3] = \
                original_image[y_min:y_max+1, x_min:x_max+1]
            
            # 设置透明度
            centered_rgba[start_y:start_y+defect_height, start_x:start_x+defect_width, 3] = \
                mask[y_min:y_max+1, x_min:x_max+1] * 255
            
            # 保存透明PNG图像
            cv2.imwrite(defect_path, centered_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            self.defect_image_path = defect_path
            self.update_status(f"透明缺陷图像已保存到: {defect_path}，缺陷已居中")
            QMessageBox.information(self, "保存成功", f"透明缺陷图像已保存\n{defect_path}\n缺陷部分已居中放置")
            
            # 启用缺陷融合按钮
            self.btn_final_fusion.setEnabled(True)
            
        except Exception as e:
            error_msg = f"保存缺陷图像出错: {str(e)}"
            self.update_status(error_msg)
            QMessageBox.critical(self, "保存错误", error_msg)
            

            
    def update_status(self, message):
        """更新状态栏信息"""
        self.statusBar().showMessage(message)
    
    def _prepare_transfer_conditions(self):
        """准备缺陷移植的条件检查，返回是否满足条件"""
        if not self.image_path or not self.target_image_path:
            self.update_status("请先加载源图像和目标图像")
            QMessageBox.warning(self, "警告", "请先加载源图像和目标图像")
            return False
            
        if len(self.masks) == 0:
            self.update_status("请先分割出缺陷区域")
            QMessageBox.warning(self, "警告", "请先分割出缺陷区域")
            return False
            
        return True
    
    def _get_transfer_result(self, blend_strength):
        """获取缺陷移植结果，支持预览和最终融合模式"""
        try:
            print(f"[DEBUG] 开始获取移植结果，融合强度: {blend_strength}")
            
            # 设置缺陷掩码到移植工具
            mask = self.masks[self.current_mask_idx]
            print(f"[DEBUG] 设置缺陷掩码，掩码形状: {mask.shape}")
            self.defect_transfer.set_defect_mask(mask)
            
            # 设置位置、缩放和旋转参数（使用0.1精度）
            x_offset = self.slider_x.value() / 10.0  # 转换为带小数点的位置值
            y_offset = self.slider_y.value() / 10.0  # 转换为带小数点的位置值
            scale = self.slider_scale.value() / 100.0  # 转换为0.01-5.0范围
            rotation = self.slider_rotate.value() / 10.0  # 转换为带小数点的角度值
            
            print(f"[DEBUG] 设置参数 - X偏移: {x_offset}, Y偏移: {y_offset}, 缩放: {scale}, 旋转: {rotation}")
            
            self.defect_transfer.set_position(x_offset, y_offset)
            self.defect_transfer.set_scale(scale)
            self.defect_transfer.set_rotation(rotation)

            # 执行缺陷融合，支持回退机制
            try:
                print("[DEBUG] 尝试使用blend_defect_with_rotation方法")
                # 尝试使用新的融合方法
                result = self.defect_transfer.blend_defect_with_rotation(blend_strength=blend_strength)
                
                if result is None:
                    print("[WARNING] 带旋转融合失败，回退到仅缩放融合")
                    result = self.defect_transfer.blend_defect(blend_strength=blend_strength)
                else:
                    print("[DEBUG] blend_defect_with_rotation执行成功")
                    
                if result is not None:
                    print(f"[DEBUG] 移植结果获取成功，图像形状: {result.shape}")
                else:
                    print("[ERROR] 移植结果为None，无法显示")
                return result

            except Exception as e:
                print(f"[DEBUG] 获取移植结果出错: {e}")
                import traceback
                traceback.print_exc()
            return None
            
            
        except Exception as e:
            error_msg = f"获取移植结果出错: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.update_status(error_msg)
            return None
    def _create_preview_with_transparent_defect(self):
        """使用透明缺陷图像创建预览"""
        if not self.defect_image_path or not self.target_image_path:
            return None
            
        try:
            # 加载目标图像
            target_image = cv2.imread(self.target_image_path)
            if target_image is None:
                return None
                
            # 加载透明缺陷图像
            defect_image = cv2.imread(self.defect_image_path, cv2.IMREAD_UNCHANGED)
            if defect_image is None:
                return None
                
            # 获取变换参数
            x_offset = self.slider_x.value()
            y_offset = self.slider_y.value()
            scale = self.slider_scale.value() / 100.0
            rotation = self.slider_rotate.value()
            alpha = self.slider_alpha.value() / 100.0
            
            # 创建副本用于预览
            preview_image = target_image.copy()
            
            # 获取缺陷图像和目标图像的尺寸
            defect_h, defect_w = defect_image.shape[:2]
            target_h, target_w = target_image.shape[:2]
            
            # 计算基础缩放比例，确保缺陷图像适应目标图像尺寸
            base_scale = min(target_w / defect_w, target_h / defect_h)
            
            # 结合用户设置的缩放参数
            final_scale = base_scale * scale
            
            # 计算最终缩放后的尺寸
            new_w = int(defect_w * final_scale)
            new_h = int(defect_h * final_scale)
            
            # 缩放缺陷图像
            if scale != 1.0:
                defect_resized = cv2.resize(defect_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                defect_resized = defect_image
            
            # 旋转缺陷图像
            if rotation != 0:
                center = (new_w // 2, new_h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                defect_rotated = cv2.warpAffine(defect_resized, rotation_matrix, (new_w, new_h))
            else:
                defect_rotated = defect_resized
            
            # 计算放置位置
            target_h, target_w = target_image.shape[:2]
            x_pos = (target_w - new_w) // 2 + x_offset
            y_pos = (target_h - new_h) // 2 + y_offset
            
            # 确保位置在图像范围内
            x_start = max(0, x_pos)
            y_start = max(0, y_pos)
            x_end = min(target_w, x_pos + new_w)
            y_end = min(target_h, y_pos + new_h)
            
            # 计算缺陷图像的有效区域
            defect_x_start = max(0, -x_pos)
            defect_y_start = max(0, -y_pos)
            defect_x_end = min(new_w, target_w - x_pos)
            defect_y_end = min(new_h, target_h - y_pos)
            
            # 提取缺陷图像的RGB和Alpha通道
            if defect_rotated.shape[2] == 4:  # RGBA
                defect_rgb = defect_rotated[:, :, :3]
                defect_alpha = defect_rotated[:, :, 3] / 255.0 * alpha
            else:  # RGB
                defect_rgb = defect_rotated
                defect_alpha = np.ones((new_h, new_w)) * alpha
            
            # 将缺陷图像合成到目标图像上（优化版本）
            # 提取目标图像的有效区域
            target_region = preview_image[y_start:y_end, x_start:x_end]
            
            # 提取缺陷图像的有效区域
            defect_region_rgb = defect_rgb[defect_y_start:defect_y_end, defect_x_start:defect_x_end]
            defect_region_alpha = defect_alpha[defect_y_start:defect_y_end, defect_x_start:defect_x_end]
            
            # 确保 alpha 通道维度正确，形状匹配
            if defect_region_alpha.ndim == 2:
                defect_region_alpha = defect_region_alpha[:, :, np.newaxis]
            
            # 验证形状是否匹配，确保可以进行广播操作
            if target_region.shape[:2] != defect_region_alpha.shape[:2]:
                print(f"[WARNING] 形状不匹配: 目标区域{target_region.shape}, 缺陷alpha{defect_region_alpha.shape}")
                # 调整缺陷区域大小以匹配目标区域
                target_h, target_w = target_region.shape[:2]
                defect_region_alpha = cv2.resize(defect_region_alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                defect_region_rgb = cv2.resize(defect_region_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                # 重新添加通道维度
                if defect_region_alpha.ndim == 2:
                    defect_region_alpha = defect_region_alpha[:, :, np.newaxis]
            
            # 使用向量化操作进行alpha混合
            try:
                target_region[:] = (
                    target_region * (1 - defect_region_alpha) + 
                    defect_region_rgb * defect_region_alpha
                ).astype(np.uint8)
            except ValueError as e:
                print(f"[ERROR] Alpha混合出错: {e}")
                print(f"目标区域形状: {target_region.shape}, 缺陷RGB形状: {defect_region_rgb.shape}, 缺陷Alpha形状: {defect_region_alpha.shape}")
            
            return preview_image
            
        except Exception as e:
            print(f"预览创建出错: {e}")
            return None
    
    def transfer_defect(self):
        """执行缺陷移植预览（使用透明缺陷图像）"""
        if not self.defect_image_path or not self.target_image_path:
            self.update_status("请先保存缺陷图像并加载目标图像")
            QMessageBox.warning(self, "警告", "请先保存缺陷图像并加载目标图像")
            return
            
        try:
            self.update_status("正在准备缺陷移植预览...")
            self.update_progress(30, "正在准备缺陷移植预览...")
            
            # 创建预览图像
            preview_image = self._create_preview_with_transparent_defect()
            
            self.update_progress(80, "预览创建完成，正在显示...")
            
            if preview_image is not None:
                # 显示预览图像，保留视图状态
                temp_preview_path = "temp_preview_result.png"
                cv2.imwrite(temp_preview_path, preview_image)
                self.target_image_viewer.set_image(temp_preview_path, preserve_view_state=True)
                
                self.update_status("缺陷预览已更新")
                self.transfer_mode = True
                
                # 保存预览图像用于后续融合
                self.preview_composite = preview_image
                
                # 启用缺陷融合按钮
                self.btn_final_fusion.setEnabled(True)
                self.update_progress(100, "缺陷预览完成")
            else:
                self.update_status("缺陷预览失败")
                self.update_progress(0, "缺陷预览失败")
                QMessageBox.critical(self, "错误", "缺陷预览失败")
            
        except Exception as e:
            error_msg = f"缺陷预览出错: {str(e)}"
            self.update_status(error_msg)
            self.update_progress(0, f"预览错误: {str(e)[:30]}...")
            QMessageBox.critical(self, "预览错误", error_msg)
    
    def update_transfer_position(self):
        """更新移植位置参数并更新预览"""
        if self.defect_image_path and self.target_image_path:
            self.transfer_defect()
    
    def update_transfer_alpha(self, value):
        """更新透明度参数并更新预览"""
        self.lbl_alpha.setText(f"{value}%")
        if self.defect_image_path and self.target_image_path:
            self.transfer_defect()
    

    
    def _save_segmentation_result(self):
        """保存分割结果"""
        # 创建保存目录
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_name = os.path.basename(self.image_path)
        base_name = os.path.splitext(image_name)[0]
        
        # 获取保存文件路径
        result_image_path = os.path.join(save_dir, f"{base_name}_result_{timestamp}.png")
        mask_path = os.path.join(save_dir, f"{base_name}_mask_{timestamp}.png")
        
        try:
            # 保存带分割结果的图像
            # 从QImage转换为OpenCV格式
            q_image = self.image_viewer.grab()
            image = q_image.toImage()
            image = image.convertToFormat(QImage.Format.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            cv2.imwrite(result_image_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            
            # 保存掩码
            mask = self.masks[self.current_mask_idx].astype(np.uint8) * 255
            cv2.imwrite(mask_path, mask)
            
            self.update_status(f"分割结果已保存到: {result_image_path} 和 {mask_path}")
            QMessageBox.information(self, "保存成功", f"分割结果已保存\n{result_image_path}\n{mask_path}")
            
        except Exception as e:
            error_msg = f"保存结果出错: {str(e)}"
            self.update_status(error_msg)
            QMessageBox.critical(self, "保存错误", error_msg)
    
    def _execute_final_transfer(self):
        """执行最终的缺陷融合并保存结果"""
        try:
            self.update_status("正在执行最终缺陷融合...")
            self.update_progress(30, "正在执行最终缺陷融合...")
            
            # 获取透明度参数
            alpha = self.slider_alpha.value() / 100.0
            
            # 执行最终的缺陷融合
            try:
                # 尝试使用新的融合方法
                final_result = self.defect_transfer.blend_defect_with_rotation(blend_strength=alpha)
            except AttributeError:
                # 如果新方法不存在，回退到旧方法
                final_result = self.defect_transfer.blend_defect(blend_strength=alpha)
            
            if final_result is not None:
                self.transfer_result = final_result
                
                # 显示最终结果，保留视图状态
                temp_result_path = "temp_final_result.png"
                cv2.imwrite(temp_result_path, final_result)
                self.target_image_viewer.set_image(temp_result_path, preserve_view_state=True)
                
                self.update_status("最终缺陷融合完成")
                self.update_progress(60, "最终缺陷融合完成，正在保存...")
                
                # 保存结果
                self._save_transfer_result()
                
                self.update_progress(100, "最终缺陷融合完成")
            else:
                self.update_status("最终缺陷融合失败")
                self.update_progress(0, "最终缺陷融合失败")
                QMessageBox.critical(self, "错误", "最终缺陷融合失败")
            
        except Exception as e:
            error_msg = f"最终缺陷融合出错: {str(e)}"
            self.update_status(error_msg)
            self.update_progress(0, f"融合错误: {str(e)[:30]}...")
            QMessageBox.critical(self, "融合错误", error_msg)
    
    def final_fusion(self):
        """执行最终的缺陷融合并保存结果"""
        if self.preview_composite is None:
            self.update_status("请先进行预览操作")
            QMessageBox.warning(self, "警告", "请先进行预览操作")
            return
            
        try:
            self.update_status("正在执行最终缺陷融合...")
            self.update_progress(30, "正在执行最终缺陷融合...")
            
            # 创建保存目录
            save_dir = "fusion_results"
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成唯一文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            source_name = os.path.basename(self.image_path)
            target_name = os.path.basename(self.target_image_path)
            base_source = os.path.splitext(source_name)[0]
            base_target = os.path.splitext(target_name)[0]
            
            # 获取保存文件路径
            result_path = os.path.join(save_dir, f"{base_source}_to_{base_target}_{timestamp}.png")
            
            # 保存最终的融合结果
            cv2.imwrite(result_path, self.preview_composite)
            
            self.update_progress(80, "融合结果已保存，正在显示...")
            
            # 显示最终结果
            self.target_image_viewer.set_image(result_path)
            
            self.update_status(f"最终缺陷融合完成，结果已保存到: {result_path}")
            self.update_progress(100, "最终缺陷融合完成")
            QMessageBox.information(self, "融合成功", f"缺陷融合完成\n结果已保存到: {result_path}")
            
        except Exception as e:
            error_msg = f"最终缺陷融合出错: {str(e)}"
            self.update_status(error_msg)
            self.update_progress(0, f"融合错误: {str(e)[:30]}...")
            QMessageBox.critical(self, "融合错误", error_msg)
    
    def _save_transfer_result(self):
        """保存移植结果"""
        if self.transfer_result is None:
            self.update_status("没有可保存的移植结果")
            QMessageBox.information(self, "提示", "没有可保存的移植结果")
            return
        
        # 创建保存目录
        save_dir = "transfer_results"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        source_name = os.path.basename(self.image_path)
        target_name = os.path.basename(self.target_image_path)
        base_source = os.path.splitext(source_name)[0]
        base_target = os.path.splitext(target_name)[0]
        
        # 获取保存文件路径
        result_path = os.path.join(save_dir, f"{base_source}_to_{base_target}_{timestamp}.png")
        
        try:
            # 保存移植结果
            cv2.imwrite(result_path, self.transfer_result)  # 直接保存RGB图像，不进行颜色转换
            
            self.update_status(f"移植结果已保存到: {result_path}")
            QMessageBox.information(self, "保存成功", f"移植结果已保存\n{result_path}")
            
        except Exception as e:
            error_msg = f"保存移植结果出错: {str(e)}"
            self.update_status(error_msg)
            QMessageBox.critical(self, "保存错误", error_msg)

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sam2_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("启动SAM2应用程序")
    # 检查系统并设置高DPI支持
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    try:
        # 创建主窗口
        window = SAM2PyQtApp()
        window.show()
        logger.info("应用程序窗口已显示")
        # 运行应用程序
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"应用程序运行出错: {type(e).__name__} - {str(e)}", exc_info=True)
        print(f"应用程序运行出错: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)