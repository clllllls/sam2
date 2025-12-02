import os
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar,
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


class ImageViewer(QWidget):
    """图像查看器组件，负责显示图像和处理用户交互"""
    
    # 信号定义
    mouse_clicked = pyqtSignal(int, int, int)  # x, y, label
    mouse_dragged = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2
    contour_updated = pyqtSignal(list, bool)  # 轮廓点列表, 是否闭合
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.image = None  # 原始图像
        self.displayed_image = None  # 显示的图像（包含标注等）
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.start_pos = None
        self.dragging = False
        self.selecting_box = False
        self.drawing_contour = False
        self.current_rect = QRect()
        self.contour_points = []
        
        # 设置背景色为浅灰色
        self.setStyleSheet("background-color: #f0f0f0;")
        
        # 允许鼠标跟踪
        self.setMouseTracking(True)
        
        # 当前操作模式 (0: 平移, 1: 点选, 2: 框选, 3: 轮廓)
        self.mode = 1
        
        # 点标记信息
        self.points = []  # 存储点坐标和标签
        self.point_size = 5
        
        # 框标记信息
        self.boxes = []
        
        # 掩码信息
        self.mask = None
        self.mask_color = [255, 0, 0, 0]  # 完全透明的红色 (不再使用填充)
        self.contour_color = [255, 255, 255, 255]  # 白色轮廓
        
        # 缺陷类型标签
        self.defect_label = ""
        
        # 绘制工具类型
        self.drawing_tool = 0  # 0: 钢笔, 1: 矩形, 2: 圆形
        
        # 用于调整框的点标记
        self.resizing_box = None  # 当前调整的框索引
        self.resize_handle = None  # 当前调整的角点索引
        self.is_resizing = False  # 是否正在调整大小
        self.dragging_box = None  # 当前拖动的框索引
        
    def set_image(self, image_path, preserve_view_state=False):
        """设置要显示的图像
        
        Args:
            image_path: 图像文件路径
            preserve_view_state: 是否保留当前的视图状态（缩放比例和偏移）
        """
        try:
            # 确保路径是字符串类型
            if not isinstance(image_path, str):
                error_msg = f"加载图像出错: 路径必须是字符串类型，而不是 {type(image_path)}"
                print(error_msg)
                return False
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                error_msg = f"加载图像出错: 文件不存在 - {image_path}"
                print(error_msg)
                return False
            
            # 保存当前视图状态
            old_scale_factor = self.scale_factor
            old_offset = QPoint(self.offset) if self.offset else QPoint(0, 0)
            
            # 加载图像
            self.image = Image.open(image_path).convert("RGB")
            self.displayed_image = self.image.copy()
            
            # 根据参数决定是否保留视图状态
            if preserve_view_state:
                self.scale_factor = old_scale_factor
                self.offset = old_offset
            else:
                self.offset = QPoint(0, 0)
                # 计算适合视图的缩放比例，确保图像完全显示在视图内
                # 获取视图大小（考虑最小高度限制）
                view_width = self.width() if self.width() > 100 else 400  # 避免宽度过小
                view_height = self.height() if self.height() > 100 else 400  # 避免高度过小
                
                # 计算缩放比例
                scale_x = view_width / self.image.width
                scale_y = view_height / self.image.height
                # 取较小的缩放比例，确保图像完全显示
                self.scale_factor = min(scale_x, scale_y, 1.0)  # 最大不超过1.0倍
            
            # 清除所有标记
            self.points = []
            self.boxes = []
            self.contour_points = []
            self.mask = None
            
            self.update()
            return True
        except Exception as e:
            error_msg = f"加载图像出错: {type(e).__name__} - {str(e)}"
            print(error_msg)
            return False
    
    def set_mode(self, mode):
        """设置操作模式: 0-平移, 1-点选, 2-框选, 3-轮廓"""
        self.mode = mode
        # 更新鼠标指针
        if mode == 0:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif mode == 1:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == 2:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == 3:
            self.setCursor(Qt.CursorShape.CrossCursor)
    
    def add_point(self, x, y, label):
        """添加一个点标记"""
        self.points.append((x, y, label))
        self.update()
    
    def add_box(self, x1, y1, x2, y2):
        """添加一个框标记"""
        self.boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        self.update()
    
    def set_mask(self, mask, color=None, label=""):
        """设置掩码
        
        Args:
            mask: 掩码数据（可以是轮廓点列表或NumPy数组）
            color: 掩码颜色
            label: 缺陷标签
        """
        self.mask = mask
        if color is not None:
            self.mask_color = color
        self.defect_label = label
        self.update()
        
    def create_polygon_mask(self, contour_points, width, height):
        """根据轮廓点创建多边形掩码
        
        Args:
            contour_points: 轮廓点列表
            width: 掩码宽度
            height: 掩码高度
            
        Returns:
            NumPy数组形式的掩码
        """
        import cv2
        # 创建空掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 将轮廓点转换为OpenCV格式
        points = np.array([[int(x), int(y)] for x, y in contour_points], dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        
        # 填充多边形
        cv2.fillPoly(mask, [points], 255)
        
        return mask
        
    def subtract_polygons(self, outer_contour, inner_contour, width, height):
        """从外多边形中减去内多边形，创建中空结构
        
        Args:
            outer_contour: 外部轮廓点列表
            inner_contour: 内部轮廓点列表
            width: 掩码宽度
            height: 掩码高度
            
        Returns:
            中空结构的掩码数组
        """
        import cv2
        # 创建外部多边形掩码
        outer_mask = self.create_polygon_mask(outer_contour, width, height)
        
        # 创建内部多边形掩码
        inner_mask = self.create_polygon_mask(inner_contour, width, height)
        
        # 执行减法操作（外部减去内部）
        hollow_mask = cv2.subtract(outer_mask, inner_mask)
        
        return hollow_mask
    
    def set_drawing_tool(self, tool):
        """设置绘制工具类型
        
        Args:
            tool: 0: 钢笔, 1: 矩形, 2: 圆形
        """
        self.drawing_tool = tool
    
    def _create_rect_contour(self, rect):
        """创建矩形轮廓点列表
        
        Args:
            rect: QRect对象
            
        Returns:
            矩形轮廓点列表
        """
        x1, y1, w, h = rect.getRect()
        x2, y2 = x1 + w, y1 + h
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]  # 闭合轮廓
    
    def _create_circle_contour(self, rect):
        """创建圆形轮廓点列表
        
        Args:
            rect: 包含圆的矩形
            
        Returns:
            圆形轮廓点列表
        """
        import math
        x1, y1, w, h = rect.getRect()
        center_x, center_y = x1 + w/2, y1 + h/2
        radius = min(w, h) / 2
        
        # 创建圆形轮廓（36个点）
        points = []
        for i in range(36):
            angle = 2 * math.pi * i / 36
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        # 闭合轮廓
        points.append(points[0])
        return points
    
    def _point_in_handle(self, pos, handle_pos):
        """检查点是否在控制点内
        
        Args:
            pos: 鼠标位置
            handle_pos: 控制点位置
            
        Returns:
            是否在控制点内
        """
        handle_size = 8  # 控制点大小
        return (handle_pos[0] - handle_size <= pos[0] <= handle_pos[0] + handle_size and
                handle_pos[1] - handle_size <= pos[1] <= handle_pos[1] + handle_size)
    
    def _get_handle_index(self, pos, contour):
        """获取鼠标点击的控制点索引
        
        Args:
            pos: 鼠标位置
            contour: 轮廓点列表
            
        Returns:
            控制点索引，如果没有点击控制点则返回-1
        """
        for i, point in enumerate(contour[:-1]):  # 排除最后一个闭合点
            if self._point_in_handle(pos, point):
                return i
        return -1
    
    def _is_in_contour(self, pos, contour):
        """检查点是否在轮廓内部
        
        Args:
            pos: 鼠标位置
            contour: 轮廓点列表
            
        Returns:
            是否在轮廓内部
        """
        # 使用射线法判断点是否在多边形内部
        x, y = pos
        inside = False
        n = len(contour) - 1  # 排除最后一个闭合点
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = contour[i]
            xj, yj = contour[j]
            
            # 检查点是否在多边形的边上
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        
        return inside
    
    def clear_all(self):
        """清除所有标记和掩码"""
        self.points = []
        self.boxes = []
        self.contour_points = []
        self.mask = None
        if self.image is not None:
            self.displayed_image = self.image.copy()
        self.resizing_box = None
        self.resize_handle = None
        self.is_resizing = False
        self.dragging_box = None
        # 确保完全重置钢笔工具状态
        self.drawing_contour = False
        self.update()
    
    def clear_last_point(self):
        """清除最后一个点标记"""
        if self.points:
            self.points.pop()
            self.update()
    
    def paintEvent(self, event):
        """绘制图像和所有标记"""
        if self.image is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 转换PIL图像为QPixmap
        img_array = np.array(self.displayed_image)
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 绘制缩放和偏移后的图像
        painter.translate(self.offset)
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawPixmap(0, 0, pixmap)
        
        # 绘制掩码
        if self.mask is not None:
            self._draw_mask(painter)
        
        # 绘制点标记
        self._draw_points(painter)
        
        # 绘制框标记
        self._draw_boxes(painter)
        
        # 绘制轮廓
        self._draw_contour(painter)
        
        # 绘制当前正在选择的框
        if self.selecting_box:
            self._draw_current_rect(painter)
            
        # 绘制缺陷类型标签
        if self.defect_label and self.mask is not None:
            self._draw_defect_label(painter)
        
        # 绘制轮廓控制点
        if self.contour_points and not self.selecting_box and not self.drawing_contour:
            self._draw_handles(painter)
    
    def _draw_mask(self, painter):
        """绘制掩码（显示半透明填充区域和轮廓线）"""
        if self.mask is None:
            return
        
        import cv2
        
        # 检查掩码类型并处理
        if isinstance(self.mask, list):
            # 处理轮廓点列表
            if not self.mask:
                return
            
            # 创建半透明填充笔刷
            fill_color = QColor(*self.mask_color[:3], 100)  # 100是alpha值，半透明
            painter.setBrush(QBrush(fill_color))
            
            # 创建轮廓线画笔
            contour_color = QColor(*self.mask_color[:3], 255)
            painter.setPen(QPen(contour_color, 2))
            
            # 转换轮廓点并绘制多边形
            q_points = [QPoint(int(point[0]), int(point[1])) for point in self.mask]
            if len(q_points) > 2:
                painter.drawPolygon(q_points)
        elif isinstance(self.mask, np.ndarray):
            # 处理NumPy数组形式的掩码（包括中空结构）
            mask_np = self.mask.astype(np.uint8)
            
            if np.any(mask_np):
                # 创建一个带有颜色和透明度的QImage
                height, width = mask_np.shape
                color = self.mask_color[:3]  # 获取RGB颜色
                alpha = 100  # 设置透明度
                
                # 创建RGBA图像
                rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                # 设置颜色和透明度
                rgba_image[:,:,0] = mask_np * color[0]  # R通道
                rgba_image[:,:,1] = mask_np * color[1]  # G通道
                rgba_image[:,:,2] = mask_np * color[2]  # B通道
                rgba_image[:,:,3] = mask_np * alpha     # Alpha通道
                
                # 将NumPy数组转换为QImage
                bytes_per_line = 4 * width
                q_image = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
                
                # 绘制QImage
                painter.drawImage(0, 0, q_image)
                
                # 另外，使用OpenCV找到所有轮廓，单独绘制轮廓线以增强视觉效果
                contours, hierarchy = cv2.findContours(
                    mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # 创建轮廓线画笔
                    contour_color = QColor(*self.mask_color[:3], 255)
                    painter.setPen(QPen(contour_color, 2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)  # 不填充，只绘制轮廓线
                    
                    # 绘制所有轮廓线
                    for contour in contours:
                        if len(contour) > 1:
                            q_points = [QPoint(point[0][0], point[0][1]) for point in contour]
                            painter.drawPolygon(q_points)

        # 绘制标签
        if self.defect_label:
            painter.setPen(Qt.PenStyle.SolidLine)
            painter.setBrush(Qt.BrushStyle.SolidPattern)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 20))
            painter.drawText(10, 20, self.defect_label)
    
    def _draw_points(self, painter):
        """绘制点标记"""
        for x, y, label in self.points:
            # 点的颜色：包含点(1)为绿色，排除点(0)为红色
            color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(255, 255, 255), 1))  # 白色边框
            
            # 绘制点 - 将radius转换为整数
            radius = int(round(self.point_size / self.scale_factor))
            painter.drawEllipse(QPoint(x, y), radius, radius)
    
    def _draw_boxes(self, painter):
        """绘制框标记"""
        for x1, y1, x2, y2 in self.boxes:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
    
    def _draw_contour(self, painter):
        """绘制轮廓"""
        if len(self.contour_points) > 1:
            painter.setPen(QPen(QColor(0, 255, 255), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # 绘制轮廓线
            for i in range(len(self.contour_points) - 1):
                x1, y1 = self.contour_points[i]
                x2, y2 = self.contour_points[i + 1]
                # 转换为整数坐标
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            
            # 如果轮廓已闭合，连接最后一个点和第一个点
            if not self.drawing_contour and len(self.contour_points) > 2:
                x1, y1 = self.contour_points[-1]
                x2, y2 = self.contour_points[0]
                # 转换为整数坐标
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            
            # 绘制最后一个点到鼠标位置的线（如果正在绘制）
            if self.drawing_contour and self.contour_points:
                last_x, last_y = self.contour_points[-1]
                mouse_pos = self.mapFromGlobal(QCursor.pos())
                mouse_x = mouse_pos.x()
                mouse_y = mouse_pos.y()
                # 转换鼠标坐标到图像坐标并四舍五入为整数
                img_x = round((mouse_x - self.offset.x()) / self.scale_factor)
                img_y = round((mouse_y - self.offset.y()) / self.scale_factor)
                # 转换为整数坐标
                painter.drawLine(int(last_x), int(last_y), img_x, img_y)
            
            # 绘制轮廓点
            for x, y in self.contour_points:
                painter.setBrush(QColor(0, 255, 255))
                radius = round(3 / self.scale_factor)
                # 转换为整数坐标
                painter.drawEllipse(QPoint(int(x), int(y)), radius, radius)
    
    def _draw_handles(self, painter):
        """绘制轮廓控制点"""
        handle_size = round(8 / self.scale_factor)
        
        for point in self.contour_points[:-1]:  # 排除最后一个闭合点
            # 绘制控制点
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            # 转换为整数坐标
            painter.drawEllipse(QPoint(int(point[0]), int(point[1])), handle_size // 2, handle_size // 2)
    
    def _draw_current_rect(self, painter):
        """绘制当前正在选择的框"""
        if not self.current_rect.isNull():
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0, 50)))
            painter.drawRect(self.current_rect)
    
    def _draw_defect_label(self, painter):
        """绘制缺陷类型标签"""
        # 找到掩码的中心点作为标签位置
        if self.mask is not None:
            # 计算掩码的中心点
            mask_np = self.mask.astype(np.uint8)
            y_indices, x_indices = np.where(mask_np > 0)
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                
                # 绘制标签背景
                painter.setBrush(QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(self.parent.font())
                
                # 计算文本尺寸
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(self.defect_label)
                text_height = font_metrics.height()
                
                # 绘制文本框
                text_rect = QRect(center_x + 10, center_y - text_height // 2, 
                                 text_width + 10, text_height + 4)
                painter.drawRect(text_rect)
                
                # 绘制文本
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.defect_label)
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if self.image is None:
            return
        
        # 转换鼠标位置到图像坐标系
        view_pos = event.position()
        x = int((view_pos.x() - self.offset.x()) / self.scale_factor)
        y = int((view_pos.y() - self.offset.y()) / self.scale_factor)
        
        if event.button() == Qt.MouseButton.LeftButton:
            # 检查是否点击了现有轮廓的控制点
            if self.contour_points and not self.selecting_box and not self.drawing_contour:
                handle_idx = self._get_handle_index((x, y), self.contour_points)
                if handle_idx >= 0:
                    self.is_resizing = True
                    self.resize_handle = handle_idx
                    self.start_pos = view_pos
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    return
                
                # 检查是否点击了轮廓内部（用于拖动）
                if self._is_in_contour((x, y), self.contour_points):
                    self.dragging_box = True
                    self.start_pos = view_pos
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return
            
            if self.mode == 0:  # 平移模式
                self.start_pos = view_pos
                self.dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.mode == 1:  # 点选模式
                # 发送包含点信号 (标签为1)
                self.mouse_clicked.emit(x, y, 1)
            elif self.mode == 2:  # 框选模式
                # 清除之前的轮廓
                self.contour_points = []
                self.start_pos = view_pos
                self.selecting_box = True
            elif self.mode == 3:  # 轮廓模式
                # 根据绘制工具类型决定操作
                if self.drawing_tool == 0:  # 钢笔工具
                    # 添加轮廓点
                    self.contour_points.append((x, y))
                    self.drawing_contour = True
                    self.contour_updated.emit(self.contour_points, False)  # 初始发送不闭合的轮廓
                    self.update()
                else:  # 矩形或圆形工具
                    # 清除之前的轮廓
                    self.contour_points = []
                    self.start_pos = view_pos
                    self.selecting_box = True
        elif event.button() == Qt.MouseButton.RightButton:
            if self.mode == 1:
                # 点选模式下右键添加排除点
                self.mouse_clicked.emit(x, y, 0)
            elif self.mode == 3 and self.contour_points and len(self.contour_points) > 2:
                # 轮廓模式下右键闭合轮廓
                self.drawing_contour = False
                # 发送闭合的轮廓
                self.contour_updated.emit(self.contour_points, True)
                self.update()
        elif event.button() == Qt.MouseButton.MiddleButton:
            # 中键移动图像
            self.start_pos = event.position()
            self.dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.image is None:
            return
        
        view_pos = event.position()
        x = int((view_pos.x() - self.offset.x()) / self.scale_factor)
        y = int((view_pos.y() - self.offset.y()) / self.scale_factor)
        
        if self.is_resizing and self.resize_handle is not None:
            # 调整轮廓点位置
            delta_x = (view_pos.x() - self.start_pos.x()) / self.scale_factor
            delta_y = (view_pos.y() - self.start_pos.y()) / self.scale_factor
            
            # 更新选中的控制点
            self.contour_points[self.resize_handle] = (
                self.contour_points[self.resize_handle][0] + delta_x,
                self.contour_points[self.resize_handle][1] + delta_y
            )
            
            # 如果是矩形，根据调整的角点更新对边的角点
            if len(self.contour_points) == 5:  # 矩形（4个点+闭合点）
                # 更新闭合点
                self.contour_points[4] = self.contour_points[0]
            
            self.start_pos = view_pos
            self.update()
            return
        
        if self.dragging_box:
            # 拖动整个轮廓
            delta_x = (view_pos.x() - self.start_pos.x()) / self.scale_factor
            delta_y = (view_pos.y() - self.start_pos.y()) / self.scale_factor
            
            # 移动所有点
            for i in range(len(self.contour_points)):
                self.contour_points[i] = (
                    self.contour_points[i][0] + delta_x,
                    self.contour_points[i][1] + delta_y
                )
            
            self.start_pos = view_pos
            self.update()
            return
        
        if self.dragging and self.start_pos is not None:
            # 平移图像
            delta = view_pos - self.start_pos
            # 将QPointF转换为QPoint
            self.offset += QPoint(int(delta.x()), int(delta.y()))
            self.start_pos = view_pos
            self.update()
        elif self.selecting_box and self.start_pos is not None:
            # 更新选择框
            x1 = min(self.start_pos.x(), view_pos.x())
            y1 = min(self.start_pos.y(), view_pos.y())
            x2 = max(self.start_pos.x(), view_pos.x())
            y2 = max(self.start_pos.y(), view_pos.y())
            
            # 转换到图像坐标系
            img_x1 = int((x1 - self.offset.x()) / self.scale_factor)
            img_y1 = int((y1 - self.offset.y()) / self.scale_factor)
            img_x2 = int((x2 - self.offset.x()) / self.scale_factor)
            img_y2 = int((y2 - self.offset.y()) / self.scale_factor)
            
            self.current_rect = QRect(img_x1, img_y1, img_x2 - img_x1, img_y2 - img_y1)
            self.update()
        
        # 对于轮廓模式，更新最后一段线的预览
        if self.drawing_contour and self.contour_points:
            self.update()
        
        # 更新光标形状（当鼠标悬停在控制点上时）
        if self.contour_points and not self.selecting_box and not self.drawing_contour and not self.dragging:
            handle_idx = self._get_handle_index((x, y), self.contour_points)
            if handle_idx >= 0:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif self._is_in_contour((x, y), self.contour_points):
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                if self.mode == 0:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                elif self.mode in [1, 2, 3]:
                    self.setCursor(Qt.CursorShape.CrossCursor)
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if self.image is None:
            return
        
        # 处理所有按钮的释放事件，包括中键
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_resizing:
                self.is_resizing = False
                self.resize_handle = None
                # 恢复光标
                if self.mode == 0:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                elif self.mode in [1, 2, 3]:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                return
            
            if self.dragging_box:
                self.dragging_box = False
                # 恢复光标
                if self.mode == 0:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                elif self.mode in [1, 2, 3]:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                return
            
            if self.dragging:
                self.dragging = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.selecting_box and self.start_pos is not None:
                # 完成框选
                self.selecting_box = False
                
                # 转换到图像坐标系
                view_pos = event.position()
                x1 = int((self.start_pos.x() - self.offset.x()) / self.scale_factor)
                y1 = int((self.start_pos.y() - self.offset.y()) / self.scale_factor)
                x2 = int((view_pos.x() - self.offset.x()) / self.scale_factor)
                y2 = int((view_pos.y() - self.offset.y()) / self.scale_factor)
                
                # 创建矩形
                rect = QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                
                # 根据绘制工具类型创建不同的轮廓
                if self.drawing_tool == 1:  # 矩形
                    self.contour_points = self._create_rect_contour(rect)
                elif self.drawing_tool == 2:  # 圆形
                    self.contour_points = self._create_circle_contour(rect)
                else:  # 默认矩形
                    self.contour_points = self._create_rect_contour(rect)
                
                # 重置当前矩形
                self.current_rect = QRect()
                
                # 更新显示
                self.update()
        elif event.button() == Qt.MouseButton.MiddleButton:
            if self.dragging:
                self.dragging = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
    
    def wheelEvent(self, event):
        """处理鼠标滚轮事件，用于缩放图像"""
        if self.image is None:
            return
        
        # 获取滚轮增量
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        
        # 获取鼠标当前位置
        mouse_pos = event.position()
        
        # 计算缩放前鼠标在图像中的位置
        old_x = (mouse_pos.x() - self.offset.x()) / self.scale_factor
        old_y = (mouse_pos.y() - self.offset.y()) / self.scale_factor
        
        # 应用缩放
        self.scale_factor *= factor
        
        # 调整偏移量，使鼠标指向的图像点保持不变
        new_x = (mouse_pos.x() - self.offset.x()) / self.scale_factor
        new_y = (mouse_pos.y() - self.offset.y()) / self.scale_factor
        
        self.offset += QPoint(
            int((new_x - old_x) * self.scale_factor),
            int((new_y - old_y) * self.scale_factor)
        )
        
        self.update()
    
    def keyPressEvent(self, event):
        """处理键盘事件"""
        if event.key() == Qt.Key.Key_Escape:
            # 取消当前操作
            if self.selecting_box:
                self.selecting_box = False
                self.current_rect = QRect()
                self.update()
            elif self.drawing_contour:
                self.contour_points = []
                self.drawing_contour = False
                self.contour_updated.emit(self.contour_points, False)  # 初始发送不闭合的轮廓
                self.update()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # 完成轮廓绘制
            if self.drawing_contour and len(self.contour_points) > 2:
                # 闭合轮廓
                self.contour_points.append(self.contour_points[0])
                self.drawing_contour = False
                self.contour_updated.emit(self.contour_points, True)
                self.update()
    
    def get_points(self):
        """获取所有点标记"""
        return self.points
    
    def get_boxes(self):
        """获取所有框标记"""
        return self.boxes
    
    def get_contour(self):
        """获取轮廓点"""
        return self.contour_points
