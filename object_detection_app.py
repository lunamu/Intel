import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QImage
from PyQt6.QtCore import Qt
import cv2
import numpy as np

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('物体检测应用')
        self.setGeometry(100, 100, 800, 600)

        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建图像显示标签
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        # 创建选择图片按钮
        self.select_button = QPushButton('选择图片', self)
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        self.image = None

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.detect_objects()

    def detect_objects(self):
        if self.image is None:
            return

        # 使用预训练的YOLOv3模型进行物体检测
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayersNames()

        height, width, channels = self.image.shape
        blob = cv2.dnn.blobFromImage(self.image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # 绘制边界框和标签
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(class_ids[i])
                color = colors[i]
                cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.image, label, (x, y + 30), font, 3, color, 3)

        # 将OpenCV图像转换为Qt图像并显示
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_image = QPixmap.fromImage(QImage(self.image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped())
        self.image_label.setPixmap(q_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ObjectDetectionApp()
    ex.show()
    sys.exit(app.exec())