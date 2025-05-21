#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFrame, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFontDatabase, QFont, QIcon

# 1) resource_path function
def resource_path(relative_path: str) -> str:
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 2) Import your blackhole_renderer
from blackhole_renderer import render_image

class BlackHoleRenderer(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedSize(650, 900)
        # Use resource_path for window icon
        self.setWindowIcon(QIcon(resource_path("window_icon.png")))
        self.setWindowTitle("WARPix")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top Widget
        self.top_widget = QWidget(self)
        self.top_widget.setObjectName("topWidget")
        top_layout = QVBoxLayout(self.top_widget)
        top_layout.setContentsMargins(20, 40, 20, 40)
        top_layout.setSpacing(20)

        # Load font using resource_path
        font_id = QFontDatabase.addApplicationFont(resource_path("PressStart2P-Regular.ttf"))
        if font_id != -1:
            custom_font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        else:
            custom_font_family = "Arial"
        self.custom_font = QFont(custom_font_family, 10)

        # Title Label
        self.title_label = QLabel("WARPix")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont(custom_font_family, 20))
        top_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        # Spin Row
        spin_row = QHBoxLayout()
        self.spin_label = QLabel("Spin (a): 0.998")
        self.spin_label.setFont(self.custom_font)
        spin_row.addWidget(self.spin_label)

        self.spin_slider = QSlider(Qt.Horizontal)
        self.spin_slider.setMinimum(0)
        self.spin_slider.setMaximum(998)
        self.spin_slider.setValue(998)
        self.spin_slider.valueChanged.connect(self.update_parameters)
        spin_row.addWidget(self.spin_slider, stretch=1)
        top_layout.addLayout(spin_row)

        # Viewing Angle Row
        angle_row = QHBoxLayout()
        self.view_angle_label = QLabel("Viewing Angle (degrees): 90")
        self.view_angle_label.setFont(self.custom_font)
        angle_row.addWidget(self.view_angle_label)

        self.view_angle_slider = QSlider(Qt.Horizontal)
        self.view_angle_slider.setMinimum(1)
        self.view_angle_slider.setMaximum(179)
        self.view_angle_slider.setValue(90)
        self.view_angle_slider.valueChanged.connect(self.update_parameters)
        angle_row.addWidget(self.view_angle_slider, stretch=1)
        top_layout.addLayout(angle_row)

        # Render Button
        self.render_button = QPushButton("Render")
        self.render_button.setFont(QFont(custom_font_family, 12))
        self.render_button.setFixedSize(130, 40)
        self.render_button.clicked.connect(self.render_image)
        top_layout.addWidget(self.render_button, alignment=Qt.AlignCenter)

        # About Button
        self.about_button = QPushButton("About")
        self.about_button.setFont(QFont(custom_font_family, 10))
        self.about_button.setFixedSize(100, 30)
        self.about_button.clicked.connect(self.show_about_dialog)
        top_layout.addWidget(self.about_button, alignment=Qt.AlignCenter)

        main_layout.addWidget(self.top_widget)

        # Horizontal black line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(3)
        line.setStyleSheet("background-color: black;")
        main_layout.addWidget(line)

        # Bottom Widget
        self.bottom_widget = QWidget(self)
        self.bottom_widget.setObjectName("bottomWidget")
        bottom_layout = QVBoxLayout(self.bottom_widget)
        bottom_layout.setContentsMargins(20, 20, 20, 20)
        bottom_layout.setSpacing(20)

        # Image label
        self.image_label = QLabel()
        self.image_label.setFixedSize(500, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        bottom_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        scale_save_layout = QVBoxLayout()
        scale_save_layout.setSpacing(30)

        # Row for MIN, colormap, MAX
        scale_layout = QHBoxLayout()
        scale_layout.setSpacing(10)

        self.min_label = QLabel("MIN")
        self.min_label.setFont(self.custom_font)
        self.min_label.setAlignment(Qt.AlignCenter)
        scale_layout.addWidget(self.min_label)

        self.color_label = QLabel()
        self.color_label.setPixmap(self.create_color_scale(50, 10, scale_factor=4))
        scale_layout.addWidget(self.color_label)

        self.max_label = QLabel("MAX")
        self.max_label.setFont(self.custom_font)
        self.max_label.setAlignment(Qt.AlignCenter)
        scale_layout.addWidget(self.max_label)

        scale_save_layout.addLayout(scale_layout)

        # Save Button
        save_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Image")
        self.save_button.setFont(self.custom_font)
        self.save_button.clicked.connect(self.save_image_transparent)
        save_layout.addWidget(self.save_button, alignment=Qt.AlignCenter)
        scale_save_layout.addLayout(save_layout)

        bottom_layout.addLayout(scale_save_layout)
        main_layout.addWidget(self.bottom_widget)

        # 3) Fix the style references for starry background and slider handle
        starry_path = resource_path("starry_background.png")
        rocket_path = resource_path("rocket_handle_rotated.png")

        self.top_widget.setStyleSheet(f"""
            QWidget#topWidget {{
                background-image: url("{starry_path}");
                background-repeat: no-repeat;
                background-position: top center;
            }}
            QWidget#topWidget, QWidget#topWidget QLabel, QWidget#topWidget QPushButton {{
                color: #FFD700;
            }}
        
            QWidget#topWidget QSlider::groove:horizontal {{
                background: #999;
                height: 12px;
                border-radius: 6px;
                margin: 0; 
            }}

            QWidget#topWidget QSlider::sub-page:horizontal {{
                background: #FFD700;
                height: 12px;
                border-radius: 8px;
            }}
            QWidget#topWidget QSlider::add-page:horizontal {{
                background: #222;
                height: 12px;
                border-radius: 8px;
            }}
            QWidget#topWidget QSlider::handle:horizontal {{
                subcontrol-origin: margin;
                subcontrol-position: center;
                image: url("{rocket_path}");
                width: 100px;
                height: 100px;
                margin: -24px 0;
            }}
        """)

        self.bottom_widget.setStyleSheet("""
            QWidget#bottomWidget {
                background-color: black;
            }
            QWidget#bottomWidget QLabel {
                color: #FFD700;
            }
        """)

        # Auto-render at startup
        QTimer.singleShot(100, self.render_image)

    def update_parameters(self):
        spin_val = self.spin_slider.value() / 1000.0
        angle_val = self.view_angle_slider.value()
        self.spin_label.setText(f"Spin (a): {spin_val:.3f}")
        self.view_angle_label.setText(f"Viewing Angle (deg): {angle_val}")

    def render_image(self):
        spin_val = self.spin_slider.value() / 1000.0
        angle_val = self.view_angle_slider.value()

        self.render_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            img_array = render_image(a_val=spin_val, th0=angle_val)
        finally:
            QApplication.restoreOverrideCursor()
            self.render_button.setEnabled(True)

        height, width, _ = img_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def create_color_scale(self, width=50, height=10, scale_factor=4):
        import matplotlib.cm as cm
        from PIL import Image

        gradient = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        gradient = np.repeat(gradient, height, axis=0)

        cmap = cm.get_cmap("hot")
        image_rgba = cmap(gradient)
        image_rgb = (image_rgba[..., :3] * 255).astype(np.uint8)

        pil_img = Image.fromarray(image_rgb)
        scaled_w = width * scale_factor
        scaled_h = height * scale_factor
        pil_img = pil_img.resize((scaled_w, scaled_h), Image.NEAREST)

        qimg = QImage(pil_img.tobytes(), scaled_w, scaled_h, scaled_w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def save_image_transparent(self):
        from PyQt5.QtWidgets import QFileDialog

        pixmap = self.image_label.pixmap()
        if pixmap is None:
            QMessageBox.information(self, "Save Image", "No image to save!")
            return

        file_dialog_options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Image As", 
            "",
            "PNG Files (*.png);;All Files (*)",
            options=file_dialog_options
        )

        if not save_path:
            return

        qimg = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        for y in range(qimg.height()):
            for x in range(qimg.width()):
                color = qimg.pixelColor(x, y)
                if color.red() < 10 and color.green() < 10 and color.blue() < 10:
                    color.setAlpha(0)
                    qimg.setPixelColor(x, y, color)

        qimg.save(save_path, "PNG")
        QMessageBox.information(self, "Save Image", f"Image saved to:\n{save_path}")

    def show_about_dialog(self):
        info_text = (
            "WARPix\n\n"
            "A retro-style, CPU-based black hole ray tracer.\n\n"
            "Spin (a) controls the black hole's angular momentum.\n"
            "Viewing Angle (degrees) changes the observer's inclination.\n\n"
            "Developed by KKostaros, 2025.\n"
            "Enjoy exploring the cosmos!"
        )
        QMessageBox.information(self, "About WARPix", info_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = BlackHoleRenderer()
    gui.show()
    sys.exit(app.exec_())


