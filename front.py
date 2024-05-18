import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QMessageBox, QComboBox, QFileDialog, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
import torch
import json
print(torch.cuda.is_available())
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict, annotate

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_mask_pixels(masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(save_dir, f"mask_{idx}.png")
        cv2.imwrite(mask_path, mask)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplicación de Cámara")
        self.resize(800, 600)
        
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)

        self.camera_combo = QComboBox(self)
        self.populate_camera_list()
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

        self.capture_button = QPushButton("Captura", self)
        self.capture_button.setFont(QFont("Arial", 14))
        self.capture_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border: none; padding: 10px; }"
                                           "QPushButton:hover { background-color: #45a049; }")
        self.capture_button.clicked.connect(self.capture_and_label_image)

        self.upload_button = QPushButton("Cargar Imagen", self)
        self.upload_button.setFont(QFont("Arial", 14))
        self.upload_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; border: none; padding: 10px; }"
                                         "QPushButton:hover { background-color: #FB8C00; }")
        self.upload_button.clicked.connect(self.upload_and_label_image)

        self.dir_button = QPushButton("Seleccionar Directorio", self)
        self.dir_button.setFont(QFont("Arial", 14))
        self.dir_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; border: none; padding: 10px; }"
                                      "QPushButton:hover { background-color: #FB8C00; }")
        self.dir_button.clicked.connect(self.select_directory)

        self.add_label_button = QPushButton("Agregar Etiquetas", self)
        self.add_label_button.setFont(QFont("Arial", 14))
        self.add_label_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; border: none; padding: 10px; }"
                                            "QPushButton:hover { background-color: #1E88E5; }")
        self.add_label_button.clicked.connect(self.add_labels)

        self.exit_button = QPushButton("Salir", self)
        self.exit_button.setFont(QFont("Arial", 14))
        self.exit_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; border: none; padding: 10px; }"
                                       "QPushButton:hover { background-color: #d32f2f; }")
        self.exit_button.clicked.connect(self.close_application)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.camera_combo, alignment=Qt.AlignCenter)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.capture_button)
        control_layout.addWidget(self.upload_button)
        control_layout.addWidget(self.dir_button)
        control_layout.addWidget(self.add_label_button)
        control_layout.addWidget(self.exit_button)

        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

        self.capture_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Capturas")
        os.makedirs(self.capture_dir, exist_ok=True)

        self.camera_index = 0
        self.camera = cv2.VideoCapture(self.camera_index)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_camera)

        self.labels = []

        self.show_welcome_message()

        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(self.sam_model)

        groundingdino_config_path = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        groundingdino_checkpoint = "./checkpoints/groundingdino_swint_ogc.pth"
        self.groundingdino_model = load_model(groundingdino_config_path, groundingdino_checkpoint)

    def populate_camera_list(self):
        index = 0
        while True:
            camera = cv2.VideoCapture(index)
            if not camera.isOpened():
                break
            self.camera_combo.addItem(f"Cámara {index}")
            camera.release()
            index += 1

    def show_welcome_message(self):
        welcome_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(welcome_image, "¡Bienvenido!", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        self.display_image(welcome_image)

    def display_camera(self):
        ret, frame = self.camera.read()
        if ret:
            self.display_image(frame)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)

    def capture_and_label_image(self):
        ret, frame = self.camera.read()
        if ret:
            image_path = os.path.join(self.capture_dir, "captura.jpg")
            cv2.imwrite(image_path, frame)

            input_point = np.array([[320, 240]])
            input_label = np.array([1])

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb)

            masks, _, _ = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )

            masks_dir = os.path.join(self.capture_dir, "masks")
            save_mask_pixels(masks, masks_dir)

            masks_np = [mask[0].detach().cpu().numpy() if torch.is_tensor(mask) else mask for mask in masks]
            masks_resized = [cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0])) for mask in masks_np]

            mask_image = np.zeros_like(frame)
            for idx, mask in enumerate(masks_resized):
                color = self.get_color(idx)
                mask_image[mask > 0.5] = color

            segmented_image = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)
            segmented_image_path = os.path.join(self.capture_dir, "captura_segmentada.jpg")
            cv2.imwrite(segmented_image_path, segmented_image)

            # Procesamiento de GroundingDINO para obtener bounding boxes
            image_pil, _ = load_image(image_path)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Utilizar GPU si está disponible
            self.groundingdino_model.to(device)

            boxes, logits, phrases = predict(
                model=self.groundingdino_model,
                image=image_pil,
                caption=",".join(self.labels),
                box_threshold=0.3,
                text_threshold=0.25,
                device=device  # Pasar el dispositivo a la función predict
            )

            annotated_image = annotate(image_pil, boxes, phrases)
            annotated_image_cv2 = np.array(annotated_image)
            annotated_image_cv2 = cv2.cvtColor(annotated_image_cv2, cv2.COLOR_RGB2BGR)
            annotated_image_path = os.path.join(self.capture_dir, "captura_annotada.jpg")
            cv2.imwrite(annotated_image_path, annotated_image_cv2)

            labels_path = os.path.join(self.capture_dir, "labels.json")
            with open(labels_path, 'w') as f:
                json.dump(self.labels, f)

            self.display_image(annotated_image_cv2)

            QMessageBox.information(self, "Captura y Etiquetado", f"¡La imagen segmentada se guardó en: {self.capture_dir}\nEtiquetas guardadas en labels.json")
        else:
            QMessageBox.warning(self, "Error", "¡No se pudo capturar la imagen!")

    def upload_and_label_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            image = cv2.imread(file_path)
            self.display_image(image)
            input_text = QInputDialog.getText(self, "Etiquetas", "Introduce etiquetas separadas por comas:")
            if input_text[1]:
                self.labels = [label.strip() for label in input_text[0].split(",")]
                QMessageBox.information(self, "Etiquetas", f"Etiquetas agregadas: {', '.join(self.labels)}")

    def select_directory(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio", "", options=options)
        if dir_path:
            self.capture_dir = dir_path
            QMessageBox.information(self, "Directorio Seleccionado", f"Directorio seleccionado: {self.capture_dir}")

    def add_labels(self):
        input_text = QInputDialog.getText(self, "Etiquetas", "Introduce etiquetas separadas por comas:")
        if input_text[1]:
            self.labels = [label.strip() for label in input_text[0].split(",")]
            QMessageBox.information(self, "Etiquetas", f"Etiquetas agregadas: {', '.join(self.labels)}")

    def close_application(self):
        self.camera.release()
        self.close()

    def get_color(self, idx):
        np.random.seed(idx)
        return tuple(np.random.randint(0, 255, 3).tolist())

    def change_camera(self, index):
        self.camera.release()
        self.camera_index = index
        self.camera = cv2.VideoCapture(self.camera_index)
        self.timer.start(30)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
