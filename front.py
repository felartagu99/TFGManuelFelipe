import os
import sys
import cv2
import numpy as np
import json
import torch
import supervision as sv
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QMessageBox, QComboBox, QFileDialog, QInputDialog, QDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
from segment_anything import SamPredictor, sam_model_registry
import xml.etree.ElementTree as ET
from xml.dom import minidom

#from groundingdino import predict, annotate
#from groundingdino.util.inference import load_image


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_mask_debug(masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(save_dir, f"debug_mask_{idx}.png")
        cv2.imwrite(mask_path, mask)
        print(f"Mask {idx} saved to {mask_path}")

def invert_mask(mask):
    return 1 - mask

def save_mask_pixels(masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(save_dir, f"mask_{idx}.png")
        cv2.imwrite(mask_path, mask)

def mask_to_bboxes(mask):
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 0:  # Filtra contornos pequeños
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, x + w, y + h))
    return bboxes

def crop_image_to_size(image, width_mm, height_mm, ppi=300):
        #Esta función se hace por el tema de la bandeja, para que no se pueda exceder de tamaño
        width_inch = width_mm / 25.4
        height_inch = height_mm / 25.4
        width_px = int(width_inch * ppi)
        height_px = int(height_inch * ppi)
        
        # Calcula las coordenadas del recorte
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        x1 = max(center_x - width_px // 2, 0)
        y1 = max(center_y - height_px // 2, 0)
        x2 = min(center_x + width_px // 2, image.shape[1])
        y2 = min(center_y + height_px // 2, image.shape[0])

        return image[y1:y2, x1:x2]

def export_to_pascal_voc(image_path, labels, bboxes):
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    image = cv2.imread(image_path)
    height, width, depth = image.shape

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = image_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(x1))
        ET.SubElement(bndbox, "ymin").text = str(int(y1))
        ET.SubElement(bndbox, "xmax").text = str(int(x2))
        ET.SubElement(bndbox, "ymax").text = str(int(y2))

    xml_str = ET.tostring(annotation, encoding="utf-8")
    xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")

    output_path = os.path.splitext(image_path)[0] + ".xml"
    with open(output_path, "w") as f:
        f.write(xml_str)

    print(f"Pascal VOC annotation saved to {output_path}")
        
def export_to_coco(image_path, labels, bboxes):

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    coco_output = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "COCO dataset generated",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [{
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": 1
        }],
        "annotations": [],
        "categories": [{"id": idx + 1, "name": label, "supercategory": "none"} for idx, label in enumerate(labels)]
    }

    for idx, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        coco_output["annotations"].append({
            "id": idx + 1,
            "image_id": 1,
            "category_id": idx + 1,
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "segmentation": [],
            "iscrowd": 0
        })

    save_path = os.path.splitext(image_path)[0] + "_coco.json"
    with open(save_path, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"Saved COCO annotation to {save_path}")

def export_to_coco_segmentation(image_path, labels, bboxes, masks):
    from datetime import datetime
    import json

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    coco_output = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "COCO dataset with segmentation generated",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [{
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": 1
        }],
        "annotations": [],
        "categories": [{"id": idx + 1, "name": label, "supercategory": "none"} for idx, label in enumerate(labels)]
    }

    for idx, (bbox, mask, label) in enumerate(zip(bboxes, masks, labels)):
        x1, y1, x2, y2 = bbox
        segmentation = []
        mask_pixels = np.where(mask > 0.5)
        segmentation.append(mask_pixels[1].tolist())
        segmentation.append(mask_pixels[0].tolist())
        annotation = {
            "id": idx + 1,
            "image_id": 1,
            "category_id": idx + 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1),
            "segmentation": segmentation,
            "iscrowd": 0
        }
        coco_output["annotations"].append(annotation)

    output_path = os.path.splitext(image_path)[0] + "_coco_segmentation.json"
    with open(output_path, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO segmentation annotation saved to {output_path}")
    
def export_to_yolo(image_path, labels, bboxes):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    yolo_output = []
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        label_idx = labels.index(label)  # assuming label is in labels list
        yolo_output.append(f"{label_idx} {x_center} {y_center} {w} {h}")

    output_path = os.path.splitext(image_path)[0] + ".txt"
    with open(output_path, "w") as f:
        f.write("\n".join(yolo_output))

    print(f"YOLO annotation saved to {output_path}")

class ConfirmationDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirmar Segmentación")
        
        dialog_width = 800
        dialog_height = 600
        self.setFixedSize(dialog_width, dialog_height)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.image = image

        # Convertir la imagen a QImage y luego a QPixmap
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(dialog_width - 20, dialog_height - 80, Qt.KeepAspectRatio)

        self.image_label.setPixmap(scaled_pixmap)

        button_layout = QHBoxLayout()
        accept_button = QPushButton("Aceptar", self)
        reject_button = QPushButton("Rechazar", self)

        accept_button.clicked.connect(self.accept)
        reject_button.clicked.connect(self.reject)

        button_layout.addWidget(accept_button)
        button_layout.addWidget(reject_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

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

        # Añadir QComboBox para seleccionar el formato de exportación
        self.format_combo = QComboBox(self)
        self.format_combo.addItem("Pascal-VOC")
        self.format_combo.addItem("COCO")        
        self.format_combo.addItem("COCO with Segmentation")        
        self.format_combo.addItem("YOLO")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.camera_combo, alignment=Qt.AlignCenter)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.capture_button)
        control_layout.addWidget(self.upload_button)
        control_layout.addWidget(self.dir_button)
        control_layout.addWidget(self.add_label_button)
        control_layout.addWidget(self.exit_button)
        
        # Añadir QComboBox al layout
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Formato de Exportación:", self))
        format_layout.addWidget(self.format_combo)
        
        main_layout.addLayout(format_layout)
        main_layout.addLayout(control_layout)
        
        self.setLayout(main_layout)

        self.capture_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Capturas")
        os.makedirs(self.capture_dir, exist_ok=True)

        self.camera_index = 0
        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_camera)

        self.labels = []

        self.timer.start(30)  # Asegúrate de iniciar el temporizador aquí

        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(self.sam_model)
        
        self.change_camera(0)
        
        #groundingdino_config_path = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        #groundingdino_checkpoint = "./checkpoints/groundingdino_swint_ogc.pth"
        # Cargar el modelo de groundingdino
        #self.groundingdino_model = torch.load(groundingdino_checkpoint, map_location=torch.device('cpu'))
        #self.groundingdino_model.eval()
    
    def populate_camera_list(self):
        index = 0
        while True:
            print(f"Trying to open camera with index {index}")
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                print(f"Camera {index} opened successfully")
                self.camera_combo.addItem(f"Cámara {index}")
                camera.release()
                index += 1
            else:
                print(f"Camera with index {index} could not be opened")
                break          
   
    def display_camera(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.display_image(frame)
            else:
                print("Failed to read frame from camera!")
                self.camera_label.setText("Error: Failed to read frame")
        else:
            print("Camera not available!")
            self.camera_label.setText("Cámara no disponible")

    def change_camera(self, index):
        print("Selected camera index:", index)
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
        self.camera_index = index
        print("Opening camera with index:", self.camera_index)
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            print("Failed to open camera!")
            QMessageBox.warning(self, "Error", "¡No se pudo abrir la cámara!")
        else:
            self.timer.start(30)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)

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

    def save_annotations(self, bboxes, masks):
        annotations = []
        for bbox, mask in zip(bboxes, masks):
            x1, y1, x2, y2 = bbox
            mask_pixels = np.where(mask > 0.5)
            annotations.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "mask_pixels": {
                    "x": mask_pixels[1].tolist(),
                    "y": mask_pixels[0].tolist()
                }
            })

        output_data = {
            "annotations": annotations,
            "labels": self.labels
        }

        annotations_path = os.path.join(self.capture_dir, "annotations.json")
        with open(annotations_path, 'w') as f:
            json.dump(output_data, f, default=int) 
            
    def capture_and_label_image(self):
        ret, frame = self.camera.read()
        if ret:
            image_path = os.path.join(self.capture_dir, "captura.jpg")
            cv2.imwrite(image_path, frame)

            # Segmentación con Segment Anything
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
            save_mask_debug(masks_resized, self.capture_dir)

            # Crear una imagen de objetos vacía
            object_image = np.zeros_like(frame)

            all_bboxes = []
            
            for idx, mask in enumerate(masks_resized):
                color = self.get_color(idx)
                object_image[mask > 0.5] = color

                # Calcular las bounding boxes
                mask_bboxes = mask_to_bboxes(mask)
                print(f"Mask {idx} bboxes: {mask_bboxes}")
                all_bboxes.extend(mask_bboxes)

            print("Bounding boxes:", all_bboxes)
            
            # Dibujar las bounding boxes en la imagen de objetos
            for bbox in all_bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Combinar la imagen de fondo (original) y la imagen de objetos
            combined_image = cv2.addWeighted(frame, 0.7, object_image, 0.3, 0)

            confirmation_dialog = ConfirmationDialog(combined_image, self)
            if confirmation_dialog.exec_() == QDialog.Accepted:
                segmented_image_path = os.path.join(self.capture_dir, "captura_segmentada.jpg")
                cv2.imwrite(segmented_image_path, combined_image)

                labels_path = os.path.join(self.capture_dir, "labels.json")
                with open(labels_path, 'w') as f:
                    json.dump(self.labels, f)

                self.save_annotations(all_bboxes, masks_resized)

                export_format = self.format_combo.currentText()
                if export_format == "Pascal-VOC":
                    export_to_pascal_voc(segmented_image_path, self.labels, all_bboxes)
                elif export_format == "COCO":
                    export_to_coco(segmented_image_path, self.labels, all_bboxes)
                elif export_format == "COCO with Segmentation":
                    if masks is None:
                        print("Masks are required for COCO-Segmentation format.")
                    else:
                        export_to_coco_segmentation(segmented_image_path, self.labels, all_bboxes, masks_resized)
                elif export_format == "YOLO":
                    export_to_yolo(segmented_image_path, self.labels, all_bboxes)
                else:
                    QMessageBox.warning(self, "Error", "Unsupported format selected.")

                self.display_image(frame)
                QMessageBox.information(self, "Captura y Etiquetado", f"¡La imagen segmentada se guardó en: {self.capture_dir}\nEtiquetas guardadas en labels.json")
            else:
                # Eliminar imágenes temporales
                if os.path.exists(image_path):
                    os.remove(image_path)
                for mask_path in os.listdir(masks_dir):
                    os.remove(os.path.join(masks_dir, mask_path))
                QMessageBox.information(self, "Captura Rechazada", "La captura segmentada fue rechazada.")
        else:
            QMessageBox.warning(self, "Error", "¡No se pudo capturar la imagen!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
