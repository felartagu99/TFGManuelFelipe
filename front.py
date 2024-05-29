import os
import shutil
import sys
import cv2
import numpy as np
import json
import torch
import supervision as sv
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QMessageBox, QComboBox, QFileDialog, QInputDialog, QDialog, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
#from segment_anything import SamPredictor, sam_model_registry
import xml.etree.ElementTree as ET
from xml.dom import minidom

from groundingdino.util.inference import load_model, load_image, predict, annotate

#Funciones aux para poder crear directorios y generar nombres con id
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_next_id(directory, base_name, extension):
    existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
    if not existing_files:
        return 1
    existing_ids = [int(f[len(base_name)+1:-len(extension)]) for f in existing_files if f[len(base_name)+1:-len(extension)].isdigit()]
    return max(existing_ids, default=0) + 1

def export_to_pascal_voc_annotations(image_path, boxes, phrases, output_dir, labels):
    image = cv2.imread(image_path)
    height, width, depth = image.shape

    # Crear el elemento raíz del XML
    annotation = ET.Element('annotation')

    # Sub-elementos básicos
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.basename(output_dir)

    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)

    path = ET.SubElement(annotation, 'path')
    path.text = image_path

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    width_el = ET.SubElement(size, 'width')
    width_el.text = str(width)
    height_el = ET.SubElement(size, 'height')
    height_el.text = str(height)
    depth_el = ET.SubElement(size, 'depth')
    depth_el.text = str(depth)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    # Sub-elementos para cada objeto
    for box, phrase in zip(boxes, phrases):
        if phrase in labels:
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)

            obj = ET.SubElement(annotation, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = phrase

            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'

            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'

            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(x_min)
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(y_min)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(x_max)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(y_max)

    xml_str = ET.tostring(annotation, encoding='utf-8')
    xml_pretty_str = minidom.parseString(xml_str).toprettyxml(indent='    ')

    # Obtener el siguiente ID para el archivo XML
    xml_id = get_next_id(output_dir, "annotated_image", ".xml")
    annotation_file_path = os.path.join(output_dir, f"annotated_image_{xml_id}.xml")

    # Guardar el archivo XML
    with open(annotation_file_path, 'w') as f:
        f.write(xml_pretty_str)
    print(f"Anotaciones Pascal VOC guardadas en: {annotation_file_path}")
    
def export_to_coco(image_path, boxes, phrases, output_dir, labels):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    annotation_dir = os.path.join(output_dir, "annotations")
    create_dir(annotation_dir)

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

    annotation_id = len(os.listdir(annotation_dir)) + 1  # Obtener el ID siguiente
    annotation_file_name = f"annotations_coco_{annotation_id}.json"
    annotation_file_path = os.path.join(annotation_dir, annotation_file_name)

    for box, phrase in zip(boxes, phrases):
        class_id = labels.index(phrase) + 1 if phrase in labels else -1
        if class_id != -1:
            x_center, y_center, box_width, box_height = box
            x_min = int((x_center - box_width / 2.0) * width)
            y_min = int((y_center - box_height / 2.0) * height)
            box_width = int(box_width * width)
            box_height = int(box_height * height)

            bbox = [x_min, y_min, box_width, box_height]

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": 1,
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0
            })

            annotation_id += 1

    with open(annotation_file_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"Anotaciones COCO guardadas en: {annotation_file_path}")

def export_to_coco_segmentation(image_path, labels, bboxes, masks):
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

def export_to_yolo(image_path, boxes, phrases, output_dir, labels):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    image_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    create_dir(image_dir)
    create_dir(labels_dir)

    # Crear archivo classes.txt
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Saved classes to {classes_path}")


    # Guardar etiquetas en formato YOLO
    txt_id = get_next_id(output_dir, "annotated_image", ".txt")
    label_file_name = f"annotated_image_{txt_id}.txt"
    label_path = os.path.join(labels_dir, label_file_name)

    yolo_annotations = []

    for box, phrase in zip(boxes, phrases):
        class_id = labels.index(phrase) if phrase in labels else -1
        if class_id != -1:  
            x_center, y_center, box_width, box_height = box
            x_center = max(0, min(x_center, 1))  # Asegurar que esté en el rango [0, 1]
            y_center = max(0, min(y_center, 1))
            box_width = max(0, min(box_width, 1))
            box_height = max(0, min(box_height, 1))
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_annotations))
    print(f"Saved YOLO annotations to {label_path}")

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

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

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

        self.upload_button = QPushButton("Cargar Imagenes", self)
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
        '''
        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(self.sam_model)
        '''
        self.change_camera(0)
        
        self.groundingdino_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./checkpoints/groundingdino_swint_ogc.pth")
    
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

    def upload_and_label_image(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Cargar Imágenes", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_paths:
            self.image_paths = file_paths
            self.current_image_index = 0
            input_text = QInputDialog.getText(self, "Etiquetas", "Introduce etiquetas separadas por comas:")
            if input_text[1]:
                self.labels = [label.strip() for label in input_text[0].split(",")]
                QMessageBox.information(self, "Etiquetas", f"Etiquetas agregadas: {', '.join(self.labels)}")
                self.progress_bar.setMaximum(len(self.image_paths))
                self.progress_bar.setValue(0)
                self.process_next_image()

    def process_next_image(self):
        if self.current_image_index < len(self.image_paths):
            original_image_path = self.image_paths[self.current_image_index]
            image_id = get_next_id(self.capture_dir, "captura", ".jpg")

            # Renombrar la imagen cargada
            new_image_path = os.path.join(self.capture_dir, f"captura_{image_id}.jpg")
            shutil.copy(original_image_path, new_image_path)
            
            # Procesar la imagen renombrada
            self.capture_and_label_image_from_path(new_image_path, image_id)
            
            self.current_image_index += 1
            self.progress_bar.setValue(self.current_image_index)
            if self.current_image_index < len(self.image_paths):
                self.process_next_image()
            else:
                QMessageBox.information(self, "Completado", "Todas las imágenes han sido procesadas.")
        else:
            QMessageBox.information(self, "Error", "No hay imágenes para procesar.")

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

    def select_directory(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio", "", options=options)
        if dir_path:
            self.capture_dir = dir_path
            QMessageBox.information(self, "Directorio Seleccionado", f"Directorio seleccionado: {self.capture_dir}")

    def add_labels(self):
        input_text, ok_pressed = QInputDialog.getText(self, "Etiquetas", "Introduce etiquetas separadas por comas:")

        if ok_pressed:
            labels = [label.strip() for label in input_text.split(",")]
            labels = [label for label in labels if label]  # Filtrar etiquetas vacías

            if labels:
                self.labels = labels
                QMessageBox.information(self, "Etiquetas", f"Etiquetas agregadas: {', '.join(self.labels)}")
            else:
                QMessageBox.critical(self, "Error", "No se han proporcionado etiquetas válidas. Por favor, introduce etiquetas separadas por comas.")
        else:
            QMessageBox.critical(self, "Error", "Se ha cancelado la entrada de etiquetas. Por favor, introduce etiquetas separadas por comas.")

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
            # Obtener el siguiente ID para los archivos
            image_id = get_next_id(self.capture_dir, "captura", ".jpg")
        
            image_path = os.path.join(self.capture_dir, f"captura_{image_id}.jpg")
            cv2.imwrite(image_path, frame)

            # Ejecutar GroundingDINO
            TEXT_PROMPT = ", ".join(self.labels)
            BOX_THRESHOLD = 0.35
            TEXT_THRESHOLD = 0.25
            image_source, image = load_image(image_path)

            boxes, logits, phrases = predict(
                model=self.groundingdino_model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device="cpu"
            )

            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            annotated_image_path = os.path.join(self.capture_dir, f"annotated_image_{image_id}.jpg")
            cv2.imwrite(annotated_image_path, annotated_frame)

            confirmation_dialog = ConfirmationDialog(annotated_frame, self)
        
            if confirmation_dialog.exec_() == QDialog.Accepted:
                labels_path = os.path.join(self.capture_dir, f"labels_{image_id}.json")
                with open(labels_path, "w") as f:
                    json.dump(self.labels, f)
            
                export_format = self.format_combo.currentText()
            
                if export_format == "Pascal-VOC":
                    # Crear la estructura de directorios para Pascal VOC
                    pascal_voc_dir = os.path.join(self.capture_dir, "exportation_in_pascalVOC")
                    annotations_dir = os.path.join(pascal_voc_dir, "Annotations")
                    images_dir = os.path.join(pascal_voc_dir, "images")

                    create_dir(pascal_voc_dir)
                    create_dir(annotations_dir)
                    create_dir(images_dir)

                    # Copiar la imagen original a la carpeta "images"
                    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                    print(f"Imagen copiada a: {images_dir}")

                    
                    export_to_pascal_voc_annotations(image_path, boxes, phrases, annotations_dir, self.labels) 
                    QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato Pascal-VOC.\nDirectorio: {self.capture_dir}")

                elif export_format == "COCO":
                    # Crear la estructura de directorios para COCO
                    coco_dir = os.path.join(self.capture_dir, "exportation_in_COCO")
                    images_dir = os.path.join(coco_dir, "images")
                    annotations_dir = os.path.join(coco_dir, "annotations")

                    create_dir(coco_dir)
                    create_dir(images_dir)
                    create_dir(annotations_dir)

                    # Copiar la imagen original a la carpeta "images"
                    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                    print(f"Imagen copiada a: {images_dir}")

                    export_to_coco(image_path, boxes, phrases, coco_dir, self.labels)
                    QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato COCO.\nDirectorio: {self.capture_dir}")
            
                elif export_format == "YOLO":
                    # Crear la estructura de directorios para YOLO
                    yolo_dir = os.path.join(self.capture_dir, "exportation_in_YOLO")
                    images_dir = os.path.join(yolo_dir, "images")
                    labels_dir = os.path.join(yolo_dir, "labels")

                    create_dir(yolo_dir)
                    create_dir(images_dir)
                    create_dir(labels_dir)

                    # Copiar la imagen original a la carpeta "images"
                    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                    print(f"Imagen copiada a: {images_dir}")

                    
                    export_to_yolo(image_path, boxes, phrases, yolo_dir, self.labels)
                    QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato YOLO.\nDirectorio: {self.capture_dir}")

            else:
                print("Captura y etiquetado cancelados por el usuario.")
                QMessageBox.information(self, "Cancelado", "Captura y etiquetado cancelados por el usuario.")
            
            print("BOXES: ", boxes)

        else:
            QMessageBox.critical(self, "Error", "No se pudo capturar la imagen.")
    
    #capture_and_label 2.0 
    def capture_and_label_image_from_path(self, image_path, image_id):
        # Ejecutar GroundingDINO
        TEXT_PROMPT = ", ".join(self.labels)
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device="cpu"
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_image_path = os.path.join(self.capture_dir, f"annotated_image_{image_id}.jpg")
        cv2.imwrite(annotated_image_path, annotated_frame)

        confirmation_dialog = ConfirmationDialog(annotated_frame, self)
        
        if confirmation_dialog.exec_() == QDialog.Accepted:
            labels_path = os.path.join(self.capture_dir, f"labels_{image_id}.json")
            with open(labels_path, "w") as f:
                json.dump(self.labels, f)
            
            export_format = self.format_combo.currentText()
            
            if export_format == "Pascal-VOC":
                # Crear la estructura de directorios para Pascal VOC
                pascal_voc_dir = os.path.join(self.capture_dir, "exportation_in_pascalVOC")
                annotations_dir = os.path.join(pascal_voc_dir, "Annotations")
                images_dir = os.path.join(pascal_voc_dir, "images")

                create_dir(pascal_voc_dir)
                create_dir(annotations_dir)
                create_dir(images_dir)

                # Copiar la imagen original a la carpeta "images"
                shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                print(f"Imagen copiada a: {images_dir}")

                export_to_pascal_voc_annotations(image_path, boxes, phrases, annotations_dir, self.labels)
                QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato Pascal-VOC.\nDirectorio: {self.capture_dir}")

            elif export_format == "COCO":
                # Crear la estructura de directorios para COCO
                coco_dir = os.path.join(self.capture_dir, "exportation_in_COCO")
                images_dir = os.path.join(coco_dir, "images")
                annotations_dir = os.path.join(coco_dir, "annotations")

                create_dir(coco_dir)
                create_dir(images_dir)
                create_dir(annotations_dir)

                # Copiar la imagen original a la carpeta "images"
                shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                print(f"Imagen copiada a: {images_dir}")

                export_to_coco(image_path, boxes, phrases, coco_dir, self.labels)
                QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato COCO.\nDirectorio: {self.capture_dir}")
            
            elif export_format == "YOLO":
                # Crear la estructura de directorios para YOLO
                yolo_dir = os.path.join(self.capture_dir, "exportation_in_YOLO")
                images_dir = os.path.join(yolo_dir, "images")
                labels_dir = os.path.join(yolo_dir, "labels")

                create_dir(yolo_dir)
                create_dir(images_dir)
                create_dir(labels_dir)

                # Copiar la imagen original a la carpeta "images"
                shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                print(f"Imagen copiada a: {images_dir}")

                export_to_yolo(image_path, boxes, phrases, yolo_dir, self.labels)
                QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato YOLO.\nDirectorio: {self.capture_dir}")

        else:
            print("Captura y etiquetado cancelados por el usuario.")
            QMessageBox.information(self, "Cancelado", "Captura y etiquetado cancelados por el usuario.")
        
        print("BOXES: ", boxes)
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())