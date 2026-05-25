import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import uic
from PyQt5.QtCore import QRect, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

try:
    import serial
except Exception:
    serial = None


APP_DIR = Path(__file__).resolve().parent
UI_FILE = APP_DIR / "smart_trash_ui.ui"
CONFIG_FILE = APP_DIR / "config.json"
LOGO_FILE = APP_DIR / "assets" / "hcmute_logo.jpg"
MODEL_FILE = APP_DIR / "models" / "trash_classifier.onnx"
MODEL_LABELS_FILE = APP_DIR / "models" / "model_classes.txt"


class CameraThread(QThread):
    frame_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.error_signal.emit("Không mở được camera. Hãy kiểm tra USB/IP camera.")
            self.running = False
            return

        while self.running:
            ok, frame = self.cap.read()
            if ok and self.running:
                self.frame_signal.emit(frame)
            time.sleep(0.03)

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.running = False

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if not self.wait(1500):
            self.terminate()
            self.wait(1000)


class SmartTrashApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(str(UI_FILE), self)

        self.camera_thread = None
        self.camera_running = False
        self.current_frame = None
        self.image_count = 0
        self.trash_classes = ["Rác hữu cơ", "Rác vô cơ", "Rác tái chế"]
        self.current_trash_class = "Chưa phân loại"
        self.current_object_label = "Vật thể"
        self.current_overlay_label = "Vật thể"
        self.ai_net = None
        self.ai_labels = self.trash_classes[:]
        self.last_model_label_detail = ""
        self.frame_counter = 0
        self.last_auto_capture_key = None
        self.last_auto_capture_time = 0.0
        self.last_seen_detection_time = 0.0
        self.auto_capture_locked = False
        self.bin_full_threshold = 90
        self.face_cascade = self.load_cascade("haarcascade_frontalface_default.xml")
        self.profile_face_cascade = self.load_cascade("haarcascade_profileface.xml")
        self.reset_detection_state()
        self.config = {
            "port": "COM3",
            "baudrate": 9600,
            "threshold": 0.5,
            "epochs": 10,
            "camera_source": "USB 0",
            "camera_url": "http://192.168.1.100:8080/video",
            "dataset_path": "dataset",
        }

        self.setup_style()
        self.setup_logo()
        self.connect_events()
        self.setup_clock()
        self.load_config()
        self.load_ai_model()
        self.refresh_stats()
        if hasattr(self, "spinBinFill"):
            self.update_bin_capacity(self.spinBinFill.value())
        self.show_page(0)

    def connect_events(self):
        self.btnPageMonitor.clicked.connect(lambda: self.show_page(0))
        self.btnPageHistory.clicked.connect(lambda: self.show_page(1))
        self.btnPageSystem.clicked.connect(lambda: self.show_page(2))
        self.btnPageSettings.clicked.connect(lambda: self.show_page(3))

        self.btnStartCamera.clicked.connect(self.start_monitor_camera)
        self.btnStopCamera.clicked.connect(lambda: self.stop_camera())
        self.btnClearLog.clicked.connect(self.clear_logs)

        self.btnSaveConfig.clicked.connect(self.save_config)
        self.btnSendUart.clicked.connect(self.send_uart_test)
        self.btnPreviewCamera.clicked.connect(self.start_preview_camera)
        self.btnClosePreview.clicked.connect(lambda: self.stop_camera())

        self.btnChooseFolder.clicked.connect(self.choose_dataset_folder)
        self.btnStartAddCamera.clicked.connect(self.start_add_camera)
        self.btnCapture.clicked.connect(self.capture_image)
        self.btnCreateDatabase.clicked.connect(self.create_database)
        self.btnOpenDataset.clicked.connect(self.open_dataset_folder)
        self.btnRefreshStats.clicked.connect(self.refresh_stats)
        if hasattr(self, "spinBinFill"):
            self.spinBinFill.valueChanged.connect(self.update_bin_capacity)

    def setup_logo(self):
        if LOGO_FILE.exists():
            pixmap = QPixmap(str(LOGO_FILE))
            if not pixmap.isNull():
                self.lblLogo.setText("")
                self.lblLogo.setPixmap(
                    pixmap.scaled(
                        self.lblLogo.width(),
                        self.lblLogo.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                self.lblLogo.setAlignment(Qt.AlignCenter)
                return

        self.lblLogo.setText("HCMUTE")

    def load_cascade(self, filename):
        path = Path(cv2.data.haarcascades) / filename
        cascade = cv2.CascadeClassifier(str(path))
        return cascade if not cascade.empty() else None

    def load_ai_model(self):
        if MODEL_LABELS_FILE.exists():
            labels = [
                line.strip().lstrip("\ufeff")
                for line in MODEL_LABELS_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip().lstrip("\ufeff")
            ]
            if labels:
                self.ai_labels = labels

        if not MODEL_FILE.exists():
            self.set_classification_result("Chưa có model AI", None)
            self.log("Chưa tìm thấy models/trash_classifier.onnx, đang chỉ nhận diện vùng vật thể.")
            return

        try:
            self.ai_net = cv2.dnn.readNetFromONNX(str(MODEL_FILE))
            self.set_classification_result("Sẵn sàng phân loại", None)
            self.log(f"Đã nạp model AI: {MODEL_FILE}")
        except Exception as exc:
            self.ai_net = None
            self.set_classification_result("Lỗi model AI", None)
            self.log(f"Lỗi nạp model AI: {exc}")

    def reset_detection_state(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=90,
            varThreshold=36,
            detectShadows=False,
        )
        self.previous_gray = None
        self.last_detection_box = None
        self.missing_detection_frames = 0
        self.detector_warmup_frames = 6
        self.stable_detection_box = None
        self.stable_detection_frames = 0

    def setup_clock(self):
        self.update_clock()
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def update_clock(self):
        days = [
            "THỨ HAI",
            "THỨ BA",
            "THỨ TƯ",
            "THỨ NĂM",
            "THỨ SÁU",
            "THỨ BẢY",
            "CHỦ NHẬT",
        ]
        now = datetime.now()
        self.lblClock.setText(f"{now:%H:%M} {days[now.weekday()]}, {now:%d/%m/%Y}")

    def show_page(self, index):
        self.stackedWidget.setCurrentIndex(index)
        buttons = [
            self.btnPageMonitor,
            self.btnPageHistory,
            self.btnPageSystem,
            self.btnPageSettings,
        ]

        for button in buttons:
            button.setProperty("active", False)
            button.style().unpolish(button)
            button.style().polish(button)

        buttons[index].setProperty("active", True)
        buttons[index].style().unpolish(buttons[index])
        buttons[index].style().polish(buttons[index])

        if index == 1:
            self.refresh_stats()

    def set_status(self, text):
        value = f"● {text}"
        if hasattr(self, "lblStatusBottom"):
            self.lblStatusBottom.setText(value)
        self.lblTopStatus.setText(value)

    def camera_source_config(self):
        source = self.comboCameraSource.currentText()
        if source == "USB 0":
            return 0
        if source == "USB 1":
            return 1
        return self.inputCameraUrl.text().strip()

    def camera_source_dataset(self):
        source = self.comboAddCamera.currentText()
        if source == "Mặc định (0)":
            return 0
        if source == "USB 1":
            return 1
        return self.inputAddCameraUrl.text().strip()

    def start_camera(self, source):
        self.stop_camera(clear=False)
        self.camera_running = True
        self.frame_counter = 0
        self.last_auto_capture_key = None
        self.last_auto_capture_time = 0.0
        self.last_seen_detection_time = 0.0
        self.auto_capture_locked = False
        self.reset_detection_state()
        self.set_classification_result(
            "Đang tìm vật thể" if self.ai_net else "Chưa có model AI",
            None,
        )
        self.camera_thread = CameraThread(source)
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.error_signal.connect(self.camera_error)
        self.camera_thread.start()
        self.set_status("Camera đang chạy")
        self.log("Đã mở camera")

    def start_monitor_camera(self):
        self.show_page(0)
        self.start_camera(self.camera_source_config())

    def start_preview_camera(self):
        self.show_page(3)
        self.start_camera(self.camera_source_config())

    def start_add_camera(self):
        self.show_page(2)
        self.start_camera(self.camera_source_dataset())

    def stop_camera(self, clear=True):
        self.camera_running = False
        if self.camera_thread:
            try:
                self.camera_thread.frame_signal.disconnect(self.update_frame)
            except TypeError:
                pass
            self.camera_thread.stop()
            self.camera_thread = None
            self.log("Đã tắt camera")

        self.current_frame = None
        self.reset_detection_state()
        self.set_status("Đang hoạt động")

        if clear:
            reset_labels = [
                (self.lblCameraMonitor, "CAMERA CHƯA ĐƯỢC KẾT NỐI"),
                (self.lblPreviewCamera, "Xem trước Camera"),
            ]
            if hasattr(self, "lblAddCamera"):
                reset_labels.append((self.lblAddCamera, "Camera thêm dữ liệu"))
            for label, text in reset_labels:
                label.clear()
                label.setText(text)

    def update_frame(self, frame):
        if not self.camera_running:
            return

        self.current_frame = frame
        self.frame_counter += 1
        detection_box = self.detect_trash_region(frame)
        if detection_box:
            self.last_seen_detection_time = time.time()
        human_detected = bool(detection_box and self.is_human_detection(frame, detection_box))
        stable_detection = bool(detection_box and self.update_detection_stability(detection_box))

        if human_detected:
            self.current_overlay_label = "Người - bỏ qua"
            self.lblAlertStatus.setText("Bỏ qua")
            self.lblAlertSub.setText("Phát hiện người, không tự chụp")
        elif detection_box and self.ai_net and stable_detection and self.frame_counter % 8 == 0:
            label, confidence = self.classify_trash_region(frame, detection_box)
            self.set_classification_result(label, confidence)
            self.auto_capture_classification(frame, detection_box, label, confidence)
        elif detection_box and not self.ai_net:
            self.current_object_label = "Vật thể"
            self.current_overlay_label = "Vật thể"
        elif detection_box and not stable_detection:
            self.current_object_label = "Vật thể"
            self.current_overlay_label = "Đang xác định vật thể"
        elif not detection_box:
            self.reset_auto_capture_if_object_left()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        page = self.stackedWidget.currentIndex()
        if page == 0:
            target = self.lblCameraMonitor
        elif page == 3:
            target = self.lblPreviewCamera
        elif hasattr(self, "lblAddCamera"):
            target = self.lblAddCamera
        else:
            return

        scaled = pixmap.scaled(
            target.width(),
            target.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        target.setPixmap(self.draw_detection_overlay(scaled, frame.shape, detection_box))

    def detect_trash_region(self, frame):
        height, width = frame.shape[:2]
        frame_area = height * width
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        motion_box = None
        if self.previous_gray is not None:
            diff = cv2.absdiff(self.previous_gray, gray)
            _, motion_mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
            motion_mask = self.clean_detection_mask(motion_mask)
            motion_box = self.find_largest_box(motion_mask, frame_area)
        self.previous_gray = gray

        foreground_mask = self.background_subtractor.apply(frame)
        _, foreground_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
        foreground_mask = self.clean_detection_mask(foreground_mask)
        foreground_box = None
        if self.detector_warmup_frames <= 0:
            foreground_box = self.find_largest_box(foreground_mask, frame_area)
        else:
            self.detector_warmup_frames -= 1

        detection_box = self.choose_detection_box(motion_box, foreground_box)
        if detection_box:
            self.last_detection_box = detection_box
            self.missing_detection_frames = 0
            return detection_box

        if self.last_detection_box and self.missing_detection_frames < 18:
            self.missing_detection_frames += 1
            return self.last_detection_box

        self.last_detection_box = None
        return None

    def clean_detection_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def find_largest_box(self, mask, frame_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(900, frame_area * 0.006)
        max_area = frame_area * 0.78
        best = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < 25 or height < 25:
                continue

            if best is None or area > best[0]:
                best = (area, (x, y, width, height))

        if not best:
            return None

        return self.expand_box(best[1], mask.shape[1], mask.shape[0], padding=14)

    def expand_box(self, box, frame_width, frame_height, padding=10):
        x, y, width, height = box
        x = max(0, x - padding)
        y = max(0, y - padding)
        right = min(frame_width, x + width + padding * 2)
        bottom = min(frame_height, y + height + padding * 2)
        return (x, y, right - x, bottom - y)

    def choose_detection_box(self, motion_box, foreground_box):
        if motion_box and foreground_box:
            motion_area = motion_box[2] * motion_box[3]
            foreground_area = foreground_box[2] * foreground_box[3]
            return motion_box if motion_area >= foreground_area * 0.45 else foreground_box
        return motion_box or foreground_box

    def update_detection_stability(self, detection_box):
        if self.stable_detection_box is None:
            self.stable_detection_box = detection_box
            self.stable_detection_frames = 1
            return False

        if self.box_iou(self.stable_detection_box, detection_box) >= 0.35:
            self.stable_detection_frames += 1
        else:
            self.stable_detection_frames = 1

        self.stable_detection_box = detection_box
        return self.stable_detection_frames >= 6

    def box_iou(self, first_box, second_box):
        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)

        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = aw * ah
        second_area = bw * bh
        union = first_area + second_area - intersection
        return intersection / union if union else 0

    def is_human_detection(self, frame, detection_box):
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = detection_box
        area_ratio = (width * height) / max(1, frame_width * frame_height)
        aspect_ratio = height / max(1, width)

        if area_ratio > 0.18 and height > frame_height * 0.52 and aspect_ratio > 1.05:
            return True

        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_count = 0

        for cascade in [self.face_cascade, self.profile_face_cascade]:
            if cascade is None:
                continue
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(32, 32),
            )
            face_count += len(faces)

        return face_count > 0

    def draw_detection_overlay(self, pixmap, frame_shape, detection_box):
        overlay = QPixmap(pixmap)
        if overlay.isNull():
            return overlay
        if not detection_box:
            return overlay

        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.Antialiasing)

        frame_height, frame_width = frame_shape[:2]
        scale_x = overlay.width() / frame_width
        scale_y = overlay.height() / frame_height
        x, y, width, height = detection_box
        box_x = int(x * scale_x)
        box_y = int(y * scale_y)
        box_width = int(width * scale_x)
        box_height = int(height * scale_y)

        painter.setPen(QPen(QColor("#ef4444"), 4))
        painter.drawRect(box_x, box_y, box_width, box_height)

        self.draw_text_badge(
            painter,
            self.current_overlay_label,
            box_x,
            box_y,
            overlay.width(),
            overlay.height(),
        )
        painter.end()
        return overlay

    def camera_error(self, message):
        QMessageBox.warning(self, "Lỗi Camera", message)
        self.set_status("Lỗi camera")
        self.lblAlertStatus.setText("Lỗi")
        self.lblAlertSub.setText("Kiểm tra camera")
        self.log(message)

    def set_classification_result(self, label, confidence=None):
        self.current_trash_class = label
        if confidence is not None:
            self.current_overlay_label = (
                f"{self.current_object_label} | {label} {confidence:.1f}%"
            )
        else:
            self.current_object_label = "Vật thể"
            self.current_overlay_label = "Vật thể"
        self.lblResult.setText(f"{label}\n{confidence:.1f}%" if confidence is not None else label)
        self.lblTrashNow.setText(label)
        self.lblConfidence.setText(f"{confidence:.1f}%" if confidence is not None else "--")
        self.lblAiClass.setText(label)
        detail = self.last_model_label_detail if confidence is not None else ""
        if hasattr(self, "spinBinFill") and self.spinBinFill.value() >= self.bin_full_threshold:
            self.lblAlertStatus.setText("Đầy")
            self.lblAlertSub.setText("Cần thu gom rác")
        else:
            self.lblAlertStatus.setText("Tốt")
            self.lblAlertSub.setText(detail or "Không có lỗi")

    def classify_trash_region(self, frame, detection_box):
        if self.ai_net is None:
            return "Chưa có model AI", None

        x, y, width, height = detection_box
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            return "Chưa phân loại", None

        blob = cv2.dnn.blobFromImage(
            crop,
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.ai_net.setInput(blob)
        scores = self.ai_net.forward().reshape(-1)
        probabilities = self.normalize_model_scores(scores)
        class_index = int(probabilities.argmax())
        confidence = float(probabilities[class_index] * 100)
        raw_label = (
            self.ai_labels[class_index]
            if class_index < len(self.ai_labels)
            else f"Lớp {class_index}"
        )
        label = self.map_model_label(raw_label)
        self.current_object_label = self.map_object_label(raw_label, detection_box)
        threshold = self.spinThreshold.value() * 100
        if confidence < threshold:
            self.last_model_label_detail = f"Model: {raw_label} - độ tin cậy thấp"
        else:
            self.last_model_label_detail = f"Model: {raw_label}"

        return label, confidence

    def normalize_model_scores(self, values):
        if (
            len(values) > 0
            and float(values.min()) >= 0.0
            and float(values.max()) <= 1.0
            and abs(float(values.sum()) - 1.0) < 0.08
        ):
            return values

        values = values - values.max()
        exp_values = np.exp(values)
        total = float(exp_values.sum())
        return exp_values / total if total else exp_values

    def map_model_label(self, raw_label):
        label = raw_label.strip().lower()
        recyclable = {"cardboard", "glass", "metal", "paper", "plastic"}
        organic = {"biological", "organic", "food"}

        if label in organic:
            return "Rác hữu cơ"
        if label in recyclable:
            return "Rác tái chế"
        return "Rác vô cơ"

    def map_object_label(self, raw_label, detection_box=None):
        label = raw_label.strip().lower()
        object_names = {
            "battery": "Pin",
            "biological": "Rác hữu cơ",
            "cardboard": "Hộp carton",
            "glass": "Chai thủy tinh",
            "metal": "Lon kim loại",
            "paper": "Giấy",
            "trash": "Rác vô cơ",
        }

        if label == "plastic":
            if detection_box:
                _, _, width, height = detection_box
                if height / max(1, width) >= 1.05:
                    return "Chai nước"
            return "Đồ nhựa"

        return object_names.get(label, "Vật thể")

    def auto_capture_classification(self, frame, detection_box, label, confidence):
        if confidence is None:
            return

        threshold = self.spinThreshold.value() * 100
        if confidence < threshold or label == "Không chắc chắn":
            return

        now = time.time()
        if now - self.last_auto_capture_time < 5.0:
            return

        folder = APP_DIR / "auto_captures" / self.safe_folder_name(label)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / datetime.now().strftime("classified_%Y%m%d_%H%M%S.jpg")
        caption = f"{self.current_object_label} | {label} {confidence:.1f}%"
        self.save_annotated_frame(path, frame, detection_box, caption)

        self.last_auto_capture_key = self.classification_capture_key(label, detection_box)
        self.last_auto_capture_time = now
        self.log_classification(label, confidence)
        self.refresh_stats()

    def classification_capture_key(self, label, detection_box):
        x, y, width, height = detection_box
        return (
            label,
            round(x / 45),
            round(y / 45),
            round(width / 45),
            round(height / 45),
        )

    def reset_auto_capture_if_object_left(self):
        if self.last_seen_detection_time and time.time() - self.last_seen_detection_time > 1.5:
            self.last_auto_capture_key = None

    def safe_folder_name(self, text):
        replacements = {
            "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
            "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
            "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
            "đ": "d",
            "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
            "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
            "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
            "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
            "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
            "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
            "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
            "ư": "u", "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
            "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
        }
        normalized = "".join(replacements.get(ch, ch) for ch in text.lower())
        return "".join(ch if ch.isalnum() else "_" for ch in normalized).strip("_")

    def save_annotated_frame(self, path, frame, detection_box, label):
        image = self.make_annotated_frame(frame, detection_box, label)
        image.save(str(path), quality=92)

    def make_annotated_frame(self, frame, detection_box, label):
        x, y, width, height = detection_box
        frame_height, frame_width = frame.shape[:2]
        font = self.load_pillow_font(22)
        measure_image = Image.new("RGB", (1, 1))
        measure_draw = ImageDraw.Draw(measure_image)
        text_box = measure_draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]

        padding = 34
        top_padding = 74
        needed_width = min(frame_width, max(width + padding * 2, text_width + 34))
        center_x = x + width / 2
        left = max(0, int(center_x - needed_width / 2))
        right = min(frame_width, left + needed_width)
        left = max(0, right - needed_width)
        top = max(0, y - top_padding)
        bottom = min(frame_height, y + height + padding)
        crop = frame[top:bottom, left:right].copy()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        annotated = Image.fromarray(rgb)
        draw = ImageDraw.Draw(annotated)

        rel_x = x - left
        rel_y = y - top
        draw.rectangle(
            [rel_x, rel_y, rel_x + width, rel_y + height],
            outline=(239, 68, 68),
            width=3,
        )
        self.draw_pillow_text_badge(draw, label, rel_x, rel_y, annotated.size, font)
        return annotated

    def draw_pillow_text_badge(self, draw, label, x, y, image_size, font=None):
        image_width, image_height = image_size
        if font is None:
            font = self.load_pillow_font(22)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        badge_width = min(image_width - 8, max(190, text_width + 24))
        badge_height = max(34, text_height + 14)
        badge_x = max(4, min(x, image_width - badge_width - 4))
        badge_y = y - badge_height - 5
        if badge_y < 4:
            badge_y = min(image_height - badge_height - 4, y + 5)

        draw.rectangle(
            [badge_x, badge_y, badge_x + badge_width, badge_y + badge_height],
            fill=(239, 68, 68),
        )
        draw.text(
            (badge_x + 12, badge_y + (badge_height - text_height) // 2 - 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    def load_pillow_font(self, size):
        font_paths = [
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        for path in font_paths:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()

    def draw_text_badge(self, painter, label, x, y, image_width, image_height):
        font = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        badge_height = 34
        badge_width = min(
            image_width - 8,
            max(170, metrics.horizontalAdvance(label) + 24),
        )
        badge_x = max(4, min(x, image_width - badge_width - 4))
        badge_y = y - badge_height - 5
        if badge_y < 4:
            badge_y = min(image_height - badge_height - 4, y + 5)

        painter.fillRect(
            QRect(badge_x, badge_y, badge_width, badge_height),
            QColor("#ef4444"),
        )
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(
            QRect(badge_x + 12, badge_y, badge_width - 24, badge_height),
            Qt.AlignVCenter | Qt.AlignLeft,
            label,
        )

    def run_manual_classification(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Chưa có camera", "Hãy mở camera trước khi kiểm tra nhận diện.")
            return

        detection_box = self.last_detection_box
        if not detection_box:
            self.set_classification_result("Chưa thấy vật thể", None)
            self.log("Chưa phát hiện vật thể trong khung hình.")
            return

        if self.ai_net is None:
            self.set_classification_result("Chưa có model AI", None)
            self.log("Đã phát hiện vật thể, nhưng chưa có model AI để phân loại.")
            QMessageBox.information(
                self,
                "Chưa có model AI",
                "Khung đỏ đã bám theo vật thể. Để phân loại đúng hữu cơ/vô cơ/tái chế, hãy thêm model ONNX vào models/trash_classifier.onnx.",
            )
            return

        label, confidence = self.classify_trash_region(self.current_frame, detection_box)
        self.set_classification_result(label, confidence)
        if confidence is None:
            self.log(f"Phân loại: {label}")
        else:
            self.log(f"Phân loại: {label} - {confidence:.1f}%")

    def save_snapshot(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có hình ảnh từ camera.")
            return

        folder = APP_DIR / "screenshots"
        folder.mkdir(exist_ok=True)
        path = folder / datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(str(path), self.current_frame)
        self.log(f"Đã chụp màn hình: {path}")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu ảnh:\n{path}")

    def choose_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục dataset")
        if folder:
            self.inputDatasetPath.setText(folder)

    def capture_image(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có hình ảnh từ camera.")
            return

        object_name = self.inputObjectName.text().strip()
        if not object_name:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Bạn chưa nhập tên đối tượng.")
            return

        dataset_root = Path(self.inputDatasetPath.text().strip() or "dataset")
        save_dir = dataset_root / self.comboTrashType.currentText() / object_name
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{object_name}_{datetime.now():%Y%m%d_%H%M%S}.jpg"

        cv2.imwrite(str(path), self.current_frame)
        self.image_count += 1
        self.lblImageCount.setText(str(self.image_count))
        self.log(f"Đã lưu ảnh dataset: {path}")
        QMessageBox.information(self, "Đã lưu", f"Ảnh đã lưu vào:\n{path}")

    def create_database(self):
        dataset_root = Path(self.inputDatasetPath.text().strip() or "dataset")
        classes = ["Hữu cơ", "Vô cơ", "Tái chế"]
        dataset_root.mkdir(parents=True, exist_ok=True)

        for class_name in classes:
            (dataset_root / class_name).mkdir(exist_ok=True)

        (dataset_root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
        (dataset_root / "metadata.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "classes": classes,
                },
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

        self.log("Đã tạo cơ sở dữ liệu dataset")
        self.refresh_stats()
        QMessageBox.information(self, "Hoàn tất", "Đã tạo dataset thành công.")

    def open_dataset_folder(self):
        dataset_root = Path(self.inputDatasetPath.text().strip() or "dataset")
        dataset_root.mkdir(parents=True, exist_ok=True)

        if hasattr(os, "startfile"):
            os.startfile(str(dataset_root))
        else:
            subprocess.Popen(["xdg-open", str(dataset_root)])

    def update_bin_capacity(self, value):
        if hasattr(self, "progressBinFill"):
            self.progressBinFill.setValue(value)
        self.lblFillLevel.setText(f"{value}%")

        if value >= self.bin_full_threshold:
            self.lblAlertStatus.setText("Đầy")
            self.lblAlertSub.setText("Cần thu gom rác")
            if hasattr(self, "lblBinWarning"):
                self.lblBinWarning.setText("Cảnh báo: thùng rác đã đầy")
                self.lblBinWarning.setProperty("full", True)
        else:
            self.lblAlertStatus.setText("Tốt")
            self.lblAlertSub.setText("Sức chứa ổn định")
            if hasattr(self, "lblBinWarning"):
                self.lblBinWarning.setText("Sức chứa ổn định")
                self.lblBinWarning.setProperty("full", False)

        if hasattr(self, "lblBinWarning"):
            self.lblBinWarning.style().unpolish(self.lblBinWarning)
            self.lblBinWarning.style().polish(self.lblBinWarning)

    def refresh_stats(self):
        today = datetime.now()
        today_key = today.strftime("%Y%m%d")
        capture_root = APP_DIR / "auto_captures"
        classes = ["Rác hữu cơ", "Rác vô cơ", "Rác tái chế"]
        lines = [f"Số rác trong ngày: {today:%d/%m/%Y}", "-" * 45]
        total = 0

        for class_name in classes:
            class_path = capture_root / self.safe_folder_name(class_name)
            if class_path.exists():
                count = len(list(class_path.glob(f"classified_{today_key}_*.jpg")))
            else:
                count = 0
            total += count
            lines.append(f"{class_name}: {count}")

        lines += ["-" * 45, f"Tổng cộng: {total}", f"Cập nhật: {today:%H:%M:%S}"]
        self.txtStats.setPlainText("\n".join(lines))
        self.lblImageCount.setText(str(total))

    def send_uart_test(self):
        if serial is None:
            QMessageBox.warning(
                self,
                "Thiếu thư viện",
                "Chưa cài pyserial. Hãy chạy: pip install pyserial",
            )
            return

        message = self.inputTestMessage.text().strip()
        if not message:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Bạn chưa nhập tin nhắn test.")
            return

        try:
            ser = serial.Serial(
                self.inputPort.text().strip(),
                int(self.comboBaudrate.currentText()),
                timeout=1,
            )
            ser.write(message.encode())
            ser.close()
            self.lblUartStatus.setText("Đã gửi")
            self.log(f"Đã gửi UART: {message}")
            QMessageBox.information(self, "Thành công", "Đã gửi tin nhắn UART.")
        except Exception as exc:
            self.lblUartStatus.setText("Lỗi")
            self.lblAlertStatus.setText("Lỗi")
            self.lblAlertSub.setText("Kiểm tra UART")
            self.log(f"Lỗi UART: {exc}")
            QMessageBox.critical(self, "Lỗi UART", str(exc))

    def save_config(self):
        self.config = {
            "port": self.inputPort.text().strip(),
            "baudrate": int(self.comboBaudrate.currentText()),
            "threshold": self.spinThreshold.value(),
            "epochs": self.spinEpochs.value(),
            "camera_source": self.comboCameraSource.currentText(),
            "camera_url": self.inputCameraUrl.text().strip(),
            "dataset_path": self.inputDatasetPath.text().strip(),
        }
        CONFIG_FILE.write_text(
            json.dumps(self.config, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        self.log("Đã lưu cấu hình")
        QMessageBox.information(self, "Thành công", "Đã lưu cấu hình.")

    def load_config(self):
        if CONFIG_FILE.exists():
            try:
                self.config.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
            except Exception as exc:
                self.log(f"Lỗi đọc config: {exc}")

        self.inputPort.setText(str(self.config.get("port", "COM3")))

        baud_index = self.comboBaudrate.findText(str(self.config.get("baudrate", 9600)))
        self.comboBaudrate.setCurrentIndex(baud_index if baud_index >= 0 else 0)

        self.spinThreshold.setValue(float(self.config.get("threshold", 0.5)))
        self.spinEpochs.setValue(int(self.config.get("epochs", 10)))

        camera_index = self.comboCameraSource.findText(
            self.config.get("camera_source", "USB 0")
        )
        self.comboCameraSource.setCurrentIndex(camera_index if camera_index >= 0 else 0)

        camera_url = self.config.get("camera_url", "")
        self.inputCameraUrl.setText(camera_url)
        self.inputAddCameraUrl.setText(camera_url)
        self.inputDatasetPath.setText(self.config.get("dataset_path", "dataset"))

    def clear_logs(self):
        self.txtLog.clear()
        self.txtHistory.clear()

    def log_classification(self, label, confidence):
        self.log(f"Phân loại: {label} - {confidence:.1f}%")

    def log(self, message):
        if not message.startswith("Phân loại:"):
            return
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        self.txtLog.append(line)
        self.txtHistory.append(line)

    def setup_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3f4f6;
            }
            QWidget {
                font-family: "Segoe UI";
                font-size: 14px;
                color: #1f2937;
            }
            QWidget#centralwidget,
            QStackedWidget {
                background: #f3f4f6;
            }
            QFrame#topBar {
                background: #ffffff;
                border-bottom: 1px solid #e5e7eb;
            }
            QFrame#navBar {
                background: #f9fafb;
                border-bottom: 1px solid #e5e7eb;
            }
            QLabel#lblLogo {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                color: #1e3a8a;
                font-size: 12px;
                font-weight: 900;
            }
            QLabel#lblBrandTitle {
                color: #111827;
                font-size: 20px;
                font-weight: 900;
            }
            QLabel#lblBrandSubTitle {
                color: #6b7280;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#btnPageMonitor,
            QPushButton#btnPageHistory,
            QPushButton#btnPageSystem,
            QPushButton#btnPageSettings {
                min-width: 150px;
                min-height: 38px;
                color: #374151;
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 900;
            }
            QPushButton#btnPageMonitor:hover,
            QPushButton#btnPageHistory:hover,
            QPushButton#btnPageSystem:hover,
            QPushButton#btnPageSettings:hover {
                background: #fff7ed;
                border-color: #fb923c;
                color: #9a3412;
            }
            QPushButton[active="true"] {
                background: #ea580c;
                border-color: #ea580c;
                color: #ffffff;
            }
            QLabel#lblStatusBottom,
            QLabel#lblTopStatus {
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 8px;
                color: #047857;
                font-size: 14px;
                font-weight: 900;
            }
            QLabel#lblClock {
                background: #f3f4f6;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                color: #374151;
                font-weight: 900;
                font-size: 13px;
            }
            QLabel#lblMonitorTitle,
            QLabel#lblHistoryTitle,
            QLabel#lblSystemTitle,
            QLabel#lblSettingsTitle {
                color: #111827;
                font-size: 26px;
                font-weight: 900;
            }
            QFrame#cameraPanel,
            QFrame#sensorPanel,
            QFrame#resultPanel,
            QFrame#logPanel,
            QFrame#historyPanel,
            QFrame#statsPanel,
            QFrame#datasetPanel,
            QFrame#capacityPanel,
            QFrame#settingsPanel {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
            QLabel#lblSensorTitle,
            QLabel#lblFillTitle,
            QLabel#lblLidTitle,
            QLabel#lblAiTitle,
            QLabel#lblAlertTitle,
            QLabel#lblLogTitle,
            QLabel#lblHistoryLogTitle,
            QLabel#lblStatsTitle,
            QLabel#lblCapacityTitle {
                color: #374151;
                font-size: 16px;
                font-weight: 900;
            }
            QLabel#lblBinWarning {
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 8px;
                color: #047857;
                font-weight: 900;
            }
            QLabel#lblBinWarning[full="true"] {
                background: #fef2f2;
                border: 1px solid #fecaca;
                color: #b91c1c;
            }
            QProgressBar#progressBinFill {
                min-height: 28px;
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                color: #111827;
                font-weight: 900;
                text-align: center;
            }
            QProgressBar#progressBinFill::chunk {
                background: #ea580c;
                border-radius: 8px;
            }
            QLabel#lblFillLevel,
            QLabel#lblLidStatus,
            QLabel#lblAiClass,
            QLabel#lblAlertStatus {
                color: #0f766e;
                font-size: 17px;
                font-weight: 900;
            }
            QLabel#lblAlertSub {
                color: #6b7280;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#lblCameraTitle,
            QLabel#lblResultTitle {
                color: #111827;
                font-size: 18px;
                font-weight: 900;
            }
            QLabel#lblCameraMonitor,
            QLabel#lblPreviewCamera,
            QLabel#lblAddCamera {
                background: #111827;
                color: #f9fafb;
                border: 1px solid #374151;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 900;
            }
            QLabel#lblResult {
                background: #f9fafb;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                color: #111827;
                font-size: 22px;
                font-weight: 900;
            }
            QLabel#lblTrashNow,
            QLabel#lblConfidence,
            QLabel#lblImageCount,
            QLabel#lblUartStatus {
                color: #ea580c;
                font-weight: 900;
            }
            QLineEdit,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox {
                min-height: 38px;
                background: #f9fafb;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding-left: 10px;
            }
            QLineEdit:focus,
            QComboBox:focus,
            QSpinBox:focus,
            QDoubleSpinBox:focus {
                border: 1px solid #ea580c;
                background: #ffffff;
            }
            QPushButton {
                min-height: 40px;
                padding: 0 16px;
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                color: #1f2937;
                font-weight: 800;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
            QPushButton#btnStartCamera,
            QPushButton#btnStartAddCamera,
            QPushButton#btnPreviewCamera,
            QPushButton#btnSendUart,
            QPushButton#btnSaveConfig {
                background: #047857;
                border: none;
                color: #ffffff;
            }
            QPushButton#btnStopCamera,
            QPushButton#btnClosePreview {
                background: #b91c1c;
                border: none;
                color: #ffffff;
            }
            QPushButton#btnSnapshot,
            QPushButton#btnTestDetect,
            QPushButton#btnCapture,
            QPushButton#btnCreateDatabase,
            QPushButton#btnOpenDataset,
            QPushButton#btnChooseFolder,
            QPushButton#btnRefreshStats {
                background: #ea580c;
                border: none;
                color: #ffffff;
            }
            QPushButton#btnClearLog {
                background: #f8fafc;
                color: #334155;
            }
            QTextEdit#txtLog,
            QTextEdit#txtHistory,
            QTextEdit#txtStats {
                background: #111827;
                color: #e5e7eb;
                border: 1px solid #374151;
                border-radius: 8px;
                padding: 10px;
                font-family: Consolas;
                font-size: 13px;
            }
            """
        )

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = SmartTrashApp()
    window.show()
    sys.exit(app.exec_())
