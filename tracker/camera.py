import cv2
import numpy as np
import os

# Forzar el transporte RTSP sobre TCP en FFmpeg y deshabilitar buffer para máxima fluidez y cero lag en canales 101
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"

import threading
import time
from ultralytics import YOLO

# Rutas de modelos de Edad y Género
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "caffemodels")

AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")

# Parámetros para Clasificación
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# Rangos amplios consolidados para evitar fluctuaciones / ruido en estimación de edad
AGE_GROUPS = ['Niño (0-12)', 'Joven (13-24)', 'Adulto (25-54)', 'Adulto Mayor (55+)']
GENDER_LIST = ['Male', 'Female']

class VideoCamera:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VideoCamera, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.source = "0"  # Default a webcam local
        self.cap = None
        self.running = False
        self.thread = None          # Hilo para capturar frames continuamente
        self.process_thread = None  # Hilo para procesar con YOLO a su propio ritmo
        self.frame_lock = threading.Lock()
        self.raw_frame_lock = threading.Lock()
        self.latest_frame = None
        self.raw_frame = None       # El frame mas reciente leido de la camara
        
        # Estadísticas de Conteo
        self.total_in = 0
        self.total_out = 0
        self.gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}
        self.age_counts = {group: 0 for group in AGE_GROUPS}
        self.age_counts["Unknown"] = 0
        
        # Historial y estado de cruce de línea (soporta líneas en cualquier dirección/ángulo P1 a P2)
        self.line_p1 = [0.0, 0.5]  # P1 [x1, y1] en escala 0.0 a 1.0
        self.line_p2 = [1.0, 0.5]  # P2 [x2, y2] en escala 0.0 a 1.0
        self.crossed_tracker = {}  # track_id -> 'side1' / 'side2'
        self.person_cache = {}  # track_id -> {'gender': str, 'age': str}
        
        # Inicialización de Modelos
        print("[...] Cargando YOLOv11...")
        self.yolo_model = YOLO("yolo11n.pt")
        print("[OK] YOLOv11 cargado.")
        
        # Cargar clasificadores OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        try:
            print("[...] Cargando modelos de Edad y Género...")
            self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
            self.gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
            print("[OK] Modelos de Edad y Género cargados.")
            self.models_loaded = True
        except Exception as e:
            print(f"[ERROR] Error al cargar modelos de Edad/Género (no descargados?): {e}")
            self.models_loaded = False
            
        self._initialized = True

    @property
    def line_y_ratio(self):
        """Devuelve la posición vertical normalizada del primer punto de la línea.

        Esta propiedad mantiene compatibilidad con ``tracker/views.py``,
        donde se consulta ``camera.line_y_ratio``.
        """
        with self.frame_lock:
            return self.line_p1[1]

    def start(self, source=None):
        with self.frame_lock:
            if source is not None:
                self.source = source
            
            if self.running:
                self.stop()
                
            self.running = True
            
            # Hilo 1: Solo captura frames y vacia el buffer
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            # Hilo 2: Procesa los frames capturados a su propio ritmo sin lag acumulado
            self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.process_thread.start()
            
            print(f"[OK] Hilos de captura y procesamiento iniciados con fuente: {self.source}")

    def stop(self):
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
            self.process_thread = None
            
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[OK] Monitoreo detenido.")

    def reset_stats(self):
        with self.frame_lock:
            self.total_in = 0
            self.total_out = 0
            self.gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}
            self.age_counts = {group: 0 for group in AGE_GROUPS}
            self.age_counts["Unknown"] = 0
            self.crossed_tracker.clear()
            self.person_cache.clear()
            print("[OK] Estadisticas reiniciadas.")

    def set_line_position(self, ratio):
        with self.frame_lock:
            r = max(0.01, min(0.99, float(ratio)))
            self.line_p1 = [0.0, r]
            self.line_p2 = [1.0, r]
            print(f"[OK] Posicion de linea horizontal actualizada a: {r}")

    def set_line_points(self, p1, p2):
        with self.frame_lock:
            x1 = max(0.0, min(1.0, float(p1[0])))
            y1 = max(0.0, min(1.0, float(p1[1])))
            x2 = max(0.0, min(1.0, float(p2[0])))
            y2 = max(0.0, min(1.0, float(p2[1])))
            self.line_p1 = [x1, y1]
            self.line_p2 = [x2, y2]
            print(f"[OK] Segmento de linea trazado actualizado: P1={self.line_p1}, P2={self.line_p2}")

    def _predict_age_gender(self, face_img):
        if not self.models_loaded:
            return "Unknown", "Unknown"
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predecir Género
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            
            # Predecir Edad y acumular probabilidades por rangos amplios para máxima estabilidad
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()[0]
            
            group_probs = {
                'Niño (0-12)': float(age_preds[0] + age_preds[1] + age_preds[2]),
                'Joven (13-24)': float(age_preds[3]),
                'Adulto (25-54)': float(age_preds[4] + age_preds[5] + age_preds[6]),
                'Adulto Mayor (55+)': float(age_preds[7])
            }
            age_group = max(group_probs, key=group_probs.get)
            
            return gender, age_group
        except Exception as e:
            print(f"Error en inferencia de edad/género: {e}")
            return "Unknown", "Unknown"

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        
        # Coordenadas absolutas de la línea trazada en el fotograma
        p1_x, p1_y = int(self.line_p1[0] * w), int(self.line_p1[1] * h)
        p2_x, p2_y = int(self.line_p2[0] * w), int(self.line_p2[1] * h)
        
        dx = p2_x - p1_x
        dy = p2_y - p1_y
        
        # 1. Ejecutar rastreo de YOLOv11
        results = self.yolo_model.track(frame, persist=True, classes=[0], imgsz=640, verbose=False)
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                gender, age = "Unknown", "Unknown"
                
                # 2. Analizar rostro para Edad y Género si no está cacheado
                if track_id in self.person_cache:
                    gender = self.person_cache[track_id]["gender"]
                    age = self.person_cache[track_id]["age"]
                else:
                    person_h = y2 - y1
                    head_y2 = y1 + int(person_h * 0.35)
                    head_crop = frame[max(0, y1):min(h, head_y2), max(0, x1):min(w, x2)]
                    
                    if head_crop.size > 0:
                        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                        if len(faces) > 0:
                            fx, fy, fw, fh = faces[0]
                            face_img = head_crop[fy:fy+fh, fx:fx+fw]
                            if face_img.size > 0:
                                gender, age = self._predict_age_gender(face_img)
                                self.person_cache[track_id] = {"gender": gender, "age": age}
                
                # 3. Control de cruce usando Producto Cruz Vectorial 2D:
                # val > 0 indica que el punto está del Lado 2 (debajo/derecha del vector trazado)
                # val <= 0 indica que el punto está del Lado 1 (arriba/izquierda del vector trazado)
                val = dx * (cy - p1_y) - dy * (cx - p1_x)
                current_side = 'side2' if val > 0 else 'side1'
                
                if track_id not in self.crossed_tracker:
                    self.crossed_tracker[track_id] = current_side
                else:
                    prev_side = self.crossed_tracker[track_id]
                    if prev_side == 'side1' and current_side == 'side2':
                        # Cruzó de Lado 1 a Lado 2 -> INGRESO (total_in)
                        self.total_in += 1
                        self.crossed_tracker[track_id] = 'side2'
                        self.gender_counts[gender] = self.gender_counts.get(gender, 0) + 1
                        self.age_counts[age] = self.age_counts.get(age, 0) + 1
                    elif prev_side == 'side2' and current_side == 'side1':
                        # Cruzó de Lado 2 a Lado 1 -> SALIDA (total_out)
                        self.total_out += 1
                        self.crossed_tracker[track_id] = 'side1'
                        self.gender_counts[gender] = self.gender_counts.get(gender, 0) + 1
                        self.age_counts[age] = self.age_counts.get(age, 0) + 1

                # 4. Dibujar recuadros y etiquetas
                color = (0, 255, 0) if gender == 'Female' else (255, 0, 0) if gender == 'Male' else (200, 200, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{track_id} | {gender} | {age}"
                cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        
        # 5. Dibujar línea trazada por el usuario (roja con puntos amarillos en los extremos)
        cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 3)
        cv2.circle(frame, (p1_x, p1_y), 6, (0, 255, 255), -1)
        cv2.circle(frame, (p2_x, p2_y), 6, (0, 255, 255), -1)
        
        mid_x = (p1_x + p2_x) // 2
        mid_y = (p1_y + p2_y) // 2
        cv2.putText(frame, "LINEA DE CONTEO [v Ingresos | ^ Salidas]", (max(10, mid_x - 140), max(20, mid_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def _capture_loop(self):
        # Intentar conectar a la fuente
        src = self.source
        if src.isdigit():
            src = int(src)
            
        self.cap = cv2.VideoCapture(src)
        
        # Configurar buffer pequeño para baja latencia si es stream de red
        if isinstance(src, str) and src.startswith("rtsp://"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        consecutive_failures = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                time.sleep(0.2)
                if consecutive_failures > 15:
                    print("[!] Intentando reconectar camara...")
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(src)
                    if isinstance(src, str) and src.startswith("rtsp://"):
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_failures = 0
                continue
                
            consecutive_failures = 0
            
            # Redimensionamiento adaptativo para fluidez extrema en canal 101 (1080p / 4K)
            h, w = frame.shape[:2]
            if w > 1024:
                new_w = 1024
                new_h = int(h * (1024.0 / w))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Guardar el frame mas nuevo en una variable intermedia limpia de forma segura
            with self.raw_frame_lock:
                self.raw_frame = frame
                
            # No poner pausas largas aqui; este hilo debe correr tan rapido como la camara entregue frames
            time.sleep(0.002)
            
        if self.cap:
            self.cap.release()
            self.cap = None

    def _processing_loop(self):
        while self.running:
            frame = None
            
            # Extraer el frame mas nuevo guardado en el buffer intermedio
            with self.raw_frame_lock:
                if self.raw_frame is not None:
                    frame = self.raw_frame.copy()
                    # Limpiamos para esperar al siguiente frame fresco y no re-procesar el mismo
                    self.raw_frame = None
            
            if frame is not None:
                # Procesar YOLO + Edad/Género
                processed_frame = self._process_frame(frame)
                
                # Codificar a JPEG
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
                if ret:
                    with self.frame_lock:
                        self.latest_frame = jpeg.tobytes()
            else:
                # Si no hay frame nuevo en espera, descansar un poco para no consumir CPU en vano
                time.sleep(0.01)
                
            # Pequeño retardo de ciclo
            time.sleep(0.005)

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame

    def get_stats(self):
        with self.frame_lock:
            return {
                "total_in": self.total_in,
                "total_out": self.total_out,
                "gender_counts": self.gender_counts,
                "age_counts": self.age_counts,
                "line_p1": self.line_p1,
                "line_p2": self.line_p2,
                "line_y_ratio": self.line_p1[1],
                "source": self.source,
                "is_running": self.running
            }
