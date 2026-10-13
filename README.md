# Sistema de Conteo y Analíticas de Personas en Tiempo Real (Django + YOLOv11)

Este proyecto es una solución completa basada en web que integra **YOLO11** de Ultralytics con clasificadores profundos de edad y género de OpenCV para monitorear flujos de video en tiempo real (cámaras web, archivos de video o flujos de red RTSP de cámaras IP Hikvision) y realizar analíticas demográficas de paso de peatones.

---

## 🚀 Funcionamiento General

El sistema opera bajo los siguientes pilares de arquitectura:

1. **Captura Asíncrona y Fluidez Ultra-Baja Latencia (Canales 101 y 102)**:
   La clase `VideoCamera` en `tracker/camera.py` ejecuta la captura de frames en un hilo de fondo (`threading.Thread`) separado de Django. 
   - **Fluidez en Canal 101 (Main Stream 1080p/4K)**: Incorpora vaciado continuo de búfer RTSP sobre TCP (`rtsp_transport;tcp|fflags;nobuffer|max_delay;500000`) y redimensionamiento adaptativo en el hilo de captura a 1024px. Esto elimina por completo congelamientos de video, cuadros grises y demoras acumuladas.
   - **Reconexión Automática**: Maneja reconexiones transparentes si la señal de la cámara IP se interrumpe temporalmente.

2. **Detección y Rastreo (YOLOv11)**:
   Utiliza el modelo optimizado **YOLO11 nano (`yolo11n.pt`)** ejecutado a `imgsz=640` para detectar personas con máxima agilidad. Para seguir a cada peatón frame a frame se utiliza el algoritmo de seguimiento integrado (`model.track(..., persist=True)`), asignando una ID única a cada individuo.

3. **Cruce de Línea Vectorial 2D (Ingresos/Salidas)**:
   Se evalúa la trayectoria de cada persona contra la regla trazada mediante el producto cruz vectorial 2D ($\vec{V} \times \vec{P}$), lo que permite contar pasos a través de reglas en cualquier ángulo o inclinación:
   * **Ingresos (Entradas)**: Ocurre cuando el centroide cruza la línea hacia **ABAJO / DERECHA** del vector trazado (↓).
   * **Salidas**: Ocurre cuando el centroide cruza la línea hacia **ARRIBA / IZQUIERDA** del vector trazado (↑).

4. **Trazado de Línea Estilo PowerPoint (P1 a P2)**:
   - **Botón `✏️ Trazar Línea en Vivo`**: Permite al usuario hacer clic y arrastrar libremente sobre la imagen en vivo del reproductor de video para dibujar cualquier segmento de línea ($P_1$ a $P_2$) o ajustar sus extremos (handles amarillo y rojo) directamente en pantalla.
   - **Control Deslizante**: La altura de la regla también se puede ajustar desde el panel lateral derecho.
   - La posición e inclinación se guarda e impacta de inmediato en el servidor Django sin interrumpir el monitoreo.

5. **Clasificación Demográfica (Edad y Género)**:
   * Cuando se detecta un peatón, el sistema recorta la zona superior del recuadro (cabeza estimada).
   * Se ejecuta un detector de rostros **Haar Cascade** de OpenCV.
   * Si encuentra un rostro, el fragmento se envía a dos redes neuronales convolucionales Caffe para predecir el **género** (Hombre/Mujer) y la **edad**.
   * **Consolidación por Rangos Amplios (Sin Vacilación / "Patinar")**: Para evitar que la estimación vacile entre clases estrechas, el sistema suma las probabilidades en 4 rangos amplios: **Niño (0-12)**, **Joven (13-24)**, **Adulto (25-54)** y **Adulto Mayor (55+)**.
   * **Optimización clave (Caché por ID)**: Para no degradar la tasa de frames, el análisis facial se corre una sola vez por persona al ser detectada. La predicción se guarda en caché y se mantiene vinculada a la ID durante todo su recorrido.

6. **Visualización y Dashboard**:
   El frontend está diseñado en modo oscuro con estética **glassmorphic**. El video procesado se sirve mediante una respuesta de streaming HTTP multipart (`/video_feed/`). En paralelo, un script de JavaScript consulta un endpoint de datos (`/stats/`) cada 1 segundo para refrescar contadores, barras porcentuales y rangos de edad de forma dinámica sin recargar la página.

---

## 🛠️ Cómo Levantarlo en una Laptop Nueva

Sigue estos pasos detallados para configurar y ejecutar el proyecto desde cero:

### 1. Clonar / Descargar el Proyecto
Copia la carpeta del proyecto `camara-project` en el disco local de la nueva laptop.

### 2. Crear y Activar un Entorno Virtual (Recomendado)
Abre la terminal de comandos (PowerShell o CMD) en la carpeta raíz del proyecto y ejecuta:

```powershell
# Crear el entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activar en Windows (CMD)
.\venv\Scripts\activate.bat

# Activar en Linux/macOS
source venv/bin/activate
```

### 3. Instalar Dependencias
Instala los paquetes de Python especificados en el archivo de requerimientos:
```powershell
pip install -r requirements.txt
```

### 4. Descargar Modelos de Edad y Género
Para obtener los archivos `.caffemodel` y `.prototxt` de las redes de clasificación demográfica de forma automática, ejecuta:
```powershell
python downloader.py
```
*(Este script creará la carpeta `caffemodels/` en la raíz y descargará los pesos de repositorios públicos seguros).*

### 5. Aplicar Migraciones de Django
Ejecuta las migraciones internas del framework para establecer la base de datos local ligera:
```powershell
python manage.py migrate
```

### 6. Iniciar el Servidor de Desarrollo
Lanza la aplicación web local:
```powershell
python manage.py runserver
```

### 7. Acceder a la Interfaz
Abre tu navegador de internet e ingresa a:
👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

## 📷 Conectar la Cámara Hikvision DS-2CD1123G0E-I

Para conectar tu cámara de red real en tiempo real:

1. Ve a la sección **Configuración de Cámara y Regla** en la parte inferior derecha de la interfaz web.
2. En el campo **Origen del Streaming**, ingresa la URL RTSP utilizando la IP de tu cámara (`192.168.1.80`) y tus credenciales:

   * **Canal Primario (101 - Máxima Resolución)**:
     ```text
     rtsp://admin:miclave123@192.168.1.80:554/Streaming/Channels/101
     ```
   * **Canal Secundario (102 - Sub-Stream)**:
     ```text
     rtsp://admin:miclave123@192.168.1.80:554/Streaming/Channels/102
     ```
3. Presiona **Aplicar Configuración**. El video se refrescará para capturar el feed de la red.

> [!NOTE]
> **Optimización Adaptativa:** Gracias al escalado en tiempo real e inferencia ligera con YOLO11, ahora puedes transmitir tanto el canal **101** como el **102** con movimiento fluido y sin lag en la transmisión.

---

## ⚠️ Solución de Problemas (Troubleshooting)

### Advertencia: `can't grab frame. Error: -1072875772` (MSMF)
Si por defecto (fuente `"0"`) la terminal de Django muestra alertas de OpenCV advirtiendo que no puede capturar frames, esto ocurre porque:
1. La computadora no cuenta con una cámara web integrada física.
2. La cámara web está siendo utilizada por otra aplicación en segundo plano (ej. Zoom, Teams, Chrome).

**Solución Offline sin Webcam:**
Puedes simular y probar todo el sistema de rastreo usando un **archivo de video grabado** (`.mp4`, `.avi`, etc.) de personas caminando:
1. Consigue un video de muestra y guárdalo en tu laptop (ejemplo: `C:\videos\personas.mp4`).
2. Ingresa la ruta completa del archivo en el campo **Origen del Streaming** de la interfaz: `C:/videos/personas.mp4`.
3. Haz clic en **Aplicar Configuración**. El sistema reproducirá el video en bucle infinito permitiéndote probar la regla de conteo e inferencias demográficas sin necesidad de cámara física.
