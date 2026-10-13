import json
import time
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tracker.camera import VideoCamera

# Instancia singleton de la camara
camera = VideoCamera()

def dashboard_view(request):
    """
    Vista principal que renderiza el dashboard de control y visualizacion.
    """
    # Si la camara no esta corriendo, iniciarla por defecto con webcam (0)
    if not camera.running:
        camera.start(source="0")
        
    return render(request, "tracker/dashboard.html", {
        "source": camera.source,
        "line_y_ratio": camera.line_y_ratio
    })

def gen_frames():
    """
    Generador de frames para el streaming de video MJPEG.
    """
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            # Frame vacio temporal (gris) en caso de que la camara este cargando
            img = bytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')
        time.sleep(0.015)  # Latencia ultra-baja y fluidez máxima

def video_feed(request):
    """
    Endpoint que sirve el streaming en tiempo real.
    """
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def stats_view(request):
    """
    Retorna las estadisticas actuales en formato JSON.
    """
    stats = camera.get_stats()
    return JsonResponse(stats)

@csrf_exempt
def update_config(request):
    """
    Permite actualizar la posicion de la linea, la fuente de la camara o reiniciar contadores.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            
            # 1. Actualizar posición o vector de la línea trazada (P1 y P2)
            if "line_p1" in data and "line_p2" in data:
                camera.set_line_points(data["line_p1"], data["line_p2"])
            elif "line_y_ratio" in data:
                camera.set_line_position(data["line_y_ratio"])
                
            # 2. Actualizar fuente de video (IP o webcam)
            if "source" in data:
                new_source = str(data["source"]).strip()
                if new_source:
                    camera.start(source=new_source)
                    
            # 3. Reiniciar contadores
            if data.get("action") == "reset":
                camera.reset_stats()
                
            return JsonResponse({"status": "success", "message": "Configuracion actualizada exitosamente"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
            
    return JsonResponse({"status": "error", "message": "Metodo no soportado"}, status=405)
