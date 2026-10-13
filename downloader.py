import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "caffemodels")

# Diccionario de archivos y sus URLs candidatas
FILES = {
    "age_deploy.prototxt": [
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt",
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/age_deploy.prototxt"
    ],
    "gender_deploy.prototxt": [
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_deploy.prototxt"
    ],
    "age_net.caffemodel": [
        "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_net.caffemodel"
    ],
    "gender_net.caffemodel": [
        "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel",
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel"
    ]
}

def download_file(filename, urls):
    dest_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        print(f"[OK] {filename} ya existe y tiene un tamaño valido.")
        return True

    print(f"[...] Descargando {filename}...")
    for url in urls:
        try:
            print(f"Intentando desde: {url}")
            # Agregar cabecera User-Agent para evitar bloqueos
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                # Descargar en bloques con reporte de progreso
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 256  # 256 KB
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgreso: {percent:.2f}% ({downloaded // 1024} KB / {total_size // 1024} KB)", end="")
                    else:
                        print(f"\rDescargado: {downloaded // 1024} KB", end="")
                print()
            print(f"[OK] Descargado exitosamente: {filename}")
            return True
        except Exception as e:
            print(f"\n[ERROR] Error al descargar de {url}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
    return False

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Creado directorio: {MODELS_DIR}")

    success = True
    for filename, urls in FILES.items():
        if not download_file(filename, urls):
            print(f"[ERROR CRITICO] No se pudo descargar {filename}")
            success = False
            
    if success:
        print("\n[OK] ¡Todos los modelos se han configurado correctamente!")
    else:
        print("\n[ERROR] Algunos modelos no pudieron ser descargados. Por favor revisa tu conexion de red o las URLs.")

if __name__ == "__main__":
    main()
