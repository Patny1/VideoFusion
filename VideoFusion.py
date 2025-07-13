"""
Video Fusion Tool with Polygonal Masking
-----------------------------------------------------

Autor: Patny
Fecha: 13 de julio
Repositorio: https://github.com/Patny1/VideoFusion

Descripción:
------------
Este programa permite fusionar en tiempo real múltiples fuentes de video (cámara y archivo .mp4)
dentro de zonas poligonales definidas manualmente sobre una imagen de fondo (marco).

Los usuarios pueden seleccionar dos regiones poligonales interactivamente, una para cada fuente,
y visualizar los resultados de forma comparativa:

- Ventana "Original Video": inserta los videos sin aplicar procesamiento
- Ventana "Video Fusion": muestra los videos insertados tras aplicar procesamiento

Controles:
----------
- 'a': Seleccionar/Redefinir región para la cámara.
- 's': Seleccionar/Redefinir región para el video.
- Click izquierdo: Añadir puntos al polígono seleccionado.
- 'c': Cerrar el polígono activo.
- 'Enter': Iniciar la reproducción de cámara y video.
- 'Esc': Salir del programa.

Dependencias:
-------------
- OpenCV (cv2)
- NumPy

Requisitos:
-----------
- Python 3.12 (creado en esta versión)
- Cámara web funcional (opcional)
- Archivo de video (formato .mp4 )
- Imagen de fondo (JPG o PNG )

Licencia:
---------
Este software se distribuye bajo la Licencia MIT

"""

import cv2
import numpy as np

# Archivos
video_path = 'noche.mp4'#'gatto_nero2.mp4'
background_path = 'marco1.jpg'

# Variables globales
polygons = {'a': [], 's': []}
polygon_closed = {'a': False, 's': False}
current_key = None
#video_capture = None
cam_capture = None
playing = False

# Cargar imagen de fondo
background = cv2.imread(background_path)
if background is None:
    raise Exception("No se pudo cargar la imagen 'Cielo.jpg'")
background = cv2.resize(background, (800, 500))  # Escalar si es necesario

#abre video
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print(" Error al abrir el video.")
    raise Exception("No se pudo cargar el video")





#FUNCIONES
#dIBUJAR POLIGONO
def draw_polygon(image, points, closed, color=(0, 255, 0)):
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], color, 2)
        if closed and len(points) >= 3:
            cv2.line(image, points[-1], points[0], color, 2)
    for p in points:
        cv2.circle(image, p, 5, color, -1)

#morfológica
def new_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    #mask = cv2.erode(mask, kernel, iterations=10)
    #mask = cv2.dilate(mask, kernel, iterations=10)
    return mask


#FUSIONAR VIDEO
def warp_video_to_polygon(frame, dest_points, image, is_original=False):
    if len(dest_points) < 3:
        return image

    bbox = cv2.boundingRect(np.array(dest_points))
    x, y, w, h = bbox
    resized_frame = cv2.resize(frame, (w, h))

    mask = np.zeros((h, w, 3), dtype=np.uint8)
    polygon = np.array([[px - x, py - y] for (px, py) in dest_points], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    mask = new_mask(mask)
    roi = image[y:y + h, x:x + w]

    if is_original:
        # Fusión sin efectos (bitwise)
        masked_frame = cv2.bitwise_and(resized_frame, mask)
        mask_inv = cv2.bitwise_not(mask)
        background_part = cv2.bitwise_and(roi, mask_inv)
        combined = cv2.add(background_part, masked_frame)
        image[y:y+h, x:x+w] = combined
    else:
        # Fusión con blending
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        gray_mask = cv2.GaussianBlur(gray_mask, (15, 15), 0)
        normalized_mask = gray_mask.astype(float) / 255.0
        alpha = normalized_mask[..., np.newaxis]
        blended = (resized_frame.astype(float) * alpha + roi.astype(float) * (1 - alpha)).astype(np.uint8)
        image[y:y + h, x:x + w] = blended

    return image


def process_frame(frame):
    #frame=equalize_hist_color(frame)

    #frame= cv2.GaussianBlur(frame, (5, 5), 0)  #filtro gausiano evita dientes de sierra, efecto de suavizado

    # Ajuste manual de brillo y contraste
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=20)  # alpha=contraste, beta=brillo


    #frame=cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)#suavizado

   # frame=clahe_hist(frame)

   # frame = cv2.bitwise_not(frame)

    return frame

#CAPTURAR MAUSE
def mouse_callback(event, x, y, flags, param):
    global polygons, current_key, polygon_closed
    if event == cv2.EVENT_LBUTTONDOWN and current_key in ['a', 's'] and not polygon_closed[current_key]:
        polygons[current_key].append((x, y))


#equalizar
def equalize_hist_color(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    y_eq = cv2.equalizeHist(y)  # Solo ecualizas la luminancia
    img_eq = cv2.merge((y_eq, cr, cb))


   # img_eq=cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
    #img_eq[:,:,1] = cv2.equalizeHist(img_eq[:,:,1])
    #img_eq[:,:,2] = cv2.equalizeHist(img_eq[:,:,2])

    return cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2BGR)
#equalizar clahe
def clahe_hist(img):

    # clipLimit defines threshold to limit the contrast in case of noise in our image
    # tileGridSize defines the area size in which local equalization will be performed
    clahe_model = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # For ease of understanding, we explicitly equalize each channel individually
    img_b = clahe_model.apply(img[:, :, 0])
    img_g = clahe_model.apply(img[:, :, 1])
    img_r = clahe_model.apply(img[:, :, 2])

    # Next we stack our equalized channels back into a single image
    img_clahe = np.stack((img_b,img_g,img_r), axis=2)



    # Using Numpy to calculate the histogram
   # color = ('b', 'g', 'r')
    #for i, col in enumerate(color):
     #   histr, _ = np.histogram(img_clahe[:, :, i], 256, [0, 256])
      #  plt.plot(histr, color=col)
       # plt.xlim([0, 256])
    #plt.show()

    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #l, a, b = cv2.split(lab)

    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #cl = clahe.apply(l)

    #img_clahe = cv2.merge((cl, a, b))
    #img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)


    return img_clahe



cv2.namedWindow("Video Fusion")
cv2.setMouseCallback("Video Fusion", mouse_callback)

cv2.namedWindow("Original Video")

print("Presiona 'a' o 's' para seleccionar un polígono.")
print("Clic para agregar puntos.")
print("Presiona 'c' para cerrar el polígono.")
#print("Presiona 'z' para cargar video.")
print("Presiona 'Enter' para iniciar reproducción.")
print("Presiona 'ESC' para salir.")

while True:
    display = background.copy()
    original=background.copy()

    # Dibujar polígonos
    for key in ['a', 's']:
        color = (20, 255, 20) if key == 'a' else (255, 20, 20)
        draw_polygon(display, polygons[key], polygon_closed[key], color)

    # Mostrar video/cámara si están activos
    if playing:
        if cam_capture and cam_capture.isOpened():
            ret_cam, frame_cam = cam_capture.read()
            if ret_cam and polygon_closed['a']:
                frame_cam_original=frame_cam.copy()
               # original = warp_video_to_polygon(frame_cam_original, polygons['a'], original)

                # Sin procesamiento (bitwise solo)
                original = warp_video_to_polygon(frame_cam_original, polygons['a'], original, is_original=True)

                frame_cam=process_frame(frame_cam)
                #display = warp_video_to_polygon(frame_cam, polygons['a'], display)

                # Con filtros y blending
                display = warp_video_to_polygon(frame_cam, polygons['a'], display, is_original=False)


                #cv2.imshow("Processed Frame", frame_cam)
                #cv2.imshow("Original Frame", frame_cam_original)


        if video_capture and video_capture.isOpened():
            ret_vid, frame_vid = video_capture.read()
            if not ret_vid:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_vid, frame_vid = video_capture.read()
            if ret_vid and polygon_closed['s']:
                frame_vid_original=frame_vid.copy()
                #original = warp_video_to_polygon(frame_vid_original, polygons['s'], original)
                original = warp_video_to_polygon(frame_vid_original, polygons['s'], original, is_original=True)

                frame_vid=process_frame(frame_vid)
                #display = warp_video_to_polygon(frame_vid, polygons['s'], display)
                display = warp_video_to_polygon(frame_vid, polygons['s'], display, is_original=False)
                #cv2.imshow("Processed Frame",frame_vid)
                #cv2.imshow("Original Frame", frame_vid_original)

    cv2.imshow("Video Fusion", display)
    cv2.imshow("Original Video", original)



    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('a'):
        polygons['a'] = []
        polygon_closed['a'] = False
        current_key = 'a'
        print("Redefiniendo polígono 'a'")
    elif key == ord('s'):
        polygons['s'] = []
        polygon_closed['s'] = False
        current_key = 's'
        print("Redefiniendo polígono 's'")


    elif key == ord('c') and current_key in ['a', 's']:
        if len(polygons[current_key]) >= 3:
            polygon_closed[current_key] = True
            print(f" Polígono '{current_key}' cerrado.")

    elif key == 13:  # Enter
        if not cam_capture:
            cam_capture = cv2.VideoCapture(0)

            if not cam_capture.isOpened():
                print("No se pudo abrir la cámara.")
                cam_capture = None
            else:
                print("Cámara iniciada.")
        if video_capture and video_capture.isOpened():

            playing = True
            print(" Reproducción iniciada.")

cv2.destroyAllWindows()
if video_capture:
    video_capture.release()
if cam_capture:
    cam_capture.release()
print("Application closed.")