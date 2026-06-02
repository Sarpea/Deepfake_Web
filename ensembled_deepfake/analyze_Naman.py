import cv2
import torch
import numpy as np

from .deep_fake_detection.processor_deepfake import DeepFakeProcessor #hay que tener en cuenta que el modelo usa face-recognition, pero se ha eliminado del modelo la secuencia que lo requiere (puede cambiar el comportamiento)
from .deep_fake_detection.modeling import load_model

def analyze_video_Naman(video_path: str, interval: int = 3, ia_threshold: float = 0.75, device='cpu'):
    
    """
    Lógica de decisión: si tan solo un clip ha sido clasificado como IA, se considera que todo el video es engañoso 
    interval: cantidad de frames saltados por cada frame analizado
    ia_threshold: porcentaje minimo para clasificar el clip como IA
    device: cpu o cuda, para poder ejecutar el modelo usando cpu o gpu
    """

    # Inicializar el preprocesador y modelo
    processor = DeepFakeProcessor()
    model = load_model().to(device) # como recordatorio, el output es (real/fake)
    model.eval()

    # VARIABLES IMPORTANTES PARA EL MODELO
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_length = 20
    clip_amount = int(total_frames // (clip_length * interval))
    if total_frames % (clip_length * interval) > (clip_length * interval) // 2:
        clip_amount += 1
    clip_frames = []
    clip_count = 0
    is_ia = False
    ia_clips = 0
    frame_cont = 0
    predictions = [] # informacion especifica de cada clip (util para almacenar en una mongo)
    ia_predictions = [] # nivel de confianza de IA


    # PROCESANDO VIDEO
    # El bucle a continuacion extrae bloques de 20 frames mediante read -> append, luego los procesa, guarda los resultados y empieza de nuevo hasta que el video termine.
    while True:
        ret, frame = cap.read()
        if not ret:  
            break      
        # Convertir BGR -> RGB
        frame_cont += 1
        if frame_cont % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_frames.append(frame_rgb)


        # Se crean clips de 20 frames y se procesa cada clip por separado
            if len(clip_frames) == clip_length:   
                print(f"Processing clip {clip_count + 1}/{clip_amount}")
                probs, pred_idx, confidence = process_video_naman(clip_frames=clip_frames, processor=processor, model=model, device=device)
                    
                # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
                ia_prob = probs[1]
                if ia_prob >= 0.5: 
                    if ia_prob > ia_threshold:
                        is_ia = True
                    ia_clips += 1
                    ia_predictions.append(probs[1]) # nos servirá para calcular la confianza de la prediccion más adelante
                

            # Calcular segundo de inicio y fin del clip
                clip_start = round((frame_cont-clip_length * interval) / fps, 2)
                clip_end = round(frame_cont / fps, 2)


            # Almacenar resultados en una lista aparte para los clips
                predictions.append({
                    "clip_num": clip_count+1,
                    "clip_period": (clip_start, clip_end),
                    "prediction": "IA" if pred_idx else "REAL",
                    "confidence": round(confidence, 2)
                })
                clip_count += 1
                clip_frames = []
    cap.release()

    # Procesar ultimo clip del video (Este fragmento es igual que el del bucle, sirve para procesar el ultimo clip)
    # Hay que tener en cuenta que los tiempos de este clip varian con respecto a los otros
    if (clip_length // 2) < len(clip_frames) < clip_length:
        clip_start = round((frame_cont-(len(clip_frames) * interval)) / fps, 2)
        clip_end = round(frame_cont / fps, 2)
        while len(clip_frames) < clip_length:
            clip_frames.append(frame_rgb)
        print(f"Processing clip {clip_count + 1}/{clip_amount}")
        probs, pred_idx, confidence = process_video_naman(clip_frames=clip_frames, processor=processor, model=model, device=device)
        ia_prob = probs[1]
        if ia_prob >= 0.5: 
            if ia_prob > ia_threshold:
                is_ia = True
            ia_clips += 1
            ia_predictions.append(probs[1]) 
        predictions.append({
            "clip_num": clip_count+1,
            "clip_period": (clip_start, clip_end),
            "prediction": "IA" if pred_idx else "REAL",
            "confidence": round(confidence, 2) # [0-100]
        })



    # VIDEO PROCESADO 

    # Confianza media de los clips detectados como IA
    ia_confidences = np.mean(ia_predictions) if ia_predictions else 0

    

    # Resultado
    if len(predictions) == 0: # Evitar el error si el video estaba vacío
        return None

    return {
        "label": "IA" if is_ia else "Real",
        "ia_confidence": ia_confidences, # [0-1]
        "ia_clips": ia_clips,
        "clips_info": predictions,
    }



def process_video_naman(clip_frames, processor, model, device):
    inputs = processor(frames=clip_frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        _, logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # array normalizado [%real, %fake]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx]) * 100
    
    return probs, pred_idx, confidence