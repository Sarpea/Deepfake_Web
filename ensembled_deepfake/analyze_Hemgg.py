import cv2, os, torch, numpy as np
# importante saber que la ultima version de transformers da error en este momento, "transformers==4.45.2" es la mejor que funciona actualmente
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
from pathlib import Path

# Ruta del modelo local

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_video_Hemgg(video_path: str, interval: int = 3, ia_threshold: float = 0.85):

    """
    interval: cantidad de frames saltados por cada frame analizado
    ia_threshold: a partir de que probabilidad se considera IA seguro
    """
    
    MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model_Video_MAE_1")
    
    # Carga del modelo y preprocesador
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # datos para el proceso
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_length = 16 #constante para cada modelo
    frame_cont = 0
    clip_amount = int(total_frames // (clip_length * interval))
    if total_frames % (clip_length * interval) > (clip_length * interval) // 2:
        clip_amount += 1
    clip_frames = []
    clip_count = 0
    is_ia = False
    ia_clips = 0
    predictions = []
    ia_predictions = [] # nivel de confianza de IA

    while True:
        ret, frame = cap.read()
        if not ret:  
            break      
        # Convertir BGR -> RGB
        frame_cont += 1
        if frame_cont % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            assert frame_rgb.dtype == np.uint8, f"Dtype incorrecto: {frame_rgb.dtype}"
            clip_frames.append(frame_rgb)

            if len(clip_frames) == clip_length:
                print(f"Processing clip {clip_count + 1}/{clip_amount}")
                probs = process_video_hemgg(clip_frames, processor, model)
                ia_prob = probs[0][1].item()
                if ia_prob >= 0.5: 
                    if ia_prob > ia_threshold:
                        is_ia = True
                    ia_clips += 1
                    ia_predictions.append(ia_prob) # nos servirá para calcular la confianza de la prediccion más adelante
                

            # Calcular segundo de inicio y fin del clip
                clip_start = round((frame_cont-(clip_length*interval)) / fps, 2)
                clip_end = round(frame_cont / fps, 2)


            # Almacenar resultados en una lista aparte para los clips
                predictions.append({
                    "clip_num": clip_count+1,
                    "clip_period": (clip_start, clip_end),
                    "prediction": "IA" if ia_prob > 0.5 else "REAL",
                    "confidence": round(ia_prob * 100, 2) # [0-100]
                })
                clip_count += 1
                clip_frames = []
    
    # Procesar ultimo clip del video (Este fragmento es igual que el del bucle, sirve para procesar el ultimo clip)
    # Hay que tener en cuenta que los tiempos de este clip varian con respecto a los otros
    if (clip_length // 2) < len(clip_frames) < clip_length:
        clip_start = round((frame_cont-len(clip_frames)*interval) / fps, 2)
        clip_end = round(frame_cont / fps, 2)
        while len(clip_frames) < clip_length:
            clip_frames.append(frame_rgb)
        print(f"Processing clip {clip_count + 1}/{clip_amount}")
        probs = process_video_hemgg(clip_frames, processor, model)
        ia_prob = probs[0][1].item()
        if ia_prob >= 0.5: 
            if ia_prob > ia_threshold:
                is_ia = True
            ia_clips += 1
            ia_predictions.append(ia_prob)
        predictions.append({
            "clip_num": clip_count+1,
            "clip_period": (clip_start, clip_end),
            "prediction": "IA" if ia_prob > 0.5 else "REAL",
            "confidence": round(ia_prob * 100, 2) # [0-100]
        })
    
    cap.release()

    # VIDEO PROCESADO 


    # Confianza media de los clips detectados como IA
    ia_confidences = np.mean(ia_predictions) if ia_predictions else 0

    

    # Resultado
    
    return {
        "label": "IA" if is_ia else "Real",
        "ia_confidence": ia_confidences, # [0-1]
        "ia_clips": ia_clips,
        "clips_info": predictions,
    }


def process_video_hemgg(clip_frames, processor, model):
    inputs = processor(images=clip_frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs