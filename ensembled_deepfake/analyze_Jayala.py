import cv2, os
import torch
import numpy as np

from .cvit_deepfake_video_detection_V1.model_architecture import CViT 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_video_jayala(video_path: str, interval: int = 3, ia_threshold: float = 0.75, device='cpu'):
    
    # 1. Crear el modelo con los mismos parámetros que se entrenó
    model = CViT(
        image_size=224,
        patch_size=7,
        num_classes=2,
        channels=512,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    )

    MODEL_WEIGHTS = os.path.join(BASE_DIR, "cvit_deepfake_video_detection_V1", "cvit2_deepfake_detection_ep_50.pth")

    # 2. Cargar los pesos
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # 3. Mover el modelo a CPU o GPU
    model.to(device)
    model.eval()

    # VARIABLES IMPORTANTES PARA EL MODELO
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_length = 30
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
            processed_frame = preprocess_frame(frame, device)
            clip_frames.append(processed_frame)


        # Se crean clips de 20 frames y se procesa cada clip por separado
            if len(clip_frames) == clip_length:   

                #Funcion de procesado
                print(f"Processing clip {clip_count + 1}/{clip_amount}")
                pred_idx, confidence = process_video_jayala(clip_frames=clip_frames, model=model)
                    
                # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
                if pred_idx:
                    if confidence >= ia_threshold:
                        is_ia = True
                    ia_clips += 1
                    ia_predictions.append(confidence) # nos servirá para calcular la confianza con la que se ha detectado IA en la prediccion más adelante
                

            # Calcular segundo de inicio y fin del clip
                clip_start = round((frame_cont-clip_length * interval) / fps, 2)
                clip_end = round(frame_cont / fps, 2)


            # Almacenar resultados en una lista aparte para los clips
                predictions.append({
                    "clip_num": clip_count+1,
                    "clip_period": (clip_start, clip_end),
                    "prediction": "IA" if pred_idx else "REAL",
                    "confidence": round(confidence*100, 2) # [0-100]
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
            clip_frames.append(processed_frame)
        print(f"Processing clip {clip_count + 1}/{clip_amount}")
        pred_idx, confidence = process_video_jayala(clip_frames=clip_frames, model=model)
        if pred_idx: 
            if confidence >= ia_threshold:
                is_ia = True
            ia_clips += 1
            ia_predictions.append(confidence) 
        confidence = round(confidence, 4)
        predictions.append({
            "clip_num": clip_count+1,
            "clip_period": (clip_start, clip_end),
            "prediction": "IA" if pred_idx else "REAL",
            "confidence": round(confidence*100, 2) # [0-100]
        })



    # VIDEO PROCESADO 


    # Confianza media de los clips detectados como IA
    ia_confidences = np.mean(ia_predictions) if ia_predictions else 0


    return {
        "label": "IA" if is_ia else "Real",
        "ia_confidence": ia_confidences, # [0-1]
        "ia_clips": ia_clips,
        "clips_info": predictions,
    }



def process_video_jayala(clip_frames, model): 
    """
    Este modelo trabaja frame a frame a diferencia de los otros, pero vamos a dividir el video en clips de 20 frames y vamos a trabajar cada bloque por separado para que los resultados sean comparables entre modelos
    """
    ia_frames = []
    all_probs = []
    batch = torch.stack(clip_frames)  # [30,3,224,224]
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)  # probs tiene forma [30][2], (30 arrays de forma [real, fake] uno por cada frame)


    # Aplicación de lógica
    fake_frames = 0
    for p in probs:
        fake_prob = p[1].item()
        all_probs.append(fake_prob)
        if fake_prob > 0.5:
            fake_frames += 1
            ia_frames.append(fake_prob) # Para poder hacer la media exclusiva de los frames clasificados como IA
    
    ia_confidence = np.mean(all_probs)
    pred_idx = 1 if ia_confidence > 0.5 else 0 # La media de la confianza de IA tiene que ser mayor a 0.5 o no se cuenta como IA
        

    
    if pred_idx: # Si el modelo detecta IA, se almacena exclusivamente la confianza de los clips detectados como IA, para ver con que confianza ha detectado la IA
        if fake_frames >= 15: # Si al menos 50% de los frames han sido detectados como IA, la confianza va a ser solamente de los clips IA (sin diluir)
            confidence = np.mean(ia_frames) 
        else:
            confidence = ia_confidence
    else:       # Si no detecta IA, se almacena la probabilidad media de todo el clip
        confidence = (1 - ia_confidence)

    return pred_idx, confidence.item()



def preprocess_frame(frame, device):
    
     # BGR -> RGB
     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # resize to 224x224
     frame = cv2.resize(frame, (224, 224))

     # numpy → tensor float32
     frame = torch.from_numpy(frame).float() / 255.0  # [224,224,3]

     # Normalize (ImageNet)
     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
     std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

     # HWC → CHW
     frame = frame.permute(2, 0, 1)  # [3,224,224]

     # Apply normalization
     frame = (frame - mean) / std

     # move to device
     frame = frame.to(device)

     return frame