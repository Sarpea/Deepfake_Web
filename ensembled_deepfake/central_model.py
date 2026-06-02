from .analyze_Hemgg import analyze_video_Hemgg # type: ignore
from .analyze_Jayala import analyze_video_jayala # type: ignore
from .analyze_Naman import analyze_video_Naman # type: ignore
from .mongo_repo import MongoRepository

   
"""
{                  formato de video_doc recibido
    "video_id": yt.video_id,
    "title": yt.title,
    "keywords": yt.keywords,
    "views": yt.views,
    "length": yt.length,
    "author": yt.author,
    "video_path": file_path
}
"""

def process_video_values(video_path :str, interval: int = 3, device='cpu'):
    """
    video_path: ruta al video
    interval: se divide la cantidad de frames procesados por el intervalo (para ganar eficiencia)
    device: cpu o cuda solo afectan a los modelos de Naman y de Jayala, procesando en procesador(cpu) o grafica (cuda), pero el de Hemgg siempre va a ser por cpu
    Devuelve un documento con el label (IA, REAL), la confianza, y si hay consenso de los 3 modelos sobre una zanja de tiempo. 
    Tambien devuelve los clip predictions para seguir trabajando con ellos (de momento no tienen uso)

    formato de salida de los modelos:
    {
        "label": "IA" / "Real",
        "ia_confidence": ia_confidence, # [0-1]
        "ia_clips": ia_clips,
        "clips_info": {
            "clip_num": clip_num,
            "clip_period": (clip_start, clip_end),
            "prediction": "IA" / "REAL",
            "confidence": confidence # [0-100]
            }, ...
    }

    Este modelo devuelve para cada video:
    {
        "label": "IA" / "REAL",
        "confidence": confianza # [0-100]
        "ia_agreement": boolean # los 3 modelos estan de acuerdo en que un mismo fragmento tiene IA
    }
    """

    # Fase de proceso de videos (la parte critica de la ejecucion)
    print("\n Processing video with Naman712/Deep-fake-detection model \n")
    naman_values = analyze_video_Naman(video_path, interval, device=device)
    print(naman_values["label"] + "\n")
    print(naman_values["ia_confidence"])
    print("\n Processing video with Hemgg/deepfake_model_Video-MAE-1 model \n")
    hemgg_values = analyze_video_Hemgg(video_path, interval)
    print(hemgg_values["label"] + "\n")
    print(hemgg_values["ia_confidence"])
    print("\n Processing video with jayalakshmikopuri/cvit_deepfake_video_detection_V1 model \n")
    jayala_values = analyze_video_jayala(video_path, interval, device=device)
    print(jayala_values["label"] + "\n")
    print(jayala_values["ia_confidence"])


    clip_predictions = {
        "Naman": naman_values["clips_info"],
        "Hemgg": hemgg_values["clips_info"],
        "Jayala": jayala_values["clips_info"]
    }

    naman_conf = naman_values["ia_confidence"] if naman_values["label"] == "IA" else -0.5
    hemgg_conf = hemgg_values["ia_confidence"] if hemgg_values["label"] == "IA" else -0.5
    jayala_conf = jayala_values["ia_confidence"] if jayala_values["label"] == "IA" else -0.5
    
    total_confidence = enssemble_confidence(naman_conf, hemgg_conf, jayala_conf)

    if naman_values["label"] == "IA" and hemgg_values["label"] == "IA" and jayala_values["label"] == "IA":
        model_agreement = ia_collide(clip_predictions)
    else:
        model_agreement = False


    result = {
        "label": "IA" if total_confidence > 0.5 else "REAL",
        "confidence": round(total_confidence*100, 2) if total_confidence > 0.5 else round((1-total_confidence)*100, 2),
        "ia_agreement": model_agreement # agreement (boolean) se refiere a que los 3 modelos estan de acuerdo en que un mismo clip contiene IA
    }
    
    
    repo = MongoRepository()
    repo.update_video_path(video_path, result)

    return result



def enssemble_confidence(conf_Naman, conf_Hemgg, conf_jayala):
    """
    Se va a usar una tecnica de enssemble por media ponderada, ya que los modelos en cuestion tienen diferentes accuracys en los datasets en los que han sido probados
    se va a tener en cuenta eso a la hora de definir que peso tiene cada una.
    Naman: 87% -5% (porque se ha eliminado el modulo de face-recognition) 82% (Experto en FaceSwapping)
    Hemgg: 89% (Experto en deteccion de movimiento)
    Jayalakshmikopuri: Este modelo es curioso porque tiene accuracys muy distintas segun el dataset con el que ha sido probado, en los FaceSwap, FaceShifter y NaturalTextures no supera el 70% (FaceShifter es la peor con un 46%)
        sin embargo, en los datasets DeepFakeDetection y Deepfake tiene las increibles cifras de 91 y 93% respectivamente, se le va a asignar un peso de 85%, para que no afecte demasiado al criterio de los otros 2
    
    Normalizados los pesos, las ponderaciones son las siguientes:
        Naman712: 0.32
        Hemgg: 0.35
        Jayalakshmikopuri: 0.33
    """
    naman_weight = 0.32 * conf_Naman
    hemgg_weight = 0.35 * conf_Hemgg
    jayala_weight = 0.33 * conf_jayala
    return (naman_weight + hemgg_weight + jayala_weight)



def ia_collide(clip_predictions):
    """
    Devuelve True si hay una zanja de tiempo en el que los 3 modelos han detectado IA, False en caso contrario
    """
    naman_clip_info = clip_predictions["Naman"]
    hemgg_clip_info = clip_predictions["Hemgg"]
    jayala_clip_info = clip_predictions["Jayala"]

    naman_ia_clips, hemgg_ia_clips, jayala_ia_clips = ia_clips_extraction(naman_clip_info, hemgg_clip_info, jayala_clip_info)


    collide = False
    if (len(naman_ia_clips) == 0 or len(hemgg_ia_clips) == 0) or len(jayala_ia_clips) == 0:
        return False
    for j in jayala_ia_clips:                       # usamos los clips del modelo jayala porque son los mas largos
        start, end = j["period"]
        for n in naman_ia_clips:
            start_n, end_n = n["period"]
            if start_n >= end:                      #si el clip naman sobrepasa al clip de jayala rompemos el bucle
                break
            if start_n >= start and start_n < end:  # el clip naman empieza a medias del clip jayala
                collide = True
            elif end_n > start and end_n <= end:    # el clip jayala empieza a medias del clip naman
                collide = True
        
        for h in hemgg_ia_clips:                    # este bucle sigue la misma logica que el anterior
            start_n, end_n = h["period"]
            if start_n >= end:                      
                break
            if start_n >= start and start_n < end:  
                if collide:
                    return True            
            elif end_n > start and end_n <= end:    # el clip jayala empieza a medias del clip naman
                if collide:
                    return True
        collide = False

    return False




def ia_clips_extraction(naman_clip_info, hemgg_clip_info, jayala_clip_info):

    """
    Devuelve 3 listas, una por cada modelo con solo los clips que han sido clasificados como IA
    """

    naman_clips = []
    hemgg_clips = []
    jayala_clips = []
    
    for n in naman_clip_info:
        if n["prediction"] == "IA":
            data = {
                "period": n["clip_period"],
                "confidence": n["confidence"]
            }
            naman_clips.append(data)
    
    for h in hemgg_clip_info:
        if h["prediction"] == "IA":
            data = {
                "period": h["clip_period"],
                "confidence": h["confidence"]
            }
            hemgg_clips.append(data)
    
    for j in jayala_clip_info:
        if j["prediction"] == "IA":
            data = {
                "period": j["clip_period"],
                "confidence": j["confidence"]
            }
            jayala_clips.append(data)
    
    return naman_clips, hemgg_clips, jayala_clips




