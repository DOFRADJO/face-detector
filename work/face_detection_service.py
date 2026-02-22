import os
from face_detection import detect_and_recognize_faces_from_frames
from utils import save_file_async
from gateway import send_video_to_scene_microservice  # 🔁 Client HTTP

REFERENCE_DIR = "references"
FRAMES_ROOT_DIR = "frames"

async def upload_reference_image(file):
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    path = await save_file_async(file, REFERENCE_DIR)
    return {"path": path}

def recognize_face_from_video(video_name, reference_filename, scenes_api_url, threshold=0.5):
    reference_path = os.path.join(REFERENCE_DIR, reference_filename)

    # Étape 1 : vérifier que l'image de référence existe
    if not os.path.isfile(reference_path):
        return {"error": f"Image de référence introuvable : {reference_filename}"}

    # Étape 2 : appeler le microservice distant de découpe de scènes
    scene_response = send_video_to_scene_microservice(video_name, scenes_api_url, threshold)
    if "error" in scene_response:
        return {"error": f"Erreur depuis le microservice de scènes : {scene_response['error']}"}

    frames_folder = os.path.join(FRAMES_ROOT_DIR, video_name)

    if not os.path.isdir(frames_folder):
        return {"error": f"Dossier de frames introuvable après découpe : {frames_folder}"}

    # Étape 3 : faire la reconnaissance faciale
    return detect_and_recognize_faces_from_frames(frames_folder, reference_path, threshold)
