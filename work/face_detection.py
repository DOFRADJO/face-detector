import os
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from db import detections_col

def detect_and_recognize_faces_from_frames(frames_folder, reference_image_path, threshold=0.5):
    """
    Compare un visage de référence avec tous les visages détectés dans les frames d'une vidéo.

    Args:
        frames_folder (str): Dossier contenant les images extraites de la vidéo.
        reference_image_path (str): Chemin vers l'image de référence.
        threshold (float): Seuil de similarité pour confirmer une correspondance.

    Returns:
        dict: Résultats des correspondances détectées.
    """
    if not os.path.isdir(frames_folder):
        return {"error": f"Dossier introuvable : {frames_folder}"}

    if not os.path.isfile(reference_image_path):
        return {"error": f"Image de référence introuvable : {reference_image_path}"}

    # Initialisation des modèles
    mtcnn = MTCNN(keep_all=True, image_size=160)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # Embedding du visage de référence
    try:
        ref_img = Image.open(reference_image_path).convert('RGB')
        ref_face = mtcnn(ref_img)
        if ref_face is None:
            return {"error": "Aucun visage détecté dans l'image de référence."}
        ref_embedding = resnet(ref_face)[0].unsqueeze(0)
    except Exception as e:
        return {"error": f"Erreur lors du traitement de l'image de référence : {e}"}

    # Analyse des frames
    results = []
    image_files = sorted(f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png')))

    for file in image_files:
        frame_path = os.path.join(frames_folder, file)
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            boxes, _ = mtcnn.detect(frame_rgb)
            faces = mtcnn(frame_rgb)
        except Exception as e:
            continue  # Skip this frame if something goes wrong

        if faces is not None and boxes is not None:
            for i in range(faces.shape[0]):
                try:
                    emb = resnet(faces[i].unsqueeze(0))
                    sim = cosine_similarity(ref_embedding.detach().numpy(), emb.detach().numpy())[0][0]

                    if sim >= threshold:
                        match_info = {
                            "frame": file,
                            "similarity": float(sim),
                            "box": list(map(int, boxes[i]))
                        }
                        results.append(match_info)

                        # Enregistrement dans MongoDB
                        detections_col.update_one(
                            {"frame": file, "reference": os.path.basename(reference_image_path)},
                            {"$set": {
                                "frame": file,
                                "reference": os.path.basename(reference_image_path),
                                "similarity": float(sim),
                                "box": list(map(int, boxes[i]))
                            }},
                            upsert=True
                        )
                except Exception as e:
                    continue  # Skip this face if something goes wrong

    return {"matches": results}
