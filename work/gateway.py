# gateway.py
import requests

def send_video_to_scene_microservice(video_filename: str, api_url: str, threshold: float = 0.5):
    """
    Appelle le microservice de découpe pour extraire les frames de la vidéo.
    """
    try:
        response = requests.post(
            api_url,  # ex: http://localhost:8001/split_scene
            json={"video_filename": video_filename, "threshold": threshold},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur de communication avec le microservice de découpe : {str(e)}"}
