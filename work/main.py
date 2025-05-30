from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from face_detection_service import recognize_face_from_video

# 📦 App FastAPI
app = FastAPI(title="Microservice de reconnaissance faciale dans une vidéo")

# 🔄 CORS (⚠️ à restreindre pour la prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ En prod, mets ["https://ton-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Création des dossiers nécessaires
UPLOAD_DIR = "uploads"
REF_DIR = "references"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

# 🌐 Exposer les fichiers statiques (vidéos et images)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/references", StaticFiles(directory=REF_DIR), name="references")

# ✅ Test rapide
@app.get("/")
def home():
    return {"message": "Bienvenue sur le microservice de reconnaissance faciale dans une vidéo 🎥"}

# 🔍 Route principale : détection de visage après découpe en scènes
@app.post("/detect_face")
async def detect_face(
    video_file: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    threshold: float = Form(0.5),
    scenes_api_url: str = Form(...)
):
    """
    Upload de la vidéo et de l'image de référence, puis :
    1. Appel au microservice de découpe en scènes via scenes_api_url,
    2. Extraction des frames,
    3. Détection du visage cible dans ces frames.
    """
    return await recognize_face_from_video(
        video_file, reference_image, threshold, scenes_api_url
    )
