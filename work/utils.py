import os
import uuid
from fastapi import UploadFile

async def save_file_async(file: UploadFile, folder: str) -> str:
    """
    Sauvegarde asynchrone d’un fichier UploadFile dans un dossier donné avec un nom unique.

    Args:
        file (UploadFile): Fichier à sauvegarder.
        folder (str): Chemin du dossier de destination.

    Returns:
        str: Chemin absolu du fichier sauvegardé.
    """
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(folder, filename)

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    return filepath
