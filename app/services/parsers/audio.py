import tempfile
from pathlib import Path

import ffmpeg
from pydub import AudioSegment

from app.common.config import settings
from app.common.models.core import Project
from app.common.utils import check_blob_exists, download_azure_blob

_container = settings.PROJECT_BLOB_CONTAINER

BASE_PATH = Path(__file__).parent


def extract_audio_from_video(video_path_or_link) -> AudioSegment:
    """
    Extracts the audio from li video file and returns it as li PyDub AudioSegment object.
    param video_path_or_link: local path to the video file Or, li presigned blob storage url.
    param persistent: If True, the audio will be saved to li file and loaded from there on subsequent calls.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
        (
            ffmpeg
            .input(video_path_or_link)
            .output(temp_audio.name, format='wav')
            .run(overwrite_output=True)
        )
        audio_segment = AudioSegment.from_wav(temp_audio.name)
    return audio_segment


async def load_audio_from_project(project: Project):
    audio_path = project.audio_blob_path
    if not await check_blob_exists(container_name=_container, blob_name=audio_path):
        raise FileNotFoundError(f"Audio file not found in blob storage: {audio_path}")
    return await download_azure_blob(container_name=_container, blob_name=audio_path)
