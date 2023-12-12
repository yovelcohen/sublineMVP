import os
import platform
from pathlib import Path

SUPPORTED_VID_FORMATS = ['.mp3', '.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv']


def find_relevant_video_files() -> list[Path]:
    # Paths for Desktop and Downloads in macOS and Windows
    paths, os_type = [], platform.system()
    if os_type == 'Windows':
        paths.extend([
            Path('C:/Users') / os.getlogin() / 'Desktop',  # Windows Desktop
            Path('C:/Users') / os.getlogin() / 'Downloads'  # Windows Downloads
        ])
    elif os_type == 'Darwin':  # macOS
        paths.extend([
            Path.home() / 'Desktop',
            Path.home() / 'Downloads'
        ])

    all_supported_files = []
    for path in paths:
        if path.exists():
            rel_files = [
                Path(os.path.join(root, file)) for root, dirs, files in os.walk(path) for file in files
                if any(file.endswith(ext) for ext in SUPPORTED_VID_FORMATS)
            ]
            all_supported_files.extend(rel_files)

    return all_supported_files
