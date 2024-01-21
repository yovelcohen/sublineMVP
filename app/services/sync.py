import subprocess
from pathlib import Path

import srt

from app.common.models.translation import SRTBlock

BASE_PATH = Path(__file__).resolve().parent


def sync_video_to_file(video_path, srt_string) -> list[SRTBlock]:
    temp_file_p = BASE_PATH / 'temp.srt'
    temp_output = BASE_PATH / 'output.srt'
    with open(temp_file_p, 'w') as f:
        f.write(srt_string)

    command = ['ffs', video_path, '-i', temp_file_p, '-o', temp_output, '--gss']

    subprocess.run(command)

    with open(temp_output, 'r') as f:
        output = f.read()
        rows = [
            SRTBlock(index=row.index, start=row.start, end=row.end, content=row.content)
            for row in srt.parse(output)
        ]
    return rows
