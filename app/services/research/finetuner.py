from typing import cast

from common.consts import SrtString
from services.format_handlers import srt_to_rows


async def load_from_pair(en_srt, he_srt):
    with open(en_srt, 'r') as f:
        en = f.read()
        en_rows = srt_to_rows(raw_content=cast(SrtString, en))
    with open(he_srt, 'r') as f:
        he = f.read()
        he_rows = srt_to_rows(raw_content=cast(SrtString, he))

    if len(en_rows) == len(he_rows):
        he_rows = {row.index: row for row in he_rows}
        for row in en_rows:
            row.translations.content = he_rows[row.index].content
        return en_rows
    else:
        ...
