import asyncio
import logging

import magic
from beanie import PydanticObjectId

from common.consts import SrtString, XMLString, JsonStr
from common.models.core import Translation
from services.constructor import SubtitlesResults
from services.convertors import xml_to_srt
from services.format_handlers import SRTHandler, JSONHandler, QTtextHandler


async def run_translation(
        task: Translation,
        blob_content,
        mime_type: str = None
) -> SubtitlesResults | SrtString | JsonStr | XMLString:
    """

    :param task: a translation DB obj, associated with this run
    :param blob_content: raw string of the subtitles file
    :param mime_type: if not provided, will try to detect the mime type

    :returns SubtitlesResults if raw_results is True,
             otherwise returns a string of translated subtitles in the uploaded format.
    """
    if not mime_type:
        mime_type = magic.Magic()
        detected_mime_type = mime_type.from_buffer(blob_content)
        detected_mime_type = detected_mime_type.lower()
    else:
        detected_mime_type = mime_type.lower()

    if 'xml' in detected_mime_type:
        blob_content = xml_to_srt(text=blob_content)
        handler = SRTHandler
        # handler = XMLHandler
    elif 'json' in detected_mime_type:
        handler = JSONHandler
    elif 'srt' in detected_mime_type:
        handler = SRTHandler
    elif 'qtext' in detected_mime_type:
        handler = QTtextHandler
    else:
        raise ValueError(f"Unsupported file type: {detected_mime_type}")
    return await handler(raw_content=blob_content, translation_obj=task).run()


async def main(**name_to_paths):
    from common.config import settings
    from common.db import init_db
    await init_db(settings, [Translation])
    ret = []
    for name, path in name_to_paths.items():
        with open(path, 'r') as f:
            data = f.read()

        task = Translation(target_language='Hebrew', source_language='English', subtitles=[],
                           project_id=PydanticObjectId(), name=name)
        await task.save()
        ret.append(await run_translation(task=task, blob_content=data, mime_type='srt'))
    return ret


def logging_setup():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s %(name)s:%(message)s",
        force=True,
    )  # Change these settings for your own purpose, but keep force=True at least.
    logging.getLogger('httpcore').setLevel(logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)


if __name__ == '__main__':
    logging_setup()
    _paths = {
        'Suits 0104': '/Users/yovel.c/PycharmProjects/services/sublineStreamlit/srts/suits0104/original_en.srt',
        'Suits 0108': '/Users/yovel.c/PycharmProjects/services/sublineStreamlit/srts/suits0108/Suits - 1x08 - Identity Crisis.HDTV.L0L.en.srt'
    }
    _ret = asyncio.run(main(**_paths))
    print(_ret)
