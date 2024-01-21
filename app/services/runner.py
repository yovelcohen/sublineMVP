from typing import Literal, Type

import magic

from xml.etree import ElementTree as ET  # noqa
from app.services.parsers.format_handlers import SRTHandler, JSONHandler, QTextHandler, BaseHandler, XMLHandler

MIME_TYPES = Literal['xml', 'json', 'srt', 'qtext']


def verify_mime(mime_type: MIME_TYPES, blob):
    checker = magic.Magic()
    detected_mime_type = checker.from_buffer(blob)
    detected_mime_type = detected_mime_type.lower()
    if mime_type != detected_mime_type:
        raise ValueError(f"mime type {mime_type} does not match the file's mime type: {detected_mime_type}")


def get_handler_by_mime_type(mime_type: MIME_TYPES | None = None, blob_content: str = None) -> Type[BaseHandler]:
    """
    Returns li handler class based on the mime type of the file
    :param mime_type:
    :param blob_content:
    """
    if not mime_type:
        mime_type = magic.Magic()
        detected_mime_type = mime_type.from_buffer(blob_content)
        detected_mime_type = detected_mime_type.lower()
    else:
        detected_mime_type = mime_type.lower()

    if 'xml' in detected_mime_type or 'nfs' in detected_mime_type:
        handler = XMLHandler
    elif 'json' in detected_mime_type:
        handler = JSONHandler
    elif 'srt' in detected_mime_type:
        handler = SRTHandler
    elif 'qtext' in detected_mime_type:
        handler = QTextHandler
    else:
        raise ValueError(f"Unsupported file type: {detected_mime_type}")
    return handler
