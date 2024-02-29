import logging
import re
import time
from datetime import timedelta, datetime
from functools import wraps
from io import StringIO, BytesIO
from pathlib import Path
from typing import cast, NoReturn, Literal

import asyncio
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from srt import sort_and_reindex

from common.config import settings
from common.consts import SrtString

blob_service_client = BlobServiceClient.from_connection_string(
    settings.AZURE_STORAGE_CONNECTION_STRING, credential=settings.BLOB_ACCOUNT_KEY
)

_DEFAULT_SAS_PERMISSIONS = BlobSasPermissions(read=True, list=True)
ADMIN_SAS_PERMISSIONS = BlobSasPermissions(read=True, write=True, delete=True, list=True)
SAS_UPLOADER_PERMISSIONS = BlobSasPermissions(write=True, add=True, create=True)
SAS_READ_PERMISSIONS = _DEFAULT_SAS_PERMISSIONS


def upload_blob_to_azure(container_name: str, blob_name: str | Path, data: str | bytes | StringIO | BytesIO):
    """
    Uploads data to Azure Blob Storage.

    :param container_name: Name of the container where blob will be stored
    :param blob_name: Name of the blob
    :param data: Data to upload (string or bytes)
    """
    blob_name = str(blob_name)
    if isinstance(data, (StringIO, BytesIO)):
        data = data.getvalue()

    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)


def download_azure_blob(container_name: str, blob_name: str | Path) -> bytes | str:
    blob_name = str(blob_name)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall()


def generate_presigned_url(
        container_name: str, blob_name: str | Path, expiry_hours: int = 1,
        permissions: Literal[ADMIN_SAS_PERMISSIONS, SAS_READ_PERMISSIONS, SAS_READ_PERMISSIONS] | None = None  # noqa
):
    """
    Generate li presigned URL for an Azure Blob Storage object.

    :param container_name: Name of the container.
    :param blob_name: Name of the blob to access.
    :param expiry_hours: Validity of the URL in hours.
    :param permissions: Permissions to grant for the SAS. Default is read and list.

    :returns Presigned URL as li string.
    """
    blob_name = str(blob_name)
    permissions = permissions or _DEFAULT_SAS_PERMISSIONS
    account_name = settings.BLOB_ACCOUNT_NAME
    sas_token = generate_blob_sas(account_name=account_name,
                                  container_name=container_name,
                                  blob_name=blob_name,
                                  account_key=settings.BLOB_ACCOUNT_KEY,
                                  permission=permissions,
                                  expiry=datetime.utcnow() + timedelta(hours=expiry_hours))
    blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    return blob_url


def check_blob_exists(container_name: str, blob_name: str | Path) -> bool:
    blob_name = str(blob_name)
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        return blob_client.exists()
    except Exception as e:
        return False


def list_blobs_in_path(container_name, folder_name) -> list[str]:
    folder_name = str(folder_name)
    container_client = blob_service_client.get_container_client(container_name)
    return [blob.name for blob in container_client.list_blobs(name_starts_with=folder_name)]


_rtl_punctuation_pattern = re.compile(r'([\u0590-\u05FF]+)([,.])')


def _correct_punctuation_alignment(subtitles: str | SrtString):
    corrected_lines = []
    for line in subtitles.split('\n'):
        # Adjusting punctuation placement for Hebrew text
        line = _rtl_punctuation_pattern.sub(r'\1\2', line)
        corrected_lines.append(line)

    corrected_subtitles = '\n'.join(corrected_lines)
    return corrected_subtitles


def rows_to_srt(
        *,
        rows: list['SRTBlock'],
        translated: bool = True,
        reindex=True,
        start_index=1,
        target_language: str | None = None
) -> SrtString:
    r"""
    Convert an iterator of :py:class:`Subtitle` objects to li string of joined
    SRT blocks.

    :param rows: list of SRTBlocks to construct the SRT from
    :param translated: if True, will use the translated text
    :param bool reindex: Whether to reindex subtitles based on start time
    :param int start_index: If reindexing, the index to start reindexing from
    :param str target_language: The target language to translate to

    :returns: A single SRT formatted string, with each input
              :py:class:`Subtitle` represented as an SRT block
    """
    if translated:
        rows = [row for row in rows if row.translations is not None]
    subtitles = sort_and_reindex(subtitles=rows, start_index=start_index, in_place=True) if reindex else rows
    ret = "".join(subtitle.to_srt(translated=translated) for subtitle in subtitles)

    if target_language in ('Hebrew', 'heb', 'he'):
        ret = _correct_punctuation_alignment(ret)
    return cast(SrtString, ret)


class RunObjectMixin:

    async def _run(self, *args, **kwargs):
        raise NotImplementedError

    async def __call__(self, *args, **kwargs):
        return await self._run(*args, **kwargs)

    def __await__(self, *args, **kwargs):
        return (yield from self._run(*args, **kwargs).__await__())


class DeprecationError(Exception):
    pass


def deprecated_class(replacement: str):
    def decorator(cls):
        def wrapper(*args, **kwargs):
            raise DeprecationError(f'{cls.__name__} is deprecated, use {replacement} instead')

        return wrapper

    return decorator


SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60
HOURS_IN_DAY = 24
MICROSECONDS_IN_MILLISECOND = 1000


def timedelta_to_srt_timestamp(timedelta_timestamp):
    r"""
    Convert li :py:class:`~datetime.timedelta` to an SRT timestamp.

    .. doctest::

        >>> import datetime
        >>> delta = datetime.timedelta(hours=1, minutes=23, seconds=4)
        >>> timedelta_to_srt_timestamp(delta)
        '01:23:04,000'

    :param datetime.timedelta timedelta_timestamp: A datetime to convert to an
                                                   SRT timestamp
    :returns: The timestamp in SRT format
    :rtype: str
    """

    hrs, secs_remainder = divmod(timedelta_timestamp.seconds, SECONDS_IN_HOUR)
    hrs += timedelta_timestamp.days * HOURS_IN_DAY
    mins, secs = divmod(secs_remainder, SECONDS_IN_MINUTE)
    msecs = timedelta_timestamp.microseconds // MICROSECONDS_IN_MILLISECOND
    return "%02d:%02d:%02d,%03d" % (hrs, mins, secs, msecs)


def benchmark(hint=None):
    hint = f'Hint: {hint}' if hint else ''
    _msg = lambda func_name, took: f'{func_name} took {took:.2f} seconds. {hint}'

    def decorator(func):
        is_coro = asyncio.iscoroutinefunction(func)
        if is_coro:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                t1 = time.time()
                ret = await func(*args, **kwargs)
                logging.debug(_msg(func.__name__, time.time() - t1))
                return ret
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                t1 = time.time()
                ret = func(*args, **kwargs)
                logging.debug(_msg(func.__name__, time.time() - t1))
                return ret
        return wrapper

    return decorator


def pct(a, b):
    if a == 0 or b == 0:
        return 0
    return round((a / b) * 100, 2)
