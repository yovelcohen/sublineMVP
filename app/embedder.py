import asyncio
import logging
from typing import Literal

import httpx
import pandas as pd
from pydantic import BaseModel
import streamlit as st

from common.consts import THREE_MINUTES
from common.db import init_db
from common.config import settings, mongodb_settings
from common.models.core import Project

logger = logging.getLogger(__name__)


class SubtitleEditRequest(BaseModel):
    index: int
    original: str
    translation: str
    source_language: str
    target_language: str

    media_type: Literal['series', 'movie'] = 'series'
    source_media: str | None = None
    season: int | None = None
    episode: int | None = None


async def send_embed_pair_request(
        *,
        project: Project,
        target_language: str,
        row_index: int,
        original_content: str,
        new_translation: str
):
    async with httpx.AsyncClient() as client:
        data = SubtitleEditRequest(
            index=row_index,
            original=original_content,
            translation=new_translation,
            source_language=project.source_language.value,  # noqa
            target_language=target_language,
            source_media=project.name,
            season=project.media_meta.season,
            episode=project.media_meta.episode
        )
        ret = await client.post(settings.EMBEDDING_FUNC_URL, json=data.model_dump())
        logger.info(f'sent embedding function request', extra={'status_code': ret.status_code})
        return ret


SYMBOLS_TO_REMOVE = {'&quot', '♪', '{\an8}'}


@st.cache_resource
def connect_DB():
    _docs, _db = asyncio.run(
        init_db(mongodb_settings, [Project])
    )
    st.session_state['DB'] = _db
    return _docs, _db


async def _get_projects():
    return Project.find(Project.media_meta.original_subs)


@st.cache_data(ttl=THREE_MINUTES)
def get_viewer_projects():
    if st.session_state.get('DB') is None:
        connect_DB()
    ret = asyncio.run(_get_projects())
    return ret


def upload_and_embed_file():
    with st.form('embedform'):
        st.info(
            """The file should be a tabular data with the following columns: Source Media, Season, Episode, Row Index, Original Content, Translation, Source Language (2 letter code), Target Language (2 letter code).
The first 3 columns can be left empty."""
        )
        file = st.file_uploader('Upload Translation Pairs', type=['xlsx', 'csv', 'tsv'])
        if st.form_submit_button('EmbedPairsButton'):
            seperator = '\t' if file.name.endswith('.tsv') else ','
            df = (pd.read_csv(file, delimiter=seperator, header=True)
                  if file.name.endswith('.csv') else
                  pd.read_excel(file, header=True))
            data = [
                SubtitleEditRequest()
            ]
