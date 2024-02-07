import asyncio

import streamlit as st
from beanie import PydanticObjectId
import pandas as pd

from common.utils import pct
from common.config import mongodb_settings
from common.db import init_db
from common.models.core import Project, Client, ClientChannel
from common.models.users import User
from common.models.translation import Translation, TranslationFeedbackV2


async def get_translation(project_id) -> Translation:
    project_id = PydanticObjectId(project_id)
    ts = await Translation.find(Translation.project.id == project_id).sort(+Translation.updated_at).first_or_none()
    return ts


@st.cache_resource
def connect_DB():
    _docs, _db = asyncio.run(
        init_db(mongodb_settings, [Translation, TranslationFeedbackV2, Project, Client, ClientChannel, User])
    )
    st.session_state['DB'] = _db
    return _docs, _db


async def _update_results(edited_df, translation: Translation):
    tagged = {i: row['Gender'] for i, row in enumerate(edited_df.to_dict(orient='records'), start=1)
              if row['Gender'] is not None}
    num_tags = 0
    for row in translation.subtitles:
        if tag := tagged.get(row.index):
            row.speaker_gender = tag
            num_tags += 1
    await translation.save()
    st.success(f'Successfully tagged {num_tags} rows and saved project')


async def _display_gender_tagging_table(project_id: str):
    from streamlit_utils import SelectBoxColumn
    if st.session_state.get('DB') is None:
        connect_DB()

    translation = await get_translation(project_id)

    data = {'Time Stamp': [], 'English': [], 'Gender': []}
    for row in translation.subtitles:
        data['Time Stamp'].append(f'{row.start} --> {row.end}')
        data['English'].append(row.content)
        data['Gender'].append(row.speaker_gender)

    column_config = {
        'Time Stamp': st.column_config.TextColumn(width='large', disabled=True),
        'English': st.column_config.TextColumn(width='large', disabled=True),
        "Gender": SelectBoxColumn('Gender', ['Male', 'Female'])
    }

    df = pd.DataFrame(data)
    df.index = df.index + 1
    with st.sidebar:
        with_tag = [row for row in translation.subtitles if row.speaker_gender is not None]
        st.write(f'Tagged: {len(with_tag)} ({pct(len(with_tag), len(translation.subtitles))}%)')

    edited_df = st.data_editor(df, column_config=column_config, use_container_width=True)
    if st.button('Submit'):
        await _update_results(edited_df, translation)


def gender_tagger(project_id):
    return asyncio.run(_display_gender_tagging_table(project_id))
