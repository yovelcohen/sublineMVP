import asyncio
import io
import zipfile

import pandas as pd
import streamlit as st

import logging
import streamlit.logger

from beanie import PydanticObjectId
from beanie.odm.operators.find.comparison import In

from pydantic import BaseModel, model_validator, Field
from streamlit.runtime.uploaded_file_manager import UploadedFile

from auth import get_authenticator
from common.consts import SrtString
from common.models.core import Ages, Genres, Project, Client, ClientChannel
from common.config import mongodb_settings
from common.db import init_db
from common.models.translation import Translation
from common.models.users import User
from costs import costs_panel
from new_comparsion import newest_ever_compare, TranslationFeedbackV2
from new_stats import stats
from system_stats import get_stats, view_stats

streamlit.logger.get_logger = logging.getLogger
streamlit.logger.setup_formatter = None
streamlit.logger.update_formatter = lambda *a, **k: None
streamlit.logger.set_log_level = lambda *a, **k: None

# Then set our logger in a normal way
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(asctime)s %(name)s:%(message)s",
    force=True,
)  # Change these settings for your own purpose, but keep force=True at least.

streamlit_handler = logging.getLogger("streamlit")
streamlit_handler.setLevel(logging.DEBUG)
logging.getLogger('httpcore').setLevel(logging.INFO)
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('watchdog.observers').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")

name, authentication_status, username = get_authenticator().login('Login', 'main')
st.session_state["authentication_status"] = authentication_status

if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')
if st.session_state["authentication_status"] == False:
    st.error('The username or password you have entered is invalid')


def connect_DB():
    _docs, _db = asyncio.run(
        init_db(mongodb_settings, [Translation, TranslationFeedbackV2, Project, Client, ClientChannel, User])
    )
    st.session_state['DB'] = _db
    return _docs, _db


if st.session_state.get('DB') is None and st.session_state["authentication_status"] is True:
    logger.info('initiating DB Connection And collections')
    connect_DB()


class Projection(BaseModel):
    id: PydanticObjectId
    name: str

    @model_validator(mode='before')
    @classmethod
    def validate_name(cls, data: dict):
        if '_id' in data:
            _id = data.pop('_id')
            data['id'] = _id
        return data


GENRES_MAP = {member.name: member.value for member in Genres}  # noqa
AGES_MAP = {
    '0+': Ages.ZERO,
    '3+': Ages.THREE,
    '6+': Ages.SIX,
    '12+': Ages.TWELVE,
    '16+': Ages.SIXTEEN,
    '18+': Ages.EIGHTEEN
}


def string_to_enum(value: str, enum_class):
    for enum_member in enum_class:
        if enum_member.value == value or enum_member.value == value.lower():
            return enum_member
    raise ValueError(f"{value} is not a valid value for {enum_class.__name__}")


def download_button(_name: str, srt_string1: SrtString, srt_string2: SrtString = None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        with io.BytesIO(srt_string1.encode('utf-8')) as srt1_buffer:
            zip_file.writestr('subtitlesV1.srt', srt1_buffer.getvalue())
        if srt_string2:
            with io.BytesIO(srt_string2.encode('utf-8')) as srt2_buffer:
                zip_file.writestr('subtitlesV2.srt', srt2_buffer.getvalue())

    zip_buffer.seek(0)
    _name = _name.replace(' ', '_').strip()
    st.download_button(label='Download Zip', data=zip_buffer, file_name=f'{_name}_subtitles.zip',
                       mime='application/zip')


async def _delete_docs(to_delete):
    to_delete = list(map(lambda x: PydanticObjectId(x) if isinstance(x, str) else x, to_delete))
    q = Translation.find(In(Translation.id, to_delete))
    ack = await q.delete()
    return ack.deleted_count


def _parse_file_upload(uploaded_file: UploadedFile):
    if uploaded_file.name.endswith('nfs') or uploaded_file.name.endswith('xml'):
        st.session_state['format'] = 'xml'
    elif uploaded_file.name.endswith('srt'):
        st.session_state['format'] = 'srt'
    elif uploaded_file.name.endswith('txt'):
        st.session_state['format'] = 'qtext'

    if st.session_state['format'] == 'xml':
        st.session_state['stage'] = 1
        string_data = uploaded_file.getvalue().decode("utf-8")
    else:
        st.session_state['stage'] = 1
        string_data = uploaded_file.getvalue()
    return string_data


class NameOnlyProjection(BaseModel):
    id: PydanticObjectId = Field(..., alias='_id')
    name: str


async def get_name_from_proj(obj_: Translation):
    _id = obj_.project.id if isinstance(obj_.project, Project) else obj_.project.ref.id
    p = await Project.find(Project.id == _id).project(NameOnlyProjection).first_or_none()
    if not p:
        raise ValueError('project not found???')
    return p.name


class NameProject(Project):
    name: str


def subtitles_viewer_from_db():
    if st.session_state.get('DB') is None:
        connect_DB()

    existing_feedbacks = asyncio.run(TranslationFeedbackV2.find_all().to_list())
    names = [d.name for d in existing_feedbacks]
    project_names = {
        proj.id: proj.name for proj in asyncio.run(Project.find_all().project(Projection).to_list())
    }

    existing, new = dict(), dict()
    for proj_id, _name in project_names.items():
        if _name in names:
            existing[proj_id] = _name
        else:
            new[proj_id] = _name

    with st.form('forma'):
        chosenObj = st.selectbox('Choose Translation', options=list(new.values()))
        revision = st.file_uploader('Additional Subtitles', type=['srt', 'xml', 'txt', 'nfs'])
        submit = st.form_submit_button('Get')
        if submit:
            _id = [k for k, v in new.items() if v == chosenObj][0]
            st.session_state['projectId'] = _id
            if revision:
                string_data = _parse_file_upload(revision)
                st.session_state['file'] = string_data

    with st.form('forma2'):
        chosenObj = st.selectbox('Update Existing Review', options=list(existing.values()))
        revision = st.file_uploader('Additional Subtitles', type=['srt', 'xml', 'txt', 'nfs'])
        submit = st.form_submit_button('Get')
        if submit:
            _id = [k for k, v in existing.items() if v == chosenObj][0]
            st.session_state['projectId'] = _id
            if revision:
                string_data = _parse_file_upload(revision)
                st.session_state['file'] = string_data

    if 'projectId' in st.session_state:
        project_id: str = st.session_state['projectId']
        newest_ever_compare(project_id)


def manage_existing():
    data = asyncio.run(get_stats())
    for stats in data:
        st.header(stats.version.value.upper())
        data = [
            {'ID': str(row['id']), 'name': row['name'], 'State': row['State'],
             'Reviewed': row['Reviewed'], 'Delete': row['Delete'], 'Amount Rows': row['Amount Rows'],
             'Amount OG Words': row['Amount OG Words'], 'Amount Translated Words': row['Amount Translated Words']}
            for row in stats.translations
        ]
        df = pd.DataFrame(data)
        edited_df = st.data_editor(df, column_config={'Reviewed': st.column_config.SelectboxColumn(disabled=True)},
                                   use_container_width=True, hide_index=True)

        if st.button('Delete'):
            rows = edited_df.to_dict(orient='records')
            to_delete = [proj['ID'] for proj in rows if proj['Delete'] is True]
            if to_delete:
                ack = asyncio.run(_delete_docs(to_delete=to_delete))
                logger.info(f'Deleted {ack} translation projects')
                st.success(f'Successfully Deleted {ack} Projects, Refresh Page To See Changes')

        st.divider()


if st.session_state["authentication_status"] is True:
    page_names_to_funcs = {
        'Subtitles Viewer': subtitles_viewer_from_db,
        'Engine Stats': view_stats,
        'Engine Stats 2': stats,
        'Manage Existing Translations': manage_existing,
        'Costs Breakdown': costs_panel
    }
    app_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
    page_names_to_funcs[app_name]()
