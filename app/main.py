import asyncio
import io
import zipfile
from collections import defaultdict

import pandas as pd
import streamlit as st

import logging
import streamlit.logger

from beanie import PydanticObjectId, Link
from beanie.odm.operators.find.comparison import In

from pydantic import BaseModel, model_validator, Field
from streamlit.runtime.uploaded_file_manager import UploadedFile

from common.consts import SrtString
from common.models.core import Ages, Genres, Project, Client, ClientChannel
from common.config import mongodb_settings
from common.db import init_db
from common.models.translation import Translation, SRTBlock, ModelVersions
from common.models.users import User

# Streamlit Applications
from auth import get_authenticator
from costs import costs_panel
from new_comparsion import newest_ever_compare, TranslationFeedbackV2
from new_stats import stats
from system_stats import get_stats, view_stats, get_data_for_stats, TWO_HOURS

streamlit.logger.get_logger = logging.getLogger
streamlit.logger.setup_formatter = None
streamlit.logger.update_formatter = lambda *a, **k: None
streamlit.logger.set_log_level = lambda *a, **k: None
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(asctime)s %(name)s:%(message)s",
    force=True,
)

streamlit_handler = logging.getLogger("streamlit")
streamlit_handler.setLevel(logging.DEBUG)
logging.getLogger('httpcore').setLevel(logging.INFO)
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('watchdog.observers').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

if not st.session_state.get('setPageConf') == 1:
    st.set_page_config(layout="wide")
    st.session_state['setPageConf'] = 1
    st.cache_resource.clear()

name, authentication_status, username = get_authenticator().login('Login', 'main')
st.session_state["authentication_status"] = authentication_status

if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')

if st.session_state["authentication_status"] == False:
    st.error('The username or password you have entered is invalid')


@st.cache_resource
def connect_DB():
    st.cache_data.clear()
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


class TranslationLight(BaseModel):
    id: PydanticObjectId = Field(..., alias='_id')
    project: Link[Project]
    engine_version: ModelVersions = Field(default=ModelVersions.V039, alias='modelVersion')

    @property
    def project_id(self):
        if isinstance(self.project, Link):
            return self.project.ref.id
        return self.project.id


async def _get_viewer_data():
    fbs, projects, translations = await asyncio.gather(
        TranslationFeedbackV2.find_all().to_list(),
        Project.find_all().project(Projection).to_list(),
        Translation.find_all().project(TranslationLight).to_list()
    )
    return {'fbs': fbs, 'projects': projects, 'translations': translations}


@st.cache_data(ttl=TWO_HOURS)
def get_viewer_data():
    if st.session_state.get('DB') is None:
        connect_DB()
    ret = asyncio.run(_get_viewer_data())
    return ret


def subtitles_viewer_from_db():
    ret = get_viewer_data()
    existing_feedbacks, projects, translations = ret['fbs'], ret['projects'], ret['translations']
    names = [d.name for d in existing_feedbacks]
    proj_id_to_name = {proj.id: proj.name for proj in projects}
    proj_name_to_id = {proj.name: proj.id for proj in projects}
    proj_id_to_ts = defaultdict(list)
    for ts in translations:
        proj_id_to_ts[ts.project_id].append(ts)

    existing, new = dict(), dict()
    for proj_id, _name in proj_id_to_name.items():
        if _name in names:
            existing[proj_id] = _name
        else:
            new[proj_id] = _name

    def format_name(n):
        _id = proj_name_to_id[n]
        available_versions_repr = f'({", ".join([t.engine_version.value for t in proj_id_to_ts[_id]])})'
        return f'{n} {available_versions_repr}'

    with st.form('forma'):
        chosenObj = st.selectbox('Choose Translation', options=list(new.values()), format_func=format_name)
        revision = st.file_uploader('Additional Subtitles', type=['srt', 'xml', 'txt', 'nfs'])
        submit = st.form_submit_button('Get')
        if submit:
            _id = [k for k, v in new.items() if v == chosenObj][0]
            st.session_state['projectId'] = _id
            if revision:
                string_data = _parse_file_upload(revision)
                st.session_state['file'] = string_data

    with st.form('forma2'):
        chosenObj = st.selectbox('Update Existing Review', options=list(existing.values()), format_func=format_name)
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
    data, fbs = get_data_for_stats()
    data = asyncio.run(get_stats(data, fbs))
    for _stats in data:
        st.header(_stats.version.value.upper())
        data = [
            {'ID': str(row['id']), 'name': row['name'], 'State': row['State'],
             'Reviewed': row['Reviewed'], 'Delete': row['Delete'], 'Amount Rows': row['Amount Rows'],
             'Amount OG Words': row['Amount OG Words'], 'Amount Translated Words': row['Amount Translated Words']}
            for row in _stats.translations
        ]
        df = pd.DataFrame(data)
        edited_df = st.data_editor(df, column_config={'Reviewed': st.column_config.SelectboxColumn(disabled=True)},
                                   use_container_width=True, hide_index=True)

        if st.button('Delete', key=f'{_stats.version.value}_delete'):
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
