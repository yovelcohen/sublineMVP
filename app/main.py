import asyncio

import streamlit as st

import logging
import streamlit.logger

from common.models.core import Project, Client, ClientChannel
from common.config import mongodb_settings
from common.db import init_db
from common.models.translation import Translation
from common.models.users import User

# Streamlit Applications
from auth import get_authenticator
from costs import costs_panel
from new_comparsion import TranslationFeedbackV2, subtitles_viewer_from_db
from prompt_viewer import view_prompts
from system_stats import view_stats

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

if st.session_state["authentication_status"] is True:
    page_names_to_funcs = {
        'Subtitles Viewer': subtitles_viewer_from_db,
        'Engine Stats': view_stats,
        'Costs Breakdown': costs_panel,
        'Prompt Viewer': view_prompts,
    }
    app_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
    page_names_to_funcs[app_name]()
