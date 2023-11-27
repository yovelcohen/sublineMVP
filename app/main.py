import asyncio
import logging
import time
from typing import cast

import streamlit as st
from io import StringIO
import srt as srt_lib

from logic.constructor import SrtString, SRTTranslator, TranslatedSRT
from logic.function import SRTBlock

# Disable the Streamlit's overrides
import logging
import streamlit.logger

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

logger = logging.getLogger(__name__)


async def translate(_name, _rows, _target_language, _model):
    with st.spinner('Translating...'):
        t1 = time.time()
        logger.debug('starting translation request')
        translator = SRTTranslator(target_language=_target_language, project_name=_name, rows=_rows, model=_model)
        _ret = await translator(num_rows_in_chunk=75)
        st.session_state['name'] = _ret
        t2 = time.time()
        logger.debug('finished translation, took %s seconds', t2 - t1)


def clean():
    del st.session_state['name']


if st.session_state.get('stage', 0) != 1:
    with st.form("Translate"):
        name = st.text_input("Name")
        type_ = st.selectbox("Type", ["Movie", "Series"])
        uploaded_file = st.file_uploader("Upload a file", type="srt")
        source_language = st.selectbox("Source Language", ["English", "Hebrew", 'French', 'Arabic', 'Spanish'])
        target_language = st.selectbox("Source Language", ["Hebrew", 'French', 'Arabic', 'Spanish'])
        model = st.selectbox('Model', ['best', 'good'])

        assert source_language != target_language
        submitted = st.form_submit_button("Translate")
        if submitted:
            st.session_state['stage'] = 1
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = cast(SrtString, stringio.read())
            rows = [SRTBlock(index=row.index, start=row.start, end=row.end, content=row.content)
                    for row in srt_lib.parse(string_data)]
            asyncio.run(translate(_name=name, _rows=rows, _target_language=target_language, _model=model))

    if st.session_state.get('name', False):
        srt: TranslatedSRT = st.session_state['name']
        st.download_button('Download SRT', data=srt.srt, file_name=f'{name}_{target_language}',
                           mime='srt', type='primary', on_click=clean)
