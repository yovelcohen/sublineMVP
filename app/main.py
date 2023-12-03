import asyncio
import datetime
import time
from typing import cast

import streamlit as st
from io import StringIO
import srt as srt_lib
from streamlit.runtime.uploaded_file_manager import UploadedFile

from logic.constructor import SrtString, SRTTranslator, SubtitlesResults, SRTBlock

import logging
import streamlit.logger

from logic.xml_reader import translate_xml, extract_text_from_xml

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


async def translate(_name, _rows, _target_language, _model, _num_versions: int | None = None):
    t1 = time.time()
    logger.debug('starting translation request')
    _num_versions = None if _num_versions == 1 else _num_versions
    translator = SRTTranslator(target_language=_target_language, project_name=_name, rows=_rows, model=_model)
    _ret = await translator(num_rows_in_chunk=50, num_options=_num_versions)
    st.session_state['name'] = _ret
    t2 = time.time()
    logger.debug('finished translation, took %s seconds', t2 - t1)
    st.session_state['bar'].progress(100, text='Done!')


async def translate_via_xml(_name, _model, xml_string, target_language):
    with st.spinner('Translating...'):
        t1 = time.time()
        logger.debug('starting translation request')
        root, parent_map, blocks = extract_text_from_xml(xml_data=xml_string)
        ret = await translate_xml(name=_name, model=_model, target_language=target_language,
                                  blocks=blocks, root=root, parent_map=parent_map)
        st.session_state['name'] = ret
        t2 = time.time()
        logger.debug('finished translation, took %s seconds', t2 - t1)


def clean():
    del st.session_state['name']


def parse_srt(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = cast(SrtString, stringio.read())
    rows = [SRTBlock(index=row.index, start=row.start, end=row.end, content=row.content)
            for row in srt_lib.parse(string_data)]
    return rows


def translate_form():
    if st.session_state.get('stage', 0) != 1:
        with st.form("Translate"):
            name = st.text_input("Name")
            type_ = st.selectbox("Type", ["Movie", "Series"])
            uploaded_file = st.file_uploader("Upload a file", type=["srt", "xml", 'nfs'])
            source_language = st.selectbox("Source Language", ["English", "Hebrew", 'French', 'Arabic', 'Spanish'])
            target_language = st.selectbox("Target Language", ["Hebrew", 'French', 'Arabic', 'Spanish'])
            model = st.selectbox('Model', ['good', 'best'], help="good is faster, best is more accurate")
            num_versions = st.selectbox('Amount Of Options', [1, 2, 3], help='How many translation versions to return')

            assert source_language != target_language
            submitted = st.form_submit_button("Translate")
            if submitted:
                bar = st.progress(0, 'Parsing File...')
                st.session_state['bar'] = bar
                if uploaded_file.name.endswith('srt'):
                    st.session_state['stage'] = 1
                    rows = parse_srt(uploaded_file=uploaded_file)
                    asyncio.run(translate(_name=name, _rows=rows, _target_language=target_language, _model=model,
                                          _num_versions=num_versions))
                else:
                    st.session_state['stage'] = 1
                    string_data = uploaded_file.getvalue().decode("utf-8")
                    asyncio.run(translate_via_xml(xml_string=string_data, _name=name, _model=model,
                                                  target_language=target_language))

        if st.session_state.get('name', False):
            srt = st.session_state['name']
            if isinstance(srt, SubtitlesResults):
                text = srt.to_srt(target_language=target_language)
            else:
                text = srt
            mime = uploaded_file.name.rsplit('.')[-1]
            st.download_button('Download Translation', data=text, file_name=f'{name}_{target_language}.{mime}',
                               type='primary', on_click=clean)


def format_time(_time: datetime.timedelta):
    return str(_time.total_seconds() * 1000)


def parse_file(uploaded_file) -> list[SRTBlock]:
    if uploaded_file.name.endswith('srt'):
        st.session_state['stage'] = 1
        rows = parse_srt(uploaded_file=uploaded_file)
    else:
        st.session_state['stage'] = 1
        string_data = uploaded_file.getvalue().decode("utf-8")
        root, parent_map, rows = extract_text_from_xml(xml_data=string_data)
    return rows


def subtitle_viewer():
    def view_vol(_col, rows, start, _rows_per_page):
        with _col:
            for row in rows[start:start + _rows_per_page]:
                st.write(f"{format_time(row.start)} --> {format_time(row.end)}")
                st.write(row.content)

    st.title("Subtitle Viewer")
    rows_per_page = 10  # Adjust the number of rows per page as needed

    with st.form("file_upload"):
        files: list[UploadedFile] = st.file_uploader("Upload Subtitle Files", type=["srt", "xml", 'nfs'],
                                                     accept_multiple_files=True)
        submitted = st.form_submit_button("Compare")

        if submitted:
            subtitles_lists = [parse_file(file) for file in files]
            st.session_state['subtitles_lists'] = subtitles_lists

    if 'subtitles_lists' in st.session_state and st.session_state['subtitles_lists']:
        subtitles_lists = st.session_state['subtitles_lists']

        # Initialize session state for pagination
        for i in range(len(subtitles_lists)):
            if f'start_row_{i}' not in st.session_state:
                st.session_state[f'start_row_{i}'] = 0

        columns = st.columns(len(subtitles_lists))
        for i, (col, subtitles) in enumerate(zip(columns, subtitles_lists)):
            view_vol(col, subtitles, st.session_state[f'start_row_{i}'], rows_per_page)

            # Pagination buttons
            prev, _, next = col.columns([1, 2, 1])
            if next.button("Next", key=f'next_{i}'):
                st.session_state[f'start_row_{i}'] += rows_per_page
            if prev.button("Previous", key=f'prev_{i}'):
                st.session_state[f'start_row_{i}'] = max(0, st.session_state[f'start_row_{i}'] - rows_per_page)


def save_approved_translations():
    ...


page_names_to_funcs = {
    "Translate": translate_form,
    "Subtitle Viewer": subtitle_viewer,
    # 'Fine Tune': finetuner
}

demo_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
