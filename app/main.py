import asyncio
import datetime
import time
from typing import cast

import streamlit as st
from io import StringIO
import srt as srt_lib

from logic.constructor import SrtString, SRTTranslator, TranslatedSRT
from logic.function import SRTBlock

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


async def translate(_name, _rows, _target_language, _model):
    t1 = time.time()
    logger.debug('starting translation request')
    translator = SRTTranslator(target_language=_target_language, project_name=_name, rows=_rows, model=_model)
    _ret = await translator(num_rows_in_chunk=50)
    st.session_state['name'] = _ret
    t2 = time.time()
    logger.debug('finished translation, took %s seconds', t2 - t1)
    st.session_state['bar'].progress(100, text='Done!')


async def translate_via_xml(_name, _model, xml_string, target_language):
    with st.spinner('Translating...'):
        t1 = time.time()
        logger.debug('starting translation request')
        root, parent_map, blocks = extract_text_from_xml(xml_string)
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

            assert source_language != target_language
            submitted = st.form_submit_button("Translate")
            if submitted:
                bar = st.progress(0, 'Parsing File...')
                st.session_state['bar'] = bar
                if uploaded_file.name.endswith('srt'):
                    st.session_state['stage'] = 1
                    rows = parse_srt(uploaded_file=uploaded_file)
                    asyncio.run(translate(_name=name, _rows=rows, _target_language=target_language, _model=model))
                else:
                    st.session_state['stage'] = 1
                    string_data = uploaded_file.getvalue().decode("utf-8")
                    asyncio.run(translate_via_xml(xml_string=string_data, _name=name, _model=model,
                                                  target_language=target_language))

        if st.session_state.get('name', False):
            srt = st.session_state['name']
            if isinstance(srt, TranslatedSRT):
                text = srt.srt
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
        root, parent_map, rows = extract_text_from_xml(string_data)
    return rows


def subtitle_viewer():
    st.title("Subtitle Viewer")
    with st.form("Translate"):
        file1 = st.file_uploader("Upload a file 1", type=["srt", "xml", 'nfs'])
        file2 = st.file_uploader("Upload a file 2", type=["srt", "xml", 'nfs'])

        submitted = st.form_submit_button("Compare")
        if submitted:
            subtitles1, subtitles2 = parse_file(file1), parse_file(file2)
            if len(subtitles1) != len(subtitles2):
                st.write("Warning: Subtitle lists are not of the same length!")

            for s1, s2 in zip(subtitles1, subtitles2):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"{format_time(s1.start)} --> {format_time(s1.end)}")
                    st.write(s1.content)

                with col2:
                    st.write(f"{format_time(s2.start)} --> {format_time(s2.end)}")
                    st.write(s2.content)


def save_approved_translations():
    ...


def finetuner():
    st.title("Finetuner")
    submit = st.button('Submit Accepted Translations')
    if submit:
        save_approved_translations()

    with st.form("Finetune"):
        name = st.text('Series/Movie Name')
        file1 = st.file_uploader("Upload a file 1", type=["srt", "xml", 'nfs'])
        file2 = st.file_uploader("Upload a file 2", type=["srt", "xml", 'nfs'])
        st.session_state['saved'] = []
        submitted = st.form_submit_button("Compare")

        if submitted:
            subtitles1, subtitles2 = parse_file(file1), parse_file(file2)
            if len(subtitles1) != len(subtitles2):
                st.write("Warning: Subtitle lists are not of the same length!")
            i = 0
            for s1, s2 in zip(subtitles1, subtitles2):
                col1, col2, col3 = st.columns([0.4, 0.4, 0.2])

                with col1:
                    st.write(f"{format_time(s1.start)} --> {format_time(s1.end)}")
                    st.write(s1.content)

                with col2:
                    st.write(f"{format_time(s2.start)} --> {format_time(s2.end)}")
                    st.write(s2.content)

                with col3:
                    agree = st.checkbox('Accept Translation', key=f'accept_{i}')
                    if agree:
                        st.session_state['saved'].append({'original': s1.content, 'translation': s2.content})
                i += 1


page_names_to_funcs = {
    "Translate": translate_form,
    "Subtitle Viewer": subtitle_viewer,
    'Fine Tune': finetuner
}

demo_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
