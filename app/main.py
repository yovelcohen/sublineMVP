import asyncio
import datetime
import time

import pandas as pd
import streamlit as st

import logging
import streamlit.logger

from beanie import PydanticObjectId, Document
from beanie.exceptions import CollectionWasNotInitialized
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.common.models.core import Translation, SRTBlock
from app.services.runner import run_translation, XMLHandler, SRTHandler
from app.common.config import settings
from app.common.db import init_db

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


class TranslationFeedback(Document):
    name: str
    total_rows: int
    marked_rows: list[dict]

    @property
    def error_pct(self):
        return round(((len(self.marked_rows) / self.total_rows) / 100), 2)


if not st.session_state.get('DB'):
    logger.info('initiating DB Connection And collections')
    db, docs = asyncio.run(init_db(settings, [Translation, TranslationFeedback]))
    st.session_state['DB'] = db
    logger.info('Finished DB Connection And collections init process')


async def translate(_name, file: str, _target_language, _model):
    t1 = time.time()
    logger.debug('starting translation request')
    task = Translation(target_language=_target_language, source_language='English', subtitles=[],
                       project_id=PydanticObjectId())
    await task.save()
    ret = await run_translation(task=task, model=_model, revision=True, blob_content=file, raw_results=False)
    st.session_state['name'] = ret
    t2 = time.time()
    logger.debug('finished translation, took %s seconds', t2 - t1)
    st.session_state['bar'].progress(100, text='Done!')


def clean():
    del st.session_state['name']


def translate_form():
    if st.session_state.get('stage', 0) != 1:
        with st.form("Translate"):
            name = st.text_input("Name")
            uploaded_file = st.file_uploader("Upload a file", type=["srt", "xml", 'nfs'])
            source_language = st.selectbox("Source Language", ["English", "Hebrew", 'French', 'Arabic', 'Spanish'])
            target_language = st.selectbox("Target Language", ["Hebrew", 'French', 'Arabic', 'Spanish'])
            model = st.selectbox('Model', ['good', 'best'], help="good is faster, best is more accurate")

            assert source_language != target_language
            submitted = st.form_submit_button("Translate")
            if submitted:
                bar = st.progress(0, 'Parsing File...')
                st.session_state['bar'] = bar
                st.session_state['isSrt'] = bool(uploaded_file.name.endswith('srt'))
                if st.session_state['isSrt']:
                    st.session_state['stage'] = 1
                    string_data = uploaded_file.getvalue()
                else:
                    st.session_state['stage'] = 1
                    string_data = uploaded_file.getvalue().decode("utf-8")
                asyncio.run(
                    translate(_name=name, file=string_data, _target_language=target_language, _model=model)
                )

        if st.session_state.get('name', False):
            results = st.session_state['name']
            mime = uploaded_file.name.rsplit('.')[-1]
            st.download_button('Download Translation', data=results, file_name=f'{name}_{target_language}.{mime}',
                               type='primary', on_click=clean)


def format_time(_time: datetime.timedelta):
    return str(_time.total_seconds() * 1000)


def xml_to_rows(elements) -> list[SRTBlock]:
    def parse_ttml_timestamp(timestamp_str):
        if not timestamp_str:
            return
        milliseconds_str = timestamp_str.rstrip("t")
        milliseconds = int(milliseconds_str)
        return datetime.timedelta(milliseconds=milliseconds)

    ids = [elem.attrib[key] for elem in elements for key in elem.attrib if key.endswith("id")]
    blocks = [
        SRTBlock(
            content=elem.text.strip(),
            index=pk,
            style=elem.attrib.get("style"),
            region=elem.attrib.get("region"),
            start=parse_ttml_timestamp(elem.attrib.get("begin")),
            end=parse_ttml_timestamp(elem.attrib.get("end")),
        )
        for pk, elem in zip(ids, elements)
    ]
    return blocks


mock_translation = Translation(project_id=PydanticObjectId(), subtitles=[], target_language='null')


def parse_file(uploaded_file) -> list[str]:
    handler = SRTHandler if uploaded_file.name.endswith('srt') else XMLHandler
    st.session_state['stage'] = 1
    handler = handler(raw_content=uploaded_file.getvalue().decode("utf-8"), translation_obj=mock_translation)
    return [row.content for row in handler.to_rows()]


async def save_to_db(rows, df, name):
    params = dict(marked_rows=rows, total_rows=len(df), name=name)
    try:
        obj = TranslationFeedback(**params)
    except CollectionWasNotInitialized as e:
        await init_db(settings, [TranslationFeedback])
        obj = TranslationFeedback(**params)
    await obj.save()
    return obj


def subtitle_viewer():
    st.title("Subtitle Viewer")

    with st.form("file_upload"):
        name: str = st.text_input(label='Movie Or Series Name, i.e; FriendsS05E12')
        original_content: UploadedFile = st.file_uploader(
            "Upload Original Source Language Subtitle File",
            type=["srt", "xml", 'nfs'],
            accept_multiple_files=False
        )
        given_translation: UploadedFile = st.file_uploader(
            "Upload Original Translated Subtitle File",
            type=["srt", "xml", 'nfs'],
            accept_multiple_files=False
        )
        third_revision: UploadedFile = st.file_uploader(
            "Glix Generated Subtitle File",
            type=["srt", "xml", 'nfs'],
            accept_multiple_files=False
        )

        submitted = st.form_submit_button("Compare")

        if submitted:
            subtitles = {
                'Original Language': parse_file(original_content),
                'Glix Translation 1': parse_file(given_translation),
                'Glix Translation 2': parse_file(third_revision)
            }
            st.session_state['subtitles'] = subtitles
            # st.session_state['videoPath'] = video

    if 'subtitles' in st.session_state and st.session_state['subtitles']:
        subtitles = st.session_state['subtitles']
        config = {
            'Original Language': st.column_config.TextColumn(width='large'),
            'Glix Translation 1': st.column_config.TextColumn(width='large'),
            'Glix Translation 2': st.column_config.TextColumn(width='large'),
            'Error 1': st.column_config.SelectboxColumn(
                width='medium', label='Text Error 1', required=False,
                options=['Gender Mistake', 'Time Tenses', 'Names', 'Slang',
                         'Name "as is"', 'not fit in context', 'Plain Wrong Translation']),
            'Error 2': st.column_config.SelectboxColumn(
                width='medium', label='Text Error 2', required=False,
                options=['Gender Mistake', 'Time Tenses', 'Names', 'Slang',
                         'Name "as is"', 'not fit in context', 'Plain Wrong Translation'])
        }
        df = pd.DataFrame(subtitles)
        df['Error 1'] = pd.Series(['None'] * len(df), index=df.index)
        df['Error 2'] = pd.Series(['None'] * len(df), index=df.index)
        edited_df = st.data_editor(df, key='df', column_config=config, use_container_width=True)
        if st.button('Submit'):
            edited_df['Error 1'] = edited_df['Error 1'].apply(lambda x: None if x == 'None' else x)
            edited_df['Error 2'] = edited_df['Error 2'].apply(lambda x: None if x == 'None' else x)
            marked_rows = edited_df[(edited_df['Error 1']).notna() | (edited_df['Error 2'].notna())].to_dict(
                orient='records'
            )

            feedback_obj = asyncio.run(save_to_db(marked_rows, edited_df, name=name))
            st.write('Successfully Saved Results to DB!')
            st.info(f"Num Rows: {len(edited_df)}\n Num Mistakes: {len(marked_rows)} ({feedback_obj.error_pct} %))")

        if st.button('Export Results'):
            blob = edited_df.to_csv(header=True)
            st.download_button('Export', data=blob, file_name=f'{name}.csv', type='primary', on_click=clean)


page_names_to_funcs = {"Translate": translate_form, "Subtitle Viewer": subtitle_viewer}

demo_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
