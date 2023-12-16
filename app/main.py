import asyncio
import datetime
import io
import time
import zipfile
from collections import defaultdict

import openai
import pandas as pd
import streamlit as st

import logging
import streamlit.logger

from beanie import PydanticObjectId, Document
from beanie.exceptions import CollectionWasNotInitialized
from pydantic import BaseModel, model_validator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from common.consts import SrtString
from common.models.core import Translation, SRTBlock
from common.config import settings
from common.db import init_db
from services.runner import run_translation, XMLHandler, SRTHandler
from services.constructor import SubtitlesResults, pct

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
                       project_id=PydanticObjectId(), name=_name)
    await task.save()
    try:
        ret: SubtitlesResults = await run_translation(task=task, model=_model, blob_content=file, raw_results=True)
        st.session_state['name'] = ret
    except openai.APITimeoutError as e:
        st.error('Translation Failed Due To OpenAI API Timeout, Please Try Again Later')
        await task.delete()
        raise e

    t2 = time.time()
    logger.debug('finished translation, took %s seconds', t2 - t1)
    st.session_state['bar'].progress(100, text='Done!')


def clean():
    del st.session_state['name']


def download_button(name: str, srt_string1: SrtString, srt_string2: SrtString):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        with io.BytesIO(srt_string1.encode('utf-8')) as srt1_buffer:
            zip_file.writestr('subtitlesV1.srt', srt1_buffer.getvalue())

        with io.BytesIO(srt_string2.encode('utf-8')) as srt2_buffer:
            zip_file.writestr('subtitlesV2.srt', srt2_buffer.getvalue())

    zip_buffer.seek(0)
    name = name.replace(' ', '_').strip()
    st.download_button(label='Download Zip', data=zip_buffer, file_name=f'{name}_subtitles.zip', mime='application/zip')


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
            results: SubtitlesResults = st.session_state['name']
            download_button(
                name=name,
                srt_string1=results.to_srt(revision=False, translated=True),
                srt_string2=results.to_srt(revision=True, translated=True)
            )


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


def display_comparsion_panel(name, subtitles):
    labels = ['Gender Mistake', 'Time Tenses', 'Names', 'Slang',
              'Name "as is"', 'not fit in context', 'Plain Wrong Translation']

    select_box_col = lambda label: st.column_config.SelectboxColumn(
        width='medium', label=label, required=False, options=labels
    )
    select_cols = ['V1 Error 1', 'V1 Error 2']
    config = {
        'Original Language': st.column_config.TextColumn(width='large'),
        'Glix Translation 1': st.column_config.TextColumn(width='large'),
        'V1 Error 1': select_box_col('V1 Error 1'),
    }
    if 'Glix Translation 2' in subtitles:
        select_cols.extend(['V2 Error 1', 'V2 Error 2'])
        config.update(
            {'Glix Translation 2': st.column_config.TextColumn(width='large'),
             'V2 Error 1': select_box_col('V2 Error 1'),
             'V2 Error 2': select_box_col('V2 Error 2'), }
        )

    # Find the length of the longest list
    max_length = max(len(lst) for lst in subtitles.values())
    # Pad shorter lists with None
    for key in subtitles:
        subtitles[key] += [None] * (max_length - len(subtitles[key]))

    df = pd.DataFrame(subtitles)
    for col in select_cols:
        df[col] = pd.Series([None] * len(df), index=df.index)

    edited_df = st.data_editor(df, key='df', column_config=config, use_container_width=True)

    if st.button('Submit'):
        marked_rows = []
        for row in edited_df.to_dict(orient='records'):
            for col in select_cols:
                if row[col] not in ('None', None):
                    marked_rows.append(row)
                    break

        feedback_obj = asyncio.run(save_to_db(marked_rows, edited_df, name=name))
        st.write('Successfully Saved Results to DB!')
        st.info(f"Num Rows: {len(edited_df)}\n Num Mistakes: {len(marked_rows)} ({feedback_obj.error_pct} %))")

    if st.button('Export Results'):
        blob = edited_df.to_csv(header=True)
        st.download_button('Export', data=blob, file_name=f'{name}.csv', type='primary', on_click=clean)

    # st_timeline()


def subtitle_viewer():
    st.title("Subtitle Viewer")

    with st.form("file_upload"):
        name: str = st.text_input(label='Movie Or Series Name, i.e; FriendsS05E12')
        # fname = lambda path: f'{path.parent.name}/{path.name}'
        # paths = {fname(path): str(path) for path in find_relevant_video_files()}
        video_path = st.text_input(label='Video URL', value=None)
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

    if 'subtitles' in st.session_state and st.session_state['subtitles']:
        if video_path:
            with st.echo():
                st.video(
                    data=video_path,
                    # subtitles={'EN': original_content,
                    #            'Hebrew V1': given_translation,
                    #            'Hebrew V2': third_revision}
                )

        st.divider()
        st.divider()

        display_comparsion_panel(name=name, subtitles=st.session_state['subtitles'])


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


def subtitles_viewer_from_db():
    translation_names = asyncio.run(Translation.find(Translation.name != None).project(Projection).to_list())
    name_to_id = {proj.name: proj.id for proj in translation_names}
    with st.form('forma'):
        chosen_name = st.selectbox('Choose Translation', options=list(name_to_id.keys()))
        submit = st.form_submit_button('Get')
        if submit:
            chosen_id = name_to_id[chosen_name]
            translation = asyncio.run(Translation.get(chosen_id))
            rows = sorted(list(translation.subtitles), key=lambda x: x.index)
            subtitles = {
                'Original Language': [row.content for row in rows],
                'Glix Translation 1': [row.translations.content for row in rows],
            }
            revs = [row.translations.revision for row in translation.subtitles]
            if not all([r is None for r in revs]):
                subtitles['Glix Translation 2'] = revs

            st.session_state['subtitles'] = subtitles

    if 'subtitles' in st.session_state:
        display_comparsion_panel(name=chosen_name, subtitles=st.session_state['subtitles'])


class Stats(BaseModel):
    totalChecked: int
    totalFeedbacks: int
    errorsCounter: dict[str, int]
    errors: dict[str, list[str]]
    errorPct: float


async def get_stats():

    feedbacks, sum_checked_rows = [], 0
    by_error = defaultdict(list)
    await init_db(settings, [TranslationFeedback])
    async for feedback in TranslationFeedback.find_all():
        feedbacks.extend(feedback.marked_rows)
        sum_checked_rows += feedback.total_rows
        for row in feedback.marked_rows:
            if row['V1 Error 1'] not in ('None', None):
                by_error[row['V1 Error 1']].append(row)

    return Stats(
        totalChecked=sum_checked_rows,
        totalFeedbacks=len(feedbacks),
        errorsCounter={key: len(val) for key, val in by_error.items()},
        errors=by_error,
        errorPct=round((len(feedbacks) / sum_checked_rows) * 100, 2)
    )


def view_stats():
    stats: Stats = asyncio.run(get_stats())
    samples = {key: val[:5] for key, val in stats.errors.items()}
    total_errors = sum(list(stats.errorsCounter.values()))
    st.metric('Total Checked Rows', stats.totalChecked)
    st.metric('Total Errors Count', total_errors)
    for k, amount in stats.errorsCounter.items():
        in_pct = pct(amount, total_errors)
        st.metric(k, amount, f'{in_pct}% of total errors')

    st.divider()
    st.divider()
    st.header('Samples')
    for k, v in samples.items():
        st.subheader(k)
        st.table(pd.DataFrame(v))
        st.divider()


page_names_to_funcs = {
    "Translate": translate_form,
    "Subtitle Viewer": subtitle_viewer,
    'Viewer From DB': subtitles_viewer_from_db,
    'Engine Stats': view_stats
}
app_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
page_names_to_funcs[app_name]()
