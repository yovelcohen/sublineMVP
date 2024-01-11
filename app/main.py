import asyncio
import io
import statistics
import zipfile
from collections import defaultdict
from math import isnan

import pandas as pd
import streamlit as st

import logging
import streamlit.logger

from beanie import PydanticObjectId, Link
from beanie.odm.operators.find.comparison import In

from pydantic import BaseModel, model_validator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from auth import get_authenticator
from common.consts import SrtString
from common.models.core import Ages, Genres, Project, Client, ClientChannel
from common.config import mongodb_settings
from common.db import init_db
from common.models.translation import Translation, SRTBlock, ModelVersions
from common.models.users import User
from costs import costs_panel
from new_comparsion import TranslationFeedback, new_compare

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


def pct(a, b):
    if a == 0 or b == 0:
        return 0
    return round((a / b) * 100, 2)


st.set_page_config(layout="wide")

name, authentication_status, username = get_authenticator().login('Login', 'main')
st.session_state["authentication_status"] = authentication_status

if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')
if st.session_state["authentication_status"] == False:
    st.error('The username or password you have entered is invalid')

if not st.session_state.get('DB') and st.session_state["authentication_status"] is True:
    logger.info('initiating DB Connection And collections')
    db, docs = asyncio.run(
        init_db(settings=mongodb_settings,
                documents=[Translation, TranslationFeedback, Project, Client, ClientChannel],
                allow_index_dropping=True)
    )
    st.session_state['DB'] = db
    logger.info('Finished DB Connection And collections init process')


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


class Stats(BaseModel):
    version: ModelVersions
    totalChecked: int
    totalFeedbacks: int
    errorsCounter: dict[str, int]
    errors: dict[str, list[dict[str, str | None]]]
    errorPct: float
    totalOgCharacters: int
    totalTranslatedCharacters: int
    amountOgWords: int
    amountTranslatedWords: int
    translations: list[dict]


GENRES_MAP = {member.name: member.value for member in Genres}  # noqa
AGES_MAP = {
    '0+': Ages.ZERO,
    '3+': Ages.THREE,
    '6+': Ages.SIX,
    '12+': Ages.TWELVE,
    '16+': Ages.SIXTEEN,
    '18+': Ages.EIGHTEEN
}
STATES_MAP = {
    'p': 'Pending',
    'ip': 'In Progress',
    'aa': "Audio Intelligence",
    'va': "Video Intelligence",
    'co': 'Completed',
    'fa': 'Failed'
}
TEN_K, MILLION, BILLION = 10_000, 1_000_000, 1_000_000_000


def string_to_enum(value: str, enum_class):
    for enum_member in enum_class:
        if enum_member.value == value or enum_member.value == value.lower():
            return enum_member
    raise ValueError(f"{value} is not a valid value for {enum_class.__name__}")


def download_button(name: str, srt_string1: SrtString, srt_string2: SrtString = None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        with io.BytesIO(srt_string1.encode('utf-8')) as srt1_buffer:
            zip_file.writestr('subtitlesV1.srt', srt1_buffer.getvalue())
        if srt_string2:
            with io.BytesIO(srt_string2.encode('utf-8')) as srt2_buffer:
                zip_file.writestr('subtitlesV2.srt', srt2_buffer.getvalue())

    zip_buffer.seek(0)
    name = name.replace(' ', '_').strip()
    st.download_button(label='Download Zip', data=zip_buffer, file_name=f'{name}_subtitles.zip', mime='application/zip')


def _display_one_comparison_panel(rows: list[SRTBlock], revision_rows: list[SRTBlock] = None):
    rows = sorted(list(rows), key=lambda x: x.index)
    last_row = rows[-1]
    labels = ['Gender Mistake', 'Time Tenses', 'Names', 'Slang', 'Prepositions',
              'Name "as is"', 'not fit in context', 'Plain Wrong Translation']

    select_box_col = lambda label: st.column_config.SelectboxColumn(
        width='medium', label=label, required=False, options=labels
    )
    select_cols = ['Error 1', 'Error 2']
    config = {
        'Original Language': st.column_config.TextColumn(width='large'),
        'Glix Translation 1': st.column_config.TextColumn(width='large'),
        'Error 1': select_box_col('Error 1'),
        'Error 2': select_box_col('Error 2'),
    }

    if revision_rows:
        config['Additional Translation'] = st.column_config.TextColumn(width='large', disabled=True)


async def _get_translations_df() -> list[dict]:
    if not st.session_state.get('DB'):
        db, docs = asyncio.run(init_db(mongodb_settings, [Translation]))
        st.session_state['DB'] = db

    projs: list[Translation] = await Translation.find({}, fetch_links=True).to_list()

    def get_took(t):
        minutes, seconds = divmod(t, 60)
        return "{:02d}:{:02d}".format(int(minutes), int(seconds))

    return [
        {
            'id': proj.id,
            'name': await get_name_from_proj(proj),
            'Amount Rows': len(proj.subtitles),
            'State': STATES_MAP[proj.state.value],
            'Took': get_took(proj.took),
            'Engine Version': proj.engine_version.value,
            'Delete': False,
            'Amount OG Characters': sum([len(r.content) for r in proj.subtitles]),
            'Amount Translated Characters': sum(
                [len(r.translations.selection) for r in proj.subtitles if r.translations is not None]
            ),
            'Amount OG Words': sum([len(r.content.split()) for r in proj.subtitles]),
            'Amount Translated Words': sum(
                [len(r.translations.selection.split()) for r in proj.subtitles if r.translations is not None]
            )
        }
        for proj in projs
    ]


async def _get_stats() -> list[Stats]:
    await init_db(mongodb_settings, [TranslationFeedback])

    q = TranslationFeedback.find_all(fetch_links=True)
    count, data = await asyncio.gather(q.count(), _get_translations_df())
    by_v = {
        v: {'feedbacks': list(), 'sum_checked_rows': 0, 'all_names': set(),
            'name_to_count': dict(), 'by_error': defaultdict(list)}
        for v in set([t['Engine Version'] for t in data])
    }

    async for feedback in q:
        version = feedback.version
        by_v[version]['feedbacks'].extend(feedback.marked_rows)
        by_v[version]['sum_checked_rows'] += feedback.total_rows
        by_v[version]['all_names'].add(feedback.name)
        by_v[version]['name_to_count'][feedback.name] = len(feedback.marked_rows)
        for row in feedback.marked_rows:
            key = 'V1 Error 1' if 'V1 Error 1' in row else 'Error 1'
            key2 = 'V2 Error 1' if 'V2 Error 1' in row else 'Error 2'
            for k in (key, key2):
                if k in row and row[k] not in ('None', None):
                    row['Name'] = feedback.name
                    by_v[version]['by_error'][row[k]].append(row)

    for translation in data:
        version = translation['Engine Version']
        if 'translations' not in by_v[version]:
            by_v[version]['translations'] = list()
        if translation['name'] in by_v[version]['all_names'] and translation['State'] == 'Done':
            translation['Reviewed'] = True
            amount_errors = by_v[version]['name_to_count'][translation['name']]
        else:
            translation['Reviewed'] = False
            amount_errors = 0

        translation['Amount Errors'] = int(amount_errors)
        translation['Errors %'] = round(pct(amount_errors, translation['Amount Rows']), 1)
        by_v[version]['translations'].append(translation)

    stats = [Stats(
        version=v,
        totalChecked=data['sum_checked_rows'],
        totalFeedbacks=count,
        errorsCounter={key: len(val) for key, val in data['by_error'].items()},
        errors=data['by_error'],
        errorPct=pct(len(data['feedbacks']), data['sum_checked_rows']),
        totalOgCharacters=sum([row['Amount OG Characters'] for row in data['translations']]),
        totalTranslatedCharacters=sum([row['Amount Translated Characters'] for row in data['translations']]),
        amountOgWords=sum([row['Amount OG Words'] for row in data['translations']]),
        amountTranslatedWords=sum([row['Amount Translated Words'] for row in data['translations']]),
        translations=data['translations']
    ) for v, data in by_v.items()]
    return stats


def _format_number(num):
    if TEN_K <= num < MILLION:
        return f"{num / 1000:.1f}k"
    elif MILLION <= num < BILLION:
        return f"{num / MILLION:.2f}m"
    elif num >= BILLION:
        return f"{num / BILLION:.2f}b"
    else:
        return str(num)


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
    name: str


async def get_name_from_proj(obj_: Translation):
    p = await Project.find(Project.id == obj_.project.id).project(NameOnlyProjection).first_or_none()
    if not p:
        raise ValueError('project not found???')
    return p.name


class NameProject(Project):
    name: str


def subtitles_viewer_from_db():
    db, docs = asyncio.run(
        init_db(mongodb_settings, [Translation, TranslationFeedback, Project, Client, ClientChannel, User]))
    st.session_state['DB'] = db

    existing_feedbacks = asyncio.run(TranslationFeedback.find_all().to_list())
    names = [d.name for d in existing_feedbacks]
    project_names = {
        proj.id: proj.name for proj in asyncio.run(Project.find_all().project(Projection).to_list())
    }

    existing, new = dict(), dict()
    for proj_id, name in project_names.items():
        if name in names:
            existing[proj_id] = name
        else:
            new[proj_id] = name

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
        asyncio.run(new_compare(project_id))


def stats_for_version(stats: Stats):
    samples = {key: pd.DataFrame(val) for key, val in stats.errors.items()}
    total_errors = sum(list(stats.errorsCounter.values()))

    col1, col2, col3, col4 = st.columns(4)
    error_items = list(stats.errorsCounter.items())
    num_items = len(error_items)
    items_per_column = (num_items + 2) // 4  # +2 for rounding up when dividing by 3

    data = stats.translations
    for i in data:
        i.pop('Delete', None)

    df = pd.DataFrame(data)
    df = df[['name', 'State', 'Took', 'Engine Version', 'Reviewed', 'Amount Rows', 'Amount Errors', 'Errors %',
             'Amount OG Words', 'Amount Translated Words', 'Amount OG Characters', 'Amount Translated Characters']]
    df['Amount Errors'] = df['Amount Errors'].apply(lambda x: int(x) if not isnan(x) else x)
    df = df.round(2)

    with col1:
        st.metric('Amount Feedbacks', _format_number(stats.totalFeedbacks), '-', delta_color='off')
        st.metric('Original Characters Count', _format_number(stats.totalOgCharacters))
    with col2:
        st.metric('Total Checked Rows', _format_number(stats.totalChecked), '-', delta_color='off')
        st.metric('Translated Characters Count', _format_number(stats.totalTranslatedCharacters))
    with col3:
        st.metric('Total Errors Count (Rows)', _format_number(total_errors), f'{stats.errorPct}%', delta_color='off')
        st.metric('Original Words Count', _format_number(stats.amountOgWords))
    with col4:
        errors_pct = [err for err in df['Errors %'].to_list() if not isnan(err)]
        errors = [err for err in df['Amount Errors'].to_list() if not isnan(err)]
        st.metric(
            'Media Errors Count - Single Translation',
            statistics.median(errors),
            f'{statistics.median(errors_pct)}%',
            delta_color='off'
        )
        st.metric('Translated Words Count', _format_number(stats.amountTranslatedWords))
        prepositions_amount = stats.errorsCounter.get('Prepositions', 0)
        prepositions_in_pct = pct(prepositions_amount, total_errors)
        st.metric(
            'Prepositions',
            _format_number(prepositions_amount),
            f'{prepositions_in_pct}% of total errors',
            delta_color='off'
        )

    def display_metrics(column, items):
        with column:
            for key, amount in items:
                in_pct = pct(amount, total_errors)
                st.metric(key, _format_number(amount), f'{in_pct}% of total errors', delta_color='off')

    start_index = 0
    for col in [col1, col2, col3]:
        end_index = min(start_index + items_per_column, num_items)
        display_metrics(col, error_items[start_index:end_index])
        start_index = end_index

    st.header('Items')
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.divider()

    st.header('Samples')
    for k, v in samples.items():
        st.dataframe(v, use_container_width=True)
        st.divider()


def view_stats():
    stats_list = asyncio.run(_get_stats())

    for stats in stats_list:
        with st.expander(f'Version: {stats.version.value}', expanded=False):
            stats_for_version(stats)


def manage_existing():
    data = asyncio.run(_get_stats())
    for stats in data:
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
        st.divider()


if st.session_state["authentication_status"] is True:
    page_names_to_funcs = {
        'Subtitles Viewer': subtitles_viewer_from_db,
        'Engine Stats': view_stats,
        'Manage Existing Translations': manage_existing,
        'Costs Breakdown': costs_panel
    }
    app_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
    page_names_to_funcs[app_name]()
