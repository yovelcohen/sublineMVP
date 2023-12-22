import asyncio
import datetime
import io
import time
import zipfile
from collections import defaultdict
from math import isnan

import openai
import pandas as pd
import streamlit as st

import logging
import streamlit.logger

from beanie import PydanticObjectId, Document
from beanie.exceptions import CollectionWasNotInitialized
from beanie.odm.operators.find.comparison import In
from beanie.odm.operators.find.logical import Or
from pydantic import BaseModel, model_validator

from common.consts import SrtString
from common.models.core import Translation, SRTBlock, Ages, Genres, TranslationStates
from common.config import settings
from common.db import init_db
from common.utils import rows_to_srt
from services.recovery import recover_translation
from services.runner import run_translation
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
    name: str = ''
    total_rows: int
    marked_rows: list[dict]
    duration: datetime.timedelta | None = None

    @property
    def error_pct(self):
        return round(((len(self.marked_rows) / self.total_rows) / 100), 2)


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
    'ir': 'In Revision',
    'sa': 'Smart Audit',
    'd': 'Done',
    'f': 'Failed'
}
TEN_K, MILLION, BILLION = 10_000, 1_000_000, 1_000_000_000

if not st.session_state.get('DB'):
    logger.info('initiating DB Connection And collections')
    db, docs = asyncio.run(init_db(settings, [Translation, TranslationFeedback]))
    st.session_state['DB'] = db
    logger.info('Finished DB Connection And collections init process')


def string_to_enum(value: str, enum_class):
    for enum_member in enum_class:
        if enum_member.value == value or enum_member.value == value.lower():
            return enum_member
    raise ValueError(f"{value} is not a valid value for {enum_class.__name__}")


async def translate(
        _name: str,
        file: str,
        _target_language: str,
        _format: str,
        main_genre: str,
        age: Ages,
        additional_genres: list[str] = None
):
    t1 = time.time()
    logger.debug('starting translation request')
    task = Translation(target_language=_target_language, source_language='English', subtitles=[],
                       project_id=PydanticObjectId(), name=_name, main_genre=string_to_enum(main_genre, Genres),
                       age=age)
    if additional_genres:
        task.additional_genres = [string_to_enum(g, Genres) for g in additional_genres]

    await task.save()
    try:
        with st.status('Translating...'):
            ret: SubtitlesResults = await run_translation(task=task, blob_content=file, mime_type=_format)
            st.session_state['name'] = ret
    except openai.APITimeoutError as e:
        st.error('Translation Failed Due To OpenAI API Timeout, Please Try Again Later')
        await task.delete()
        raise e

    t2 = time.time()
    task.took = t2 - t1
    await task.save()
    logger.debug('finished translation, took %s seconds', t2 - t1)
    st.session_state['bar'].progress(100, text='Done!')


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


async def save_to_db(rows, df, name, last_row: SRTBlock, existing_feedback: TranslationFeedback | None = None):
    def filter_row(row):
        ke = 'V1 Error 1' if 'V1 Error 1' in row else 'Error 1'
        return row[ke] not in ('None', None)

    rows = [r for r in rows if filter_row(r)]
    if not existing_feedback:
        params = dict(marked_rows=rows, total_rows=len(df), name=name, duration=last_row.end)
        try:
            existing_feedback = TranslationFeedback(**params)
        except CollectionWasNotInitialized as e:
            await init_db(settings, [TranslationFeedback])
            existing_feedback = TranslationFeedback(**params)
    else:
        existing_feedback.marked_rows = rows
    await existing_feedback.save()
    return existing_feedback


async def get_feedback_by_name(name) -> TranslationFeedback | None:
    await init_db(settings, [TranslationFeedback])
    return await TranslationFeedback.find({'name': name}).first_or_none()


def _display_comparison_panel(name, subtitles, rows: list[SRTBlock], target_lang):
    rows = list(rows)
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
    if 'Glix Translation 2' in subtitles:
        select_cols.extend(['V2 Error 1', 'V2 Error 2'])
        config.update(
            {'Glix Translation 2': st.column_config.TextColumn(width='large'),
             'V2 Error 1': select_box_col('V2 Error 1'),
             'V2 Error 2': select_box_col('V2 Error 2')}
        )

    max_length = max(len(lst) for lst in subtitles.values())
    # Pad shorter lists with None
    for key in subtitles:
        subtitles[key] += [None] * (max_length - len(subtitles[key]))

    existing_feedback = asyncio.run(get_feedback_by_name(name))
    df = pd.DataFrame(subtitles)

    def get_row_feedback(row, k=1):
        if k == 1:
            ke = 'V1 Error 1' if 'V1 Error 1' in row else 'Error 1'
        else:
            ke = 'V2 Error 1' if 'V2 Error 1' in row else 'Error 2'
        return row[ke]

    TRANSLATION_KEY = 'Glix Translation 1'
    if existing_feedback:
        logging.info(f'Found existing feedback, updating df, amount marked rows: {len(existing_feedback.marked_rows)}')
        text_to_feedback1 = {row[TRANSLATION_KEY]: get_row_feedback(row, 1) for row in
                             existing_feedback.marked_rows}
        text_to_feedback2 = {row[TRANSLATION_KEY]: get_row_feedback(row, 2) for row in
                             existing_feedback.marked_rows}
        df['Error 1'] = df[TRANSLATION_KEY].apply(lambda x: text_to_feedback1.get(x, None))
        df['Error 2'] = df[TRANSLATION_KEY].apply(lambda x: text_to_feedback2.get(x, None))
        map_ = {row[TRANSLATION_KEY]: row for row in existing_feedback.marked_rows}
    else:
        for col in select_cols:
            df[col] = pd.Series([None] * len(df), index=df.index)
        map_ = {}

    edited_df = st.data_editor(df, key='df', column_config=config, use_container_width=True)

    if st.button('Submit'):
        marked_rows = []
        for row in edited_df.to_dict(orient='records'):
            for col in select_cols:
                if row[col] not in ('None', None):
                    if row_to_update := map_.get(row[TRANSLATION_KEY]):
                        row_to_update[col] = row[col]
                        marked_rows.append(row_to_update)
                    else:
                        marked_rows.append(row)
                    break

        feedback_obj = asyncio.run(
            save_to_db(marked_rows, edited_df, name=name, last_row=last_row, existing_feedback=existing_feedback)
        )
        st.write('Successfully Saved Results to DB!')
        st.info(f"Num Rows: {len(edited_df)}\n Num Mistakes: {len(marked_rows)} ({feedback_obj.error_pct} %))")
        logging.info('finished and saved subtitles review')

    if st.button('Export Results'):
        blob = edited_df.to_csv(header=True)
        download_button(name=name, srt_string1=rows_to_srt(rows=rows, translated=True, target_language=target_lang))
        st.download_button('Export', data=blob, file_name=f'{name}.csv', type='primary')


async def _get_translations_df() -> list[dict]:
    if not st.session_state.get('DB'):
        db, docs = asyncio.run(init_db(settings, [Translation]))
        st.session_state['DB'] = db
    projs = await Translation.find(Translation.name != None).to_list()

    def get_took(t):
        minutes, seconds = divmod(t, 60)
        return "{:02d}:{:02d}".format(int(minutes), int(seconds))

    def get_cost(usage):
        total_cost = 0
        for key, cost in (('prompt_tokens', 0.03), ('completion_tokens', 0.06)):
            if key in usage:
                num = usage[key]
                thousands, remainder = num // 1000, num % 1000
                total_cost += num * thousands
                total_cost += (remainder / 1000) * cost
        return total_cost

    return [
        {'name': proj.name,
         'Amount Rows': len(proj.subtitles),
         'State': STATES_MAP[proj.state.value],
         'Took': get_took(proj.took),
         'Delete': False,
         'Amount OG Characters': sum([len(r.content) for r in proj.subtitles]),
         'Amount Translated Characters': sum(
             [len(r.translations.content) for r in proj.subtitles if r.translations is not None]
         ),
         'Amount OG Words': sum([len(r.content.split()) for r in proj.subtitles]),
         'Amount Translated Words': sum(
             [len(r.translations.content.split()) for r in proj.subtitles if r.translations is not None]
         ),
         'token_cost': get_cost(proj.tokens_cost)
         }
        for proj in projs
    ]


async def _get_stats():
    await init_db(settings, [TranslationFeedback])
    feedbacks, sum_checked_rows, all_names, by_error = [], 0, set(), defaultdict(list)

    q = TranslationFeedback.find_all()
    count, data = await asyncio.gather(q.count(), _get_translations_df())
    name_to_count = {}
    async for feedback in q:
        feedbacks.extend(feedback.marked_rows)
        sum_checked_rows += feedback.total_rows
        all_names.add(feedback.name)
        name_to_count[feedback.name] = len(feedback.marked_rows)
        for row in feedback.marked_rows:
            key = 'V1 Error 1' if 'V1 Error 1' in row else 'Error 1'
            key2 = 'V2 Error 1' if 'V2 Error 1' in row else 'Error 2'
            for k in (key, key2):
                if k in row and row[k] not in ('None', None):
                    row['Name'] = feedback.name
                    by_error[row[k]].append(row)

    for translation in data:
        if translation['name'] in all_names and translation['State'] == 'Done':
            translation['Reviewed'] = True
            amount_errors = name_to_count[translation['name']]
            translation['Amount Errors'] = int(amount_errors)
            translation['Errors %'] = round(pct(amount_errors, translation['Amount Rows']), 1)
        else:
            translation['Reviewed'] = False

    stats = Stats(
        totalChecked=sum_checked_rows,
        totalFeedbacks=count,
        errorsCounter={key: len(val) for key, val in by_error.items()},
        errors=by_error,
        errorPct=round((len(feedbacks) / sum_checked_rows) * 100, 2),
        totalOgCharacters=sum([row['Amount OG Characters'] for row in data]),
        totalTranslatedCharacters=sum([row['Amount Translated Characters'] for row in data]),
        amountOgWords=sum([row['Amount OG Words'] for row in data]),
        amountTranslatedWords=sum([row['Amount Translated Words'] for row in data]),
        translations=data
    )
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
    q = Translation.find(In(Translation.name, to_delete))
    ack = await q.delete()
    return ack.deleted_count


##################### APP FUNCTIONS #####################
def translate_form():
    if st.session_state.get('stage', 0) != 1:
        with st.form("Translate"):
            name = st.text_input("Name")
            uploaded_file = st.file_uploader("Upload a file", type=["srt", "xml", 'nfs', 'txt'])
            source_language = st.selectbox("Source Language", ["English", "Hebrew", 'French', 'Arabic', 'Spanish'])
            target_language = st.selectbox("Target Language", ["Hebrew", 'French', 'Arabic', 'Spanish'])
            age = st.selectbox('Age', options=list(AGES_MAP.keys()), index=4)
            genre1 = st.selectbox('Category', options=list(GENRES_MAP.keys()), key='genre1')
            additionalGenres = st.multiselect('Optional Additional Genres', options=(GENRES_MAP.keys()),
                                              key='genreExtra')

            assert source_language != target_language
            submitted = st.form_submit_button("Translate")
            if submitted:
                bar = st.progress(0, 'Parsing File...')
                st.session_state['bar'] = bar
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

                asyncio.run(
                    translate(
                        _name=name, file=string_data, _target_language=target_language, age=AGES_MAP[age],
                        main_genre=GENRES_MAP[genre1], additional_genres=[GENRES_MAP[g] for g in additionalGenres],
                        _format=st.session_state['format']
                    )
                )

        if st.session_state.get('name', False):
            results: SubtitlesResults = st.session_state['name']
            subtitles = {
                'Original Language': [row.content for row in results.rows],
                'Glix Translation 1': [row.translations.content for row in results.rows if
                                       row.translations is not None],
            }
            _display_comparison_panel(name=name, subtitles=subtitles, rows=results.rows, target_lang=target_language)


def recover_form():
    db, docs = asyncio.run(init_db(settings, [Translation, TranslationFeedback]))
    st.session_state['DB'] = db

    translations = asyncio.run(
        Translation.find(Translation.name != None, Translation.state != TranslationStates.DONE.value).  # noqa
        project(Projection).to_list()
    )

    name_to_id = {proj.name: proj.id for proj in translations}
    with st.form('forma'):
        chosen_name = st.selectbox('Choose Translation', options=list(name_to_id.keys()))
        submit = st.form_submit_button('Get')
        if submit:
            chosen_id = name_to_id[chosen_name]
            translation = asyncio.run(Translation.get(chosen_id))
            st.session_state['name'] = asyncio.run(recover_translation(translation))

    if st.session_state.get('name', False):
        results: SubtitlesResults = st.session_state['name']
        subtitles = {
            'Original Language': [row.content for row in results.rows],
            'Glix Translation 1': [row.translations.content for row in results.rows if row.translations is not None],
        }
        _display_comparison_panel(name=results.translation_obj.name,
                                  subtitles=subtitles, rows=results.rows,
                                  target_lang=results.translation_obj.target_language)


def subtitles_viewer_from_db():
    db, docs = asyncio.run(init_db(settings, [Translation, TranslationFeedback]))
    st.session_state['DB'] = db

    existing_feedbacks = asyncio.run(TranslationFeedback.find_all().project(Projection).to_list())
    names = [d.name for d in existing_feedbacks]
    translations = asyncio.run(
        Translation.find(Or(Translation.name != None, In(Translation.name, names))).  # noqa
        find(Translation.state == TranslationStates.DONE.value).project(Projection).to_list()
    )
    existing, new = dict(), dict()
    for proj in translations:
        if proj.name in names:
            existing[proj.name] = proj.id
        else:
            new[proj.name] = proj.id
    _all = {**existing, **new}

    def accept_form():
        chosen_id = _all[chosen_name]
        translation = asyncio.run(Translation.get(chosen_id))
        rows = sorted(list(translation.subtitles), key=lambda x: x.index)
        st.session_state['rows'] = rows
        st.session_state['targetLang'] = translation.target_language
        subtitles = {
            'Original Language': [row.content for row in rows],
            'Glix Translation 1': [row.translations.content for row in rows if row.translations is not None],
        }
        revs = [row.translations.revision for row in translation.subtitles if row.translations is not None]
        if not all([r is None for r in revs]):
            subtitles['Glix Translation 2'] = revs
        st.session_state['tName'] = translation.name
        st.session_state['subtitles'] = subtitles
        st.session_state['lastRow'] = rows[-1]

    with st.form('forma'):
        chosen_name = st.selectbox('Choose Translation', options=list(new.keys()))
        submit = st.form_submit_button('Get')
        if submit:
            accept_form()

    with st.form('forma2'):
        chosen_name = st.selectbox('Update Existing Review', options=list(existing.keys()))
        submit = st.form_submit_button('Get')
        if submit:
            accept_form()

    if 'subtitles' in st.session_state:
        with st.expander("Download SRT", expanded=False):
            st.download_button(data=rows_to_srt(rows=st.session_state['rows'], translated=True,
                                                target_language=st.session_state['targetLang']),
                               file_name=f'{st.session_state["tName"]}.srt', label='Download SRT')

        _display_comparison_panel(
            name=chosen_name, subtitles=st.session_state['subtitles'],
            rows=st.session_state['rows'], target_lang=st.session_state['targetLang']
        )


def view_stats():
    stats = asyncio.run(_get_stats())
    samples = {key: pd.DataFrame(val) for key, val in stats.errors.items()}
    total_errors = sum(list(stats.errorsCounter.values()))

    col1, col2, col3, col4 = st.columns(4)
    error_items = list(stats.errorsCounter.items())
    num_items = len(error_items)
    items_per_column = (num_items + 2) // 4  # +2 for rounding up when dividing by 3

    with col1:
        st.metric('Total Checked Rows', _format_number(stats.totalChecked))
        st.metric('Original Characters Count', _format_number(stats.totalOgCharacters))
    with col2:
        st.metric('Total Errors Count', _format_number(total_errors))
        st.metric('Translated Characters Count', _format_number(stats.totalTranslatedCharacters))
    with col3:
        st.metric('Error Percentage', f'{stats.errorPct}%')
        st.metric('Original Words Count', _format_number(stats.amountOgWords))
    with col4:
        st.metric('Amount Feedbacks', _format_number(stats.totalFeedbacks))
        st.metric('Translated Words Count', _format_number(stats.amountTranslatedWords))
        prepositions_amount = stats.errorsCounter.get('Prepositions', 0)
        prepositions_in_pct = pct(prepositions_amount, total_errors)
        st.metric('Prepositions', _format_number(prepositions_amount), f'{prepositions_in_pct}% of total errors',
                  delta_color='off')

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

    st.divider()
    st.header('Items')
    data = stats.translations
    for i in data:
        i.pop('Delete', None)

    df = pd.DataFrame(data)
    df = df[['name', 'State', 'Took', 'Reviewed', 'Amount Rows', 'Amount Errors', 'Errors %',
             'Amount OG Words', 'Amount Translated Words', 'Amount OG Characters',
             'Amount Translated Characters', 'token_cost']]
    df['Amount Errors'] = df['Amount Errors'].apply(lambda x: int(x) if not isnan(x) else x)
    df = df.round(2)

    st.dataframe(df, hide_index=True, use_container_width=True)
    st.divider()

    st.header('Samples')
    for k, v in samples.items():
        with st.expander(k, expanded=False):
            st.dataframe(v, use_container_width=True)
        st.divider()


def manage_existing():
    data = asyncio.run(_get_stats())
    data = [
        {'name': row['name'], 'Amount Rows': row['Amount Rows'], 'Delete': row['Delete'], 'State': row['State'],
         'Reviewed': row['Reviewed']}
        for row in data.translations
    ]
    edited_df = st.data_editor(pd.DataFrame(data),
                               column_config={'Reviewed': st.column_config.SelectboxColumn(disabled=True)},
                               use_container_width=True)

    if st.button('Delete'):
        rows = edited_df.to_dict(orient='records')
        to_delete = [proj['name'] for proj in rows if proj['Delete'] is True]
        if to_delete:
            ack = asyncio.run(_delete_docs(to_delete=to_delete))
            logger.info(f'Deleted {ack} translation projects')
            st.success(f'Successfully Deleted {ack} Projects, Refresh Page To See Changes')


page_names_to_funcs = {
    "Translate": translate_form,
    'Subtitles Viewer': subtitles_viewer_from_db,
    'Recover Failed Translation': recover_form,
    'Engine Stats': view_stats,
    'Manage Existing Translations': manage_existing
}
app_name = st.sidebar.selectbox("Choose app", page_names_to_funcs.keys())
page_names_to_funcs[app_name]()
