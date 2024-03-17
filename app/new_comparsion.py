import asyncio
import logging
import re
from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import streamlit as st
from beanie import PydanticObjectId, Link
from beanie.odm.operators.find.comparison import In
from pydantic import BaseModel, Field, model_validator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from common.models.consts import ModelVersions
from common.config import mongodb_settings
from common.db import init_db
from common.consts import THREE_MINUTES
from common.models.users import User
from common.models.core import Project, ClientChannel, Client
from common.models.translation import (
    Translation, SRTBlock, TranslationFeedbackV2, MarkedRow
)
from common.utils import rows_to_srt, pct
from services.parsers.convertors import xml_to_srt
from services.parsers.format_handlers import srt_to_rows


def is_multi_modal(v: ModelVersions):
    return '0.3.' in v.value


def version_to_tuple(version: ModelVersions):
    parts = version.value[1:].split(".")
    return tuple(int(part) for part in parts)


def _show_sidebar(
        project_name: str,
        project_id: PydanticObjectId | str,
        target_language: str,
        version_to_translation: dict[ModelVersions, Translation],
        feedbacks: dict[ModelVersions, TranslationFeedbackV2] | None = None
):
    amount, by_error_count = None, None
    if feedbacks:
        by_error_count = {v: {} for v in feedbacks.keys()}
        for v, fbs in feedbacks.items():
            for row in fbs.marked_rows:
                err = row['error']
                if err not in by_error_count[v]:
                    by_error_count[v][err] = 1
                else:
                    by_error_count[v][err] += 1

        amount = list(feedbacks.values())[0].total_rows

    available_versions = list(version_to_translation.keys())

    with st.sidebar:
        info = {
            'Name': project_name,
            'Project ID': project_id,
            'Target Language': target_language,
            'Available Version': ', '.join([v.value for v in available_versions]),
            'Translations IDs': f'\n'.join(
                [f'{v.value}: {t.id}' for v, t in version_to_translation.items() if t is not None]
            )
        }
        if amount is not None:
            info['Amount Rows'] = amount

        for name, val in info.items():
            st.info(f'{name}: {val}')

        if by_error_count is not None:
            st.divider()
            st.header('Errors Count', divider='orange')
            for version, counts in by_error_count.items():
                fb = feedbacks[version]
                st.subheader(
                    f'{version.value} Errors - {len(fb.marked_rows)} ({pct(len(fb.marked_rows), amount)} %)',
                    divider='rainbow'
                )
                counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
                repr_obj = '  \n'.join([f'{err.value} - {count}' for err, count in counts.items()])
                st.info(repr_obj)

        if any([is_multi_modal(v) for v in available_versions]):
            st.divider()
            st.info(
                'Additional Variations Segment: This translation has a > v3 version, those contain several variations for each row. this tab will show you those.'
                'Green Marked Rows are the selected variation for each row.'
            )


tag_re = re.compile(r'<[^>]+>')


def strip_tags(xml_str):
    return tag_re.sub('', xml_str)


async def construct_comparison_df(
        version_to_translation: dict[ModelVersions, Translation],
        existing_feedbacks: dict[ModelVersions, TranslationFeedbackV2],
        extra_rows: list[SRTBlock] | None = None
):
    newest_v = max(list(version_to_translation.keys()), key=version_to_tuple)
    t = version_to_translation[newest_v]
    err_key = f'{newest_v.value} Error'

    error_cols, english_key = list(), f'English (from: {t.engine_version.value.lower()})'
    data = {
        'Time Stamp': [f'{row.start} --> {row.end}' for row in t.subtitles],
        english_key: [row.content for row in t.subtitles]
    }
    if extra_rows:
        data['Uploaded Subtitles'] = [strip_tags(row.content) for row in extra_rows]

    column_config = {english_key: st.column_config.TextColumn(width='large', disabled=True)}

    existing_errors_map: dict[str, dict[str, MarkedRow]] = dict()  # noqa
    labels = ['Gender Mistake', 'Time Tenses', 'Slang', 'Prepositions', 'Typo',
              'Name "as is"', 'Not fit in context', 'Plain Wrong Translation']
    from streamlit_utils import SelectBoxColumn

    column_config[err_key] = SelectBoxColumn(err_key, labels)
    data[newest_v.value] = [row.translations.selection if row.translations is not None else None for row in t.subtitles]
    column_config[newest_v.value] = st.column_config.TextColumn(width='large', disabled=True)
    data['Reviewer Marked'] = [
        False if row.translations and row.translations.scores is not None and row.translations.scores.IsValidTranslation == 1 else True
        for row in t.subtitles
    ]
    column_config['Is Marked'] = st.column_config.TextColumn(width='small', disabled=True)

    if any([row.speaker_gender is not None for row in t.subtitles]):
        data['Speaker Gender'] = [row.speaker_gender for row in t.subtitles]
        column_config['Speaker Gender'] = st.column_config.TextColumn(width='small', disabled=True)

    error_cols.append(err_key)

    for fb_v, fb in existing_feedbacks.items():
        if fb_v == newest_v:
            continue
        else:
            feedback = existing_feedbacks[newest_v]
            existing_errors_map[fb_v.value] = {row['original']: row for row in feedback.marked_rows}
            data[err_key] = [
                existing_errors_map[fb_v.value].get(content, {}).get('error', None) for content in data[english_key]
            ]

    maxlen = max([len(v) for v in data.values()])
    for k, v in data.items():
        if len(v) < maxlen:
            data[k] = v + ([None] * (maxlen - len(v)))

    df = pd.DataFrame(data)
    return df, error_cols, column_config, existing_errors_map, english_key


async def _update_results(
        project_name: str,
        edited_df: pd.DataFrame,
        error_cols: list[str],
        existing_errors_map: dict | DefaultDict,
        existing_feedbacks: dict[ModelVersions, TranslationFeedbackV2],
        version_to_translation: dict[ModelVersions, Translation],
        ENLGLISH_KEY: str
):
    updates_made = dict()
    for index, row in enumerate(edited_df.to_dict(orient='records'), start=1):
        for col in error_cols:
            err = row[col]
            v = ModelVersions(col.split(' ')[0])
            if err not in (None, 'None'):
                if v not in existing_errors_map:
                    existing_errors_map[v] = dict()
                if existing_errors_map[v].get(row[ENLGLISH_KEY]) is not None:
                    existing_errors_map[v][row[ENLGLISH_KEY]]['error'] = err
                else:
                    existing_errors_map[v][row[ENLGLISH_KEY]] = MarkedRow(
                        error=err, original=row[ENLGLISH_KEY],
                        translation=row[v.value], index=index
                    )
                if v not in updates_made:
                    updates_made[v] = 0
                updates_made[v] += 1

    for v in version_to_translation.keys():
        if existing := existing_feedbacks.get(v):
            existing.marked_rows = list(existing_errors_map[v].values())
            await existing.save()
        else:
            if v in existing_errors_map and len(existing_errors_map[v]) > 0:
                await TranslationFeedbackV2(
                    name=project_name,
                    version=v,  # noqa
                    total_rows=len(edited_df),
                    marked_rows=list(existing_errors_map[v].values())
                ).create()
    return updates_made


def _display_additional_variations(
        version_to_translation,
        ENLGLISH_KEY: str,
        existing_feedbacks: dict[ModelVersions, TranslationFeedbackV2]
):
    st.divider()
    with st.expander(label='Additional Variations', expanded=False):

        for v, t in version_to_translation.items():
            version_existing_feedback = existing_feedbacks.get(v)
            if is_multi_modal(v):
                st.subheader(f'{v.value} Variations')
                variations = [k.translations.available_versions() for k in t.subtitles if k.translations is not None]
                if len(variations) == 0:
                    pass
                else:
                    available_versions = max(
                        [k.translations.available_versions() for k in t.subtitles if k.translations is not None],
                        key=lambda x: len(x)
                    )

                    data = {
                        'Time Stamp': [f'{row.start} --> {row.end}' for row in t.subtitles],
                        ENLGLISH_KEY: [row.content for row in t.subtitles],
                        **{v: [] for v in available_versions},
                        'Error': [None] * len(t.subtitles)
                    }

                    def addNoneRow():
                        for r in available_versions:
                            data[r].append(None)

                    errors = {}
                    if version_existing_feedback:
                        for row in version_existing_feedback.marked_rows:
                            errors[row['original']] = row['error']

                    for idx, row in enumerate(t.subtitles):

                        if row.translations is not None:
                            row_versions = row.translations.available_versions()
                            for rev in available_versions:
                                if rev in row_versions:
                                    raw_string = getattr(row.translations, rev)
                                    if row.translations.selection == raw_string:
                                        raw_string = f':: {raw_string}'
                                    data[rev].append(raw_string)
                                else:
                                    data[rev].append(None)

                            if row.content in errors:
                                li: list = data['Error']
                                li[idx] = errors[row.content]
                                data['Error'] = li

                        else:
                            addNoneRow()

                    df = pd.DataFrame(data)

                    def highlight_green(cell):
                        if isinstance(cell, str) and cell.startswith(':: '):
                            return 'background-color: green'
                        return ''

                    styled = df.style.map(highlight_green)
                    conf = {
                        col: st.column_config.TextColumn(width='large', disabled=True) for col in available_versions
                    }
                    edited_df = st.data_editor(styled, use_container_width=True, column_config=conf)


async def get_compare_data(project_id) -> dict:
    project_id = PydanticObjectId(project_id)
    proj = await Project.get(project_id)
    ts, fbs = await asyncio.gather(
        Translation.find(Translation.project.id == project_id).to_list(),  # noqa
        TranslationFeedbackV2.find(TranslationFeedbackV2.name == proj.name).to_list()
    )
    return {'project': proj, 'translations': ts, 'feedbacks': fbs}


async def _newest_ever_compare_logic(project, translations: list[Translation], feedbacks, extra_rows=None):
    project_id, project_name = project.id, project.name
    version_to_translation: dict[ModelVersions, Translation] = {t.engine_version: t for t in translations}
    first = list(version_to_translation.values())[0]

    _show_sidebar(project_name, project_id, first.target_language, version_to_translation, feedbacks)

    df, error_cols, column_config, existing_errors_map, ENLGLISH_KEY = await construct_comparison_df(
        version_to_translation=version_to_translation,
        existing_feedbacks=feedbacks,
        extra_rows=extra_rows
    )
    edited_df = st.data_editor(df, column_config=column_config, use_container_width=True)

    _display_additional_variations(
        version_to_translation=version_to_translation,
        ENLGLISH_KEY=ENLGLISH_KEY,
        existing_feedbacks=feedbacks
    )

    if st.button('Submit'):
        updates_made = await _update_results(
            project_name=project_name,
            edited_df=edited_df,
            error_cols=error_cols,
            existing_errors_map=existing_errors_map,
            existing_feedbacks=feedbacks,
            version_to_translation=version_to_translation,
            ENLGLISH_KEY=ENLGLISH_KEY
        )
        st.success(f"Num Rows: {len(edited_df)}")
        for v, amount in updates_made.items():
            st.success(f"Num Mistakes {v.value}: {amount} ({pct(amount, len(edited_df))} %)")

        st.success('Successfully Saved Results to DB!')
        logging.info('finished and saved subtitles review')
        for v, translation in version_to_translation.items():
            srt = rows_to_srt(rows=translation.subtitles, target_language=translation.target_language)
            og_srt = rows_to_srt(rows=translation.subtitles, translated=False)
            st.download_button(
                label=f'Download {v.value} {translation.target_language.capitalize()} SRT', data=srt,
                file_name=f'{project_name}_{translation.target_language}_{v.value}.srt'
            )
            st.download_button(
                label=f'Download {v.value} English SRT', data=og_srt,
                file_name=f'{project_name}_en_{v.value}_OG.srt'
            )
        st.cache_data.clear()


@st.cache_resource
def connect_DB():
    _docs, _db = asyncio.run(
        init_db(mongodb_settings, [Translation, TranslationFeedbackV2, Project, Client, ClientChannel, User])
    )
    st.session_state['DB'] = _db
    return _docs, _db


def get_data(project_id: str):
    ret = asyncio.run(get_compare_data(project_id))
    return ret


def newest_ever_compare(project_id, extra_rows: list[SRTBlock] | None = None):
    if st.session_state.get('DB') is None:
        connect_DB()
    ret = get_data(str(project_id))
    proj, ts, fbs = ret['project'], ret['translations'], ret['feedbacks']
    existing_feedbacks: dict[ModelVersions, TranslationFeedbackV2] = {fb.version: fb for fb in fbs}
    return asyncio.run(
        _newest_ever_compare_logic(project=proj, translations=ts, feedbacks=existing_feedbacks, extra_rows=extra_rows)

    )


class TranslationLight(BaseModel):
    id: PydanticObjectId = Field(..., alias='_id')
    project: Link[Project]
    engine_version: ModelVersions = Field(default=ModelVersions.LATEST, alias='modelVersion')

    @property
    def project_id(self):
        if isinstance(self.project, Link):
            return self.project.ref.id
        return self.project.id  # noqa


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


async def _get_viewer_data() -> dict[str, list]:
    translations = await Translation.find(
        In(Translation.engine_version,
           [*[v for v in ModelVersions if is_multi_modal(v)], ModelVersions.V1.value, ])
    ).project(TranslationLight).to_list()
    pids = {t.project_id for t in translations}
    fbs, projects = await asyncio.gather(
        TranslationFeedbackV2.find_all().to_list(),
        Project.find(In(Project.id, list(pids))).project(Projection).to_list(),
    )
    return {'fbs': fbs, 'projects': projects, 'translations': translations}


@st.cache_data(ttl=THREE_MINUTES)
def get_viewer_data():
    if st.session_state.get('DB') is None:
        connect_DB()
    ret = asyncio.run(_get_viewer_data())
    return ret


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

    def parse_file(f):
        string_data = _parse_file_upload(f)
        if f.name.endswith('nfs') or f.name.endswith('xml'):
            string_data = xml_to_srt(string_data)
        rows = srt_to_rows(string_data)
        return rows

    with st.form('forma'):
        chosenObj = st.selectbox('Choose Translation', options=list(new.values()), format_func=format_name)
        revision = st.file_uploader('Additional Subtitles', type=['srt', 'xml', 'txt', 'nfs'])
        submit = st.form_submit_button('Get')
        if submit:
            _id = [k for k, v in new.items() if v == chosenObj][0]
            st.session_state['projectId'] = _id
            if revision:
                st.session_state['file'] = parse_file(revision)

    with st.form('forma2'):
        chosenObj = st.selectbox('Update Existing Review', options=list(existing.values()), format_func=format_name)
        revision = st.file_uploader('Additional Subtitles', type=['srt', 'xml', 'txt', 'nfs'])
        submit = st.form_submit_button('Get')
        if submit:
            _id = [k for k, v in existing.items() if v == chosenObj][0]
            st.session_state['projectId'] = _id
            if revision:
                st.session_state['file'] = parse_file(revision)

    if 'projectId' in st.session_state:
        project_id: str = st.session_state['projectId']
        newest_ever_compare(project_id, st.session_state.get('file', None))
