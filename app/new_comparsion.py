import asyncio
import datetime
import logging
from typing import Self

import pandas as pd
import streamlit as st
import typing_extensions
from beanie import PydanticObjectId, Document
from pydantic import Field

from common.config import mongodb_settings
from common.db import init_db
from common.models.core import Project
from common.models.translation import Translation, ModelVersions, MULTI_MODAL_MODELS
from common.utils import rows_to_srt


def SelectBoxColumn(label, labels):
    return st.column_config.SelectboxColumn(
        width='medium', label=label, required=False, options=labels
    )


class MarkedRow(typing_extensions.TypedDict):
    error: str
    original: str
    translation: str | None


class TranslationFeedback(Document):
    name: str = ''
    version: ModelVersions = Field(default=ModelVersions.V1, alias='engine_version')
    total_rows: int
    marked_rows: list[dict]
    duration: datetime.timedelta | None = None

    @property
    def error_pct(self):
        return round(((len(self.marked_rows) / self.total_rows) / 100), 2)


class TranslationFeedbackV2(Document):
    name: str
    version: ModelVersions = Field(default=ModelVersions.V1, alias='engine_version')
    total_rows: int
    marked_rows: list[MarkedRow]
    duration: datetime.timedelta | None = None

    @property
    def error_pct(self):
        return round(((len(self.marked_rows) / self.total_rows) / 100), 2)

    @classmethod
    def migrate_v1_obj(cls, obj: TranslationFeedback) -> Self:
        rows = []
        for row in obj.marked_rows:
            if 'V1 Error 1' in row:
                k = row['V1 Error 1']
            elif 'Error 1' in row:
                k = row['Error 1']
            else:
                raise ValueError('Could not find error key in row')
            rows.append(MarkedRow(error=k, original=row['Original Language'], translation=row['Glix Translation 1']))

        return cls(
            name=obj.name,
            version=obj.version,
            total_rows=obj.total_rows,
            marked_rows=rows,
            duration=obj.duration
        )


OriginalContent: str


def pct(a, b):
    if a == 0 or b == 0:
        return 0
    return round((a / b) * 100, 2)


def _show_sidebar(project_name, target_language, available_versions):
    with st.sidebar:
        info = {
            'Name': project_name,
            'Target Language': target_language,
            'Available Version': ', '.join([v.value for v in available_versions])
        }
        for name, val in info.items():
            st.info(f'{name}: {val}')

        if any([v in MULTI_MODAL_MODELS for v in available_versions]):
            st.divider()
            st.info(
                'Additional Variations Segment: This translation has a > v3 version, those contain several variations for each row. this tab will show you those.'
                'Green Marked Rows are the selected variation for each row.'
            )


async def construct_comparison_df(project_name, version_to_translation):
    first = list(version_to_translation.values())[0]
    error_cols = list()
    data = {
        'Time Stamp': [f'{row.start} --> {row.end}' for row in first.subtitles],
        'English': [row.content for row in first.subtitles]
    }

    existing_feedbacks: dict[ModelVersions, TranslationFeedbackV2] = {
        fb.version: fb for fb in await TranslationFeedbackV2.find(TranslationFeedbackV2.name == project_name).to_list()
    }

    columnConfig = {
        'English': st.column_config.TextColumn(width='large', disabled=True)
    }

    existing_errors_map: dict[OriginalContent, MarkedRow] = dict()
    labels = ['Gender Mistake', 'Time Tenses', 'Names', 'Slang', 'Prepositions',
              'Name "as is"', 'not fit in context', 'Plain Wrong Translation']
    for v, t in version_to_translation.items():
        err_key = f'{v.value} Error'
        columnConfig[err_key] = SelectBoxColumn(err_key, labels)
        version_translation = [
            row.translations.selection if row.translations is not None else None for row in t.subtitles
        ]
        data[v.value] = version_translation
        columnConfig[v.value] = st.column_config.TextColumn(width='large', disabled=True)
        error_cols.append(err_key)

        if v in existing_feedbacks:
            feedback = existing_feedbacks[v]
            existing_errors_map[v] = {
                row['original']: row for row in feedback.marked_rows
            }
            data[err_key] = [
                existing_errors_map[v].get(content, {}).get('error', None) for content in data['English']
            ]
        else:
            data[err_key] = [None] * len(data[v.value])

    maxlen = max([len(v) for v in data.values()])
    for k, v in data.items():
        if len(v) < maxlen:
            data[k] = v + ([None] * (maxlen - len(v)))

    df = pd.DataFrame(data)
    return df, error_cols, columnConfig, existing_errors_map, existing_feedbacks


async def _update_results(
        project_name, edited_df, error_cols, existing_errors_map, existing_feedbacks, version_to_translation
):
    updates_made = dict()
    for row in edited_df.to_dict(orient='records'):
        for col in error_cols:
            err = row[col]
            v = ModelVersions(col.split(' ')[0])
            if err not in (None, 'None'):
                if v not in existing_errors_map:
                    existing_errors_map[v] = dict()
                if existing_errors_map[v].get(row['English']) is not None:
                    existing_errors_map[v][row['English']]['error'] = err
                else:
                    existing_errors_map[v][row['English']] = MarkedRow(
                        error=err, original=row['English'], translation=row[v.value]
                    )
                if v not in updates_made:
                    updates_made[v] = 0
                updates_made[v] += 1

    for v in version_to_translation.keys():
        if existing := existing_feedbacks.get(v):
            existing.marked_rows = list(existing_errors_map[v].values())
            await existing.save()
        else:
            await TranslationFeedbackV2(
                name=project_name,
                version=v,
                total_rows=len(edited_df),
                marked_rows=list(existing_errors_map[v].values())
            ).create()
    return updates_made


def _display_additional_variations(version_to_translation):
    if any([v in MULTI_MODAL_MODELS for v in version_to_translation.keys()]):
        st.divider()
        with st.expander(label='Additional Variations', expanded=False):
            # TODO: Add explaining on sidebar
            for v, t in version_to_translation.items():
                if v in MULTI_MODAL_MODELS:
                    st.subheader(f'{v.value} Variations')
                    available_versions = max(
                        [k.translations.available_versions() for k in t.subtitles if k.translations is not None],
                        key=lambda x: len(x)
                    )

                    data = {
                        'Time Stamp': [f'{row.start} --> {row.end}' for row in t.subtitles],
                        'English': [row.content for row in t.subtitles],
                        **{v: [] for v in available_versions}
                    }

                    def addNoneRow():
                        for r in available_versions:
                            data[r].append(None)

                    for row in t.subtitles:
                        if row.translations is not None:
                            for rev in available_versions:
                                row_versions = row.translations.available_versions()
                                if rev in row_versions:
                                    raw_string = row.translations.get_suggestion(rev)
                                    if row.translations.selection == raw_string:
                                        raw_string = f':: {raw_string}'
                                    data[rev].append(raw_string)
                                else:
                                    data[rev].append(None)
                        else:
                            addNoneRow()
                    max_len = max([len(v) for v in data.values()])
                    data['Correct'] = [None] * max_len
                    df = pd.DataFrame(data)

                    def highlight_green(cell):
                        if isinstance(cell, str) and cell.startswith(':: '):
                            return 'background-color: green'
                        return ''

                    styled = df.style.applymap(highlight_green)
                    conf = {
                        **{col: st.column_config.TextColumn(width='large', disabled=True)
                           for col in available_versions},
                        **{'Correct': SelectBoxColumn('Correct', ['Yes', 'No'])}
                    }
                    edited_df = st.data_editor(styled, use_container_width=True, column_config=conf)


async def _newest_ever_compare(project_id):
    project_id = PydanticObjectId(project_id)
    version_to_translation: dict[ModelVersions, Translation] = {
        t.engine_version: t for t in await Translation.find(Translation.project.id == project_id).to_list()  # noqa
    }
    first = list(version_to_translation.values())[0]

    project_name = (await Project.get(project_id)).name

    _show_sidebar(project_name, first.target_language, version_to_translation.keys())

    df, error_cols, columnConfig, existing_errors_map, existing_feedbacks = await construct_comparison_df(
        project_name, version_to_translation
    )
    edited_df = st.data_editor(df, column_config=columnConfig, use_container_width=True)

    _display_additional_variations(version_to_translation)

    if st.button('Submit'):
        updates_made = await _update_results(
            project_name=project_name, edited_df=edited_df, error_cols=error_cols,
            existing_errors_map=existing_errors_map, existing_feedbacks=existing_feedbacks,
            version_to_translation=version_to_translation
        )
        st.success(f"Num Rows: {len(edited_df)}")
        for v, amount in updates_made.items():
            st.success(f"Num Mistakes {v.value}: {amount} ({pct(amount, len(edited_df))} %)")

        st.success('Successfully Saved Results to DB!')
        logging.info('finished and saved subtitles review')
        for v, translation in version_to_translation.items():
            srt = rows_to_srt(rows=translation.subtitles, target_language=translation.target_language)
            st.download_button(
                label=f'Download {v.value} SRT', data=srt,
                file_name=f'{project_name}_{translation.target_language}_{v.value}.srt'
            )


def newest_ever_compare(project_id):
    return asyncio.run(_newest_ever_compare(project_id))


async def _migrate_feedbacks():
    await init_db(mongodb_settings, [TranslationFeedbackV2, TranslationFeedback])
    feedbacks = await TranslationFeedback.find().to_list()
    objs = [TranslationFeedbackV2.migrate_v1_obj(o) for o in feedbacks]
    await TranslationFeedbackV2.insert_many(objs)
