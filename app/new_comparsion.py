import asyncio
import dataclasses
import datetime
import logging
from collections import defaultdict
from typing import Self, TypedDict

import pandas as pd
import streamlit as st
import typing_extensions
from beanie import PydanticObjectId, Document
from beanie.exceptions import CollectionWasNotInitialized
from pydantic import Field

from common.config import mongodb_settings
from common.db import init_db
from common.models.core import Project
from common.models.translation import Translation, SRTBlock, ModelVersions

labels = ['Gender Mistake', 'Time Tenses', 'Names', 'Slang', 'Prepositions',
          'Name "as is"', 'not fit in context', 'Plain Wrong Translation']

select_box_col = lambda label: st.column_config.SelectboxColumn(
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


async def get_feedback_by_name(name, v=ModelVersions.V1) -> TranslationFeedbackV2 | None:
    await init_db(mongodb_settings, [TranslationFeedbackV2])
    return await TranslationFeedbackV2.find(TranslationFeedbackV2.name == name,
                                            TranslationFeedbackV2.version == v.value).first_or_none()


async def save_to_db(rows, df, name, last_row: SRTBlock, existing_feedback: TranslationFeedbackV2 | None = None,
                     version=ModelVersions.V1):
    def filter_row(row):
        ke = 'V1 Error 1' if 'V1 Error 1' in row else 'Error 1'
        return row[ke] not in ('None', None)

    rows = [r for r in rows if filter_row(r)]
    if not existing_feedback:
        params = dict(marked_rows=rows, total_rows=len(df), name=name, duration=last_row.end, version=version)
        try:
            existing_feedback = await TranslationFeedbackV2(**params).create()
        except CollectionWasNotInitialized as e:
            await init_db(mongodb_settings, [TranslationFeedbackV2])
            existing_feedback = await TranslationFeedbackV2(**params).create()
    else:
        before = len(existing_feedback.marked_rows)
        existing_feedback.marked_rows = rows
        if not existing_feedback.duration:
            existing_feedback.duration = last_row.end

        await existing_feedback.replace()
        logging.info(f'Updated translation feedback, amount rows before: {before}, after: {len(rows)} ')

    return existing_feedback


def get_comparison_from_translations(translations: list[Translation]):
    by_v = {t.engine_version: t for t in translations}
    rows = translations[0].subtitles
    versions = 0
    data = {
        'Time Stamp': [f'{row.start} --> {row.end}' for row in rows],
        'Original Language': [row.content for row in rows],
    }
    for version, translation in by_v.items():
        data[version] = [row.translations.selection if row.translations is not None else None for row in rows]
        versions += 1
    return data, versions


def one_panel(translation: Translation):
    rows = translation.subtitles

    df = pd.DataFrame(
        {
            'Time Stamp': [f'{row.start} --> {row.end}' for row in rows],
            'Original Language': [row.content for row in rows],
            'Glix Translation 1': [row.translations.selection if row.translations is not None else None for row in rows]
        }
    )

    return df, translation.engine_version, translation.subtitles[-1]


def get_row_feedback(row, k=1):
    if k == 1:
        ke = 'V1 Error 1' if 'V1 Error 1' in row else 'Error 1'
    else:
        ke = 'V2 Error 1' if 'V2 Error 1' in row else 'Error 2'
    return row[ke]


select_cols = ['Error 1', 'Error 2']


async def even_new_compare(project_id):
    project_id = PydanticObjectId(project_id)
    project = await Project.get(project_id)
    translations = await Translation.find(Translation.project.id == project_id).to_list()  # noqa
    data, amountVersions = get_comparison_from_translations(translations=translations)

    with st.sidebar:
        info = {'Name': project.name, 'Target Language': translations[0].target_language,
                'Available Version': [obj.engine_version.value for obj in translations]}
        for name, val in info.items():
            st.info(f'{name}: {val}')

    columns = st.columns(amountVersions + 2)


async def new_compare(project_id):
    project_id = PydanticObjectId(project_id)
    project = await Project.get(project_id)
    translations = await Translation.find(Translation.project.id == project_id).to_list()  # noqa
    dfs = [one_panel(t) for t in translations]

    with st.sidebar:
        info = {'Name': project.name, 'Target Language': translations[0].target_language,
                'Available Version': [obj.engine_version.value for obj in translations]}
        for name, val in info.items():
            st.info(f'{name}: {val}')

    for df, version, last_row in dfs:
        TRANSLATION_KEY = 'Glix Translation 1'
        with st.expander(expanded=False, label=f'Engine Version: {version}'):
            config = {
                'Original Language': st.column_config.TextColumn(width='large'),
                'Glix Translation 1': st.column_config.TextColumn(width='large'),
                'Error 1': select_box_col('Error 1'),
                'Error 2': select_box_col('Error 2'),
            }
            existing_feedback = await get_feedback_by_name(project.name, v=version)
            if existing_feedback is not None:
                logging.info(
                    f'Found existing feedback, updating df, amount marked rows: {len(existing_feedback.marked_rows)}'
                )
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

            edited_df = st.data_editor(df, key=f'df{version.value}', column_config=config, use_container_width=True)

            if st.button('Submit', key=f'submit{version.value}'):
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
                    save_to_db(marked_rows, edited_df, name=project.name, last_row=last_row,
                               existing_feedback=existing_feedback, version=version)
                )
                st.write('Successfully Saved Results to DB!')
                st.info(f"Num Rows: {len(edited_df)}\n Num Mistakes: {len(marked_rows)} ({feedback_obj.error_pct} %))")
                logging.info('finished and saved subtitles review')


OriginalContent: str


def pct(a, b):
    if a == 0 or b == 0:
        return 0
    return round((a / b) * 100, 2)


async def _newest_ever_compare(project_id):
    project_id = PydanticObjectId(project_id)
    version_to_translation: dict[ModelVersions, Translation] = {
        t.engine_version: t for t in await Translation.find(Translation.project.id == project_id).to_list()  # noqa
    }
    first = list(version_to_translation.values())[0]
    error_cols = list()
    project_name = (await Project.get(project_id)).name

    with st.sidebar:
        info = {
            'Name': project_name,
            'Target Language': first.target_language,
            'Available Version': ', '.join([v.value for v in version_to_translation.keys()])
        }
        for name, val in info.items():
            st.info(f'{name}: {val}')

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
    for v, t in version_to_translation.items():
        err_key = f'{v.value} Error'
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
        columnConfig[err_key] = select_box_col(err_key)

    maxlen = max([len(v) for v in data.values()])
    for k, v in data.items():
        if len(v) < maxlen:
            data[k] = v + ([None] * (maxlen - len(v)))
    try:
        df = pd.DataFrame(data)
    except ValueError as e:
        raise e
    edited_df = st.data_editor(df, column_config=columnConfig, use_container_width=True)

    if st.button('Submit'):
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

        st.info(f"Num Rows: {len(edited_df)}")
        for v, amount in updates_made.items():
            st.info(f"Num Mistakes {v.value}: {amount} ({pct(amount, len(edited_df))} %)")
        st.write('Successfully Saved Results to DB!')
        logging.info('finished and saved subtitles review')


def newest_ever_compare(project_id):
    return asyncio.run(_newest_ever_compare(project_id))


async def _migrate_feedbacks():
    await init_db(mongodb_settings, [TranslationFeedbackV2, TranslationFeedback])
    feedbacks = await TranslationFeedback.find().to_list()
    objs = [TranslationFeedbackV2.migrate_v1_obj(o) for o in feedbacks]
    await TranslationFeedbackV2.insert_many(objs)
