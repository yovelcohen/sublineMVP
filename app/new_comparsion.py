import asyncio
import datetime
import logging

import pandas as pd
import streamlit as st
from beanie import PydanticObjectId, Document
from beanie.exceptions import CollectionWasNotInitialized
from pydantic import Field

from common.config import settings, mongodb_settings
from common.db import init_db
from common.models.core import Project
from common.models.translation import Translation, SRTBlock, ModelVersions

labels = ['Gender Mistake', 'Time Tenses', 'Names', 'Slang', 'Prepositions',
          'Name "as is"', 'not fit in context', 'Plain Wrong Translation']

select_box_col = lambda label: st.column_config.SelectboxColumn(
    width='medium', label=label, required=False, options=labels
)


class TranslationFeedback(Document):
    name: str = ''
    version: ModelVersions = Field(default=ModelVersions.V1, alias='engine_version')
    total_rows: int
    marked_rows: list[dict]
    duration: datetime.timedelta | None = None

    @property
    def error_pct(self):
        return round(((len(self.marked_rows) / self.total_rows) / 100), 2)


async def get_feedback_by_name(name, v=ModelVersions.V1) -> TranslationFeedback | None:
    await init_db(mongodb_settings, [TranslationFeedback])
    return await TranslationFeedback.find(TranslationFeedback.name == name,
                                          TranslationFeedback.version == v.value).first_or_none()


async def save_to_db(rows, df, name, last_row: SRTBlock, existing_feedback: TranslationFeedback | None = None,
                     version=ModelVersions.V1):
    def filter_row(row):
        ke = 'V1 Error 1' if 'V1 Error 1' in row else 'Error 1'
        return row[ke] not in ('None', None)

    rows = [r for r in rows if filter_row(r)]
    if not existing_feedback:
        params = dict(marked_rows=rows, total_rows=len(df), name=name, duration=last_row.end, version=version)
        try:
            existing_feedback = await TranslationFeedback(**params).create()
        except CollectionWasNotInitialized as e:
            await init_db(mongodb_settings, [TranslationFeedback])
            existing_feedback = await TranslationFeedback(**params).create()
    else:
        before = len(existing_feedback.marked_rows)
        existing_feedback.marked_rows = rows
        if not existing_feedback.duration:
            existing_feedback.duration = last_row.end

        await existing_feedback.replace()
        logging.info(f'Updated translation feedback, amount rows before: {before}, after: {len(rows)} ')

    return existing_feedback


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


async def new_compare(project_id):
    project_id = PydanticObjectId(project_id)
    project = await Project.get(project_id)
    translations = await Translation.find(Translation.project.id == project_id).to_list()  # noqa
    dfs = [one_panel(translation) for translation in translations]

    with st.sidebar:
        info = {'Name': project.name,
                'Target Language': translations[0].target_language,
                'Available Version': [obj.engine_version.value for obj in translations]}
        for name, val in info.items():
            st.info(f'{name}: {val}')

    for df, version, last_row in dfs:
        with st.expander(expanded=False, label=f'Engine Version: {version}'):
            config = {
                'Original Language': st.column_config.TextColumn(width='large'),
                'Glix Translation 1': st.column_config.TextColumn(width='large'),
                'Error 1': select_box_col('Error 1'),
                'Error 2': select_box_col('Error 2'),
            }
            existing_feedback = await get_feedback_by_name(project.name)
            if version == ModelVersions.V1 and existing_feedback is not None:
                TRANSLATION_KEY = 'Glix Translation 1'
                logging.info(
                    f'Found existing feedback, updating df, amount marked rows: {len(existing_feedback.marked_rows)}')
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
