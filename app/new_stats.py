import asyncio
from collections import defaultdict
from typing import DefaultDict

import pandas as pd
from beanie import PydanticObjectId
import streamlit as st

from common.models.core import Project
from common.models.translation import Translation
from new_comparsion import TranslationFeedbackV2


async def get_stats_df():
    projects = await Project.find_all().to_list()

    all_ts = await Translation.find_all().to_list()
    proj_id_to_ts: DefaultDict[PydanticObjectId, list[Translation]] = defaultdict(list)
    for ts in all_ts:
        proj_id_to_ts[ts.project.ref.id].append(ts)

    feedbacks = await TranslationFeedbackV2.find_all().to_list()
    name_to_fbs = defaultdict(list)
    for fb in feedbacks:
        name_to_fbs[fb.name].append(fb)

    by_error = defaultdict(list)
    for project in projects:
        translations = proj_id_to_ts[project.id]
        feedbacks = name_to_fbs.get(project.name)

        if feedbacks is None:
            continue

        og_to_rows = defaultdict(dict)
        for t in translations:
            for row in t.subtitles:
                if row.translations is not None:
                    og_to_rows[row.content].update({t.engine_version.value: row.translations.selection})

        for fb in feedbacks:
            for row in fb.marked_rows:
                data = {'English': row['original'], 'Name': fb.name, 'Error': row['error'],
                        'Translation': row['translation']}
                if og_to_rows.get(row['original']) is not None:
                    data.update(og_to_rows[row['original']])
                by_error[row['error']].append(data)

        # pad shorter lists
        max_len = max([len(v) for v in by_error.values()])
        for error_name, rows in by_error.items():
            if len(rows) < max_len:
                by_error[error_name] = rows + ([None] * (max_len - len(rows)))

    for error_type, errors in by_error.items():
        st.header(error_type)
        errors = [err_di for err_di in errors if err_di is not None]
        df = pd.DataFrame(errors)

        conf = {
            k: st.column_config.TextColumn(width='large', disabled=True) for k in df.columns
            if 'v0' in k or k == 'English' or '.3.' in k
        }
        conf['Fixed'] = st.column_config.SelectboxColumn(
            width='small', label='Fixed', required=False, options=['True', 'False']
        )
        df['Fixed'] = [None] * len(df)
        st.data_editor(df, column_config=conf)

        # TODO: Add saving, maybe updated MarkedRow to have "fixed" flag? or a completely new db model?


def stats():
    return asyncio.run(get_stats_df())
