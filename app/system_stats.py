import asyncio
import statistics
from collections import defaultdict
from math import isnan
from typing import Literal

import pandas as pd
from beanie import Link
from beanie.odm.operators.find.comparison import In
from pydantic import BaseModel

from common.models.core import Project
from common.models.translation import ModelVersions, Translation, TranslationSteps
from common.utils import pct
from new_comparsion import TranslationFeedbackV2

import streamlit as st

TEN_K, MILLION, BILLION = 10_000, 1_000_000, 1_000_000_000
TWO_HOURS = 60 * 60 * 2
THREE_MINUTES = 60 * 3


def _format_number(num):
    if TEN_K <= num < MILLION:
        return f"{num / 1000:.1f}k"
    elif MILLION <= num < BILLION:
        return f"{num / MILLION:.2f}m"
    elif num >= BILLION:
        return f"{num / BILLION:.2f}b"
    else:
        return str(num)


class Stats(BaseModel):
    version: ModelVersions | Literal['ALL']
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


STATES_MAP = {
    'pe': 'Pending',
    'ip': 'In Progress',
    'aa': "Audio Intelligence",
    'va': "Video Intelligence",
    'tr': "Translating",
    'co': 'Completed',
    'fa': 'Failed'
}

ALLOWED_VERSIONS = [v.value for v in (ModelVersions.V039, ModelVersions.V3, ModelVersions.V1, ModelVersions.V0310,
                                      ModelVersions.V0310_G, ModelVersions.V0311_GENDER)]


async def _get_translations_stats() -> list[dict]:
    translations = await Translation.find(In(Translation.engine_version, ALLOWED_VERSIONS)).to_list()
    p_ids = list({t.project_id for t in translations})
    projects = {proj.id: proj.name for proj in await Project.find(In(Project.id, p_ids)).to_list()}

    def get_took(t):
        minutes, seconds = divmod(t, 60)
        return "{:02d}:{:02d}".format(int(minutes), int(seconds))

    translations: list[Translation]
    return [
        {
            'id': translation.id,
            'name': projects[translation.project_id],
            'Amount Rows': len(translation.subtitles),
            'State': STATES_MAP[translation.flow_state.state.value],
            'Took': get_took(translation.flow_state.took),
            'Translated Rows in %': (100.0 if translation.flow_state.state == TranslationSteps.COMPLETED
                                     else pct(len(translation.rows_with_translation), len(translation.subtitles))),
            'Engine Version': translation.engine_version.value,
            'Delete': False,
            'Amount OG Characters': sum([len(r.content) for r in translation.subtitles]),
            'Amount Translated Characters': sum(
                [
                    len(r.translations.selection) for r in translation.subtitles
                    if r.translations is not None and r.translations.selection is not None
                ]
            ),
            'Amount OG Words': sum([len(r.content.split()) for r in translation.subtitles]),
            'Amount Translated Words': sum(
                [
                    len(r.translations.selection.split()) for r in translation.subtitles
                    if r.translations is not None and r.translations.selection is not None
                ]
            )
        }
        for translation in translations if translation.project_id in projects
    ]


async def _get_data():
    data, fbs = await asyncio.gather(
        _get_translations_stats(),
        TranslationFeedbackV2.find(In(TranslationFeedbackV2.version, ALLOWED_VERSIONS)).to_list()
    )
    return data, fbs


@st.cache_data(ttl=THREE_MINUTES)
def get_data_for_stats():
    data, fbs = asyncio.run(_get_data())
    return data, fbs


async def get_stats(data, fbs) -> list[Stats]:
    by_v = {
        v: {'feedbacks': list(), 'sum_checked_rows': 0, 'all_names': set(), 'translations': [],
            'name_to_count': dict(), 'by_error': defaultdict(list), 'count': 0}
        for v in set([t['Engine Version'] for t in data])
    }

    for feedback in fbs:
        version = feedback.version
        by_v[version]['feedbacks'].extend(feedback.marked_rows)
        by_v[version]['sum_checked_rows'] += feedback.total_rows
        by_v[version]['all_names'].add(feedback.name)
        by_v[version]['name_to_count'][feedback.name] = len(feedback.marked_rows)
        by_v[version]['count'] += 1
        for row in feedback.marked_rows:
            if row['error'] not in (None, 'None'):
                row['Name'] = feedback.name
                by_v[version]['by_error'][row['error']].append(row)

    for translation in data:
        version = translation['Engine Version']
        if translation['name'] in by_v[version]['all_names']:
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
        totalFeedbacks=data['count'],
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
    df = df[['name', 'State', 'Engine Version', 'Translated Rows in %', 'Reviewed', 'Amount Rows',
             'Amount Errors', 'Errors %', 'Amount OG Words', 'Amount Translated Words',
             'Amount OG Characters', 'Amount Translated Characters']]
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
        errors_pct = [err for err in df['Errors %'].to_list() if not isnan(err) and err != 0.0]
        errors = [err for err in df['Amount Errors'].to_list() if not isnan(err) and err != 0]
        if len(errors) > 0:
            st.metric(
                'Median Errors Count - Single Translation',
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
    data, fbs = get_data_for_stats()
    stats_list = asyncio.run(get_stats(data, fbs))
    translations, errorCounts, errors = list(), dict(), dict()
    for stat in stats_list:
        translations.extend(stat.translations)
        for key, val in stat.errorsCounter.items():
            if key not in errorCounts:
                errorCounts[key] = 0
            errorCounts[key] += val
        for key, val in stat.errors.items():
            if key not in errors:
                errors[key] = list()
            errors[key].extend(val)

    total_stats = Stats(
        version='ALL',  # noqa
        totalChecked=sum([s.totalChecked for s in stats_list]),
        totalFeedbacks=sum([s.totalFeedbacks for s in stats_list]),
        errorsCounter=errorCounts,
        errors=errors,
        errorPct=sum([s.errorPct for s in stats_list]),
        totalOgCharacters=sum([s.totalOgCharacters for s in stats_list]),
        totalTranslatedCharacters=sum([s.totalTranslatedCharacters for s in stats_list]),
        amountOgWords=sum([s.amountOgWords for s in stats_list]),
        amountTranslatedWords=sum([s.amountTranslatedWords for s in stats_list]),
        translations=translations
    )
    tabs = st.tabs([stats.version.value.upper() for stats in stats_list] + ['Total'])
    for tab, stats in zip(tabs[:-1], stats_list):
        with tab:
            stats_for_version(stats)
    with tabs[-1]:
        stats_for_version(total_stats)
