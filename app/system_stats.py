import asyncio
import statistics
from collections import defaultdict
from math import isnan
from typing import Literal

import pandas as pd
from beanie import Link
from pydantic import BaseModel

from common.config import mongodb_settings
from common.db import init_db
from common.models.core import Project, Client
from common.models.translation import ModelVersions, Translation
from common.utils import pct
from new_comparsion import TranslationFeedbackV2

import streamlit as st

TEN_K, MILLION, BILLION = 10_000, 1_000_000, 1_000_000_000
TWO_HOURS = 60 * 60 * 2

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


async def _get_translations_stats() -> list[dict]:
    translations = await Translation.find_all().to_list()
    projects = {proj.id: proj.name for proj in await Project.find_all().to_list()}

    def get_took(t):
        minutes, seconds = divmod(t, 60)
        return "{:02d}:{:02d}".format(int(minutes), int(seconds))

    translations: list[Translation]
    return [
        {
            'id': translation.id,
            'name': projects[translation.project_id],
            'Amount Rows': len(translation.subtitles),
            'State': STATES_MAP[translation.state.value],
            'Took': get_took(translation.took),
            'Engine Version': translation.engine_version.value,
            'Delete': False,
            'Amount OG Characters': sum([len(r.content) for r in translation.subtitles]),
            'Amount Translated Characters': sum(
                [len(r.translations.selection) for r in translation.subtitles if r.translations is not None]
            ),
            'Amount OG Words': sum([len(r.content.split()) for r in translation.subtitles]),
            'Amount Translated Words': sum(
                [len(r.translations.selection.split()) for r in translation.subtitles if r.translations is not None]
            )
        }
        for translation in translations
    ]


async def _get_data():
    if st.session_state.get('DB') is None:
        db, docs = await init_db(mongodb_settings, [Translation, TranslationFeedbackV2, Project, Client])
        st.session_state['DB'] = db
    data, fbs = await asyncio.gather(
        _get_translations_stats(), TranslationFeedbackV2.find_all(fetch_links=True).to_list()
    )
    return data, fbs


@st.cache_data(ttl=TWO_HOURS)
def get_data_for_stats():
    data, fbs = asyncio.run(_get_data())
    return data, fbs


async def get_stats(data, fbs) -> list[Stats]:
    by_v = {
        v: {'feedbacks': list(), 'sum_checked_rows': 0, 'all_names': set(), 'translations': [],
            'name_to_count': dict(), 'by_error': defaultdict(list), 'count': 0}
        for v in set([t['Engine Version'] for t in data])
    }

    # TODO: Needs to be revamped for TranslationFeedbackV2
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
