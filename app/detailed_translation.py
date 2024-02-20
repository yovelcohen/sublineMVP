import json

import pandas as pd
import streamlit as st

from common.models.translation import TranslationSuggestions
from common.utils import download_azure_blob, list_blobs_in_path


def load_suggestions(suggestions: TranslationSuggestions | list[str]) -> TranslationSuggestions:
    if isinstance(suggestions, TranslationSuggestions):
        return suggestions
    return TranslationSuggestions.from_strings(suggestions)


def _get_panel_data_for_blob(blob_path):
    data = download_azure_blob('projects', blob_path)
    if isinstance(data, bytes):
        data = json.loads(data.decode('utf-8'))
    serialized, selections_map = dict(), dict()
    for row_index, row_data in data['data'].items():
        if selection := row_data.pop('selected', None):
            selections_map[row_index] = selection
        serialized[row_index] = {
            tmp: TranslationSuggestions.model_validate(sugg_set)
            for tmp, sugg_set in row_data.items() if isinstance(sugg_set, dict)
        }
    return serialized, selections_map


def get_panel_html_for_blob(
        data: dict[int, dict[float | str, TranslationSuggestions | str]], selections_map: dict[int, str]
) -> pd.DataFrame:
    rows, temps, selections, suggestions_data = list(), list(), list(), list()

    # Find the maximum number of suggestions for dynamic column creation
    max_suggestions_length = max(
        [len(ts.get_suggestions()) for temp_data in data.values() for ts in temp_data.values()]
    )

    for row_index, temp_data in data.items():
        for temp, translation_suggestions in temp_data.items():
            selections.append(selections_map.get(row_index, ''))
            rows.append(row_index)
            temps.append(temp)
            suggestions_data.append(
                [getattr(translation_suggestions, f'v{i}', None) for i in range(1, max_suggestions_length + 1)]
            )

    columns = ['v{}'.format(i) for i in range(1, max_suggestions_length + 1)]
    return pd.DataFrame(
        suggestions_data, columns=columns,
        index=pd.MultiIndex.from_tuples(zip(rows, selections, temps), names=['Row Index', 'Selection', 'Temperatures'])
    )


def detailed_translation_panel():
    def format_name(v):
        name = v.split('debug_gpt_reports/')[-1].split('.json')[0]
        name, version = name.split('_v')
        return f'{name} (V{version})'

    with st.form('formdtp'):
        options = list_blobs_in_path(container_name='projects', folder_name='debug_gpt_reports/')
        chosen_blob_path = st.selectbox('Choose Translation', options=options, format_func=format_name)
        submit = st.form_submit_button('Get')
        if submit:
            data, selections_map = _get_panel_data_for_blob(chosen_blob_path)
            df = get_panel_html_for_blob(data, selections_map)
            st.session_state['dtp_df'] = df
            html_table = df.to_html()
            st.markdown(html_table, unsafe_allow_html=True)

    if st.session_state.get('dtp_df', None) is not None:
        with st.sidebar:
            st.header('Download as csv')
            st.download_button('Download', st.session_state['dtp_df'].to_csv(), f'suggestions.csv', 'text/csv')
