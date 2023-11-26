import asyncio
from typing import cast

import openai
import streamlit as st
from io import StringIO
import srt as srt_lib

from logic.constructor import SrtString, SRTTranslator, TranslatedSRT
from logic.function import SRTBlock



async def translate(bar, _name, _rows, _target_language):
    bar.progress(10, 'Loading..')
    translator = SRTTranslator(target_language=_target_language, project_name=_name, rows=_rows)
    _ret = await translator(num_rows_in_chunk=150, bar=bar)
    st.session_state['name'] = _ret
    bar.progress(100, 'Done!')


def clean():
    del st.session_state['name']


with st.form("Translate"):
    name = st.text_input("Name")
    type_ = st.selectbox("Type", ["Movie", "Series"])
    uploaded_file = st.file_uploader("Choose a file")
    source_language = st.selectbox("Source Language", ["English", "Hebrew", 'French', 'Arabic', 'Spanish'])
    target_language = st.selectbox("Source Language", ["Hebrew", 'French', 'Arabic', 'Spanish'])
    assert source_language != target_language
    submitted = st.form_submit_button("Translate")
    if submitted:
        _bar = st.progress(0, text='parsing srt...')
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = cast(SrtString, stringio.read())
        rows = [SRTBlock(index=row.index, start=row.start, end=row.end, content=row.content)
                for row in srt_lib.parse(string_data)]
        asyncio.run(translate(_name=name, _rows=rows, _target_language=target_language, bar=_bar))

if st.session_state.get('name', False):
    srt: TranslatedSRT = st.session_state['name']
    st.download_button('Download SRT', data=srt.srt, file_name=f'{name}_{target_language}', mime='srt', type='primary',
                       on_click=clean)
