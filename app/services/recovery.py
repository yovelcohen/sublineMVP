import logging

import streamlit as st

from common.models.core import Translation
from services.constructor import SubtitlesResults
from services.runner import SRTHandler


async def recover_translation(obj: Translation) -> SubtitlesResults:
    name = obj.name
    if obj:
        bar = st.progress(5, 'Loading translation from breakpoint...')
        st.session_state['bar'] = bar
        logging.info(f"Found translation with name: {name}")
        translator = SRTHandler(translation_obj=obj, raw_content='')
        ret = await translator.run(recovery_mode=True, raw_results=True)
        return ret
    else:
        logging.error(f"Translation with name: {name} not found")
        st.error(f"Translation with name: {name} not found")
        raise KeyError(f"Translation with name: {name} not found")
