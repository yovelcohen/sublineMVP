import logging

import streamlit as st

from common.models.core import Translation
from services.constructor import SubtitlesResults
from services.runner import SRTHandler


async def recover_translation(obj: Translation) -> SubtitlesResults:
    name = obj.name
    if obj:
        missing_str = f'{len([x for x in obj.subtitles if x.translations is None])} rows missing translation, out of {len(obj.subtitles)}'
        st.toast(f"Found translation with name: {name},\n{missing_str}")
        logging.info(f"Found translation with name: {name}")
        translator = SRTHandler(translation_obj=obj, raw_content='')
        ret = await translator.run(recovery_mode=True)
        return ret
    else:
        logging.error(f"Translation with name: {name} not found")
        st.error(f"Translation with name: {name} not found")
        raise KeyError(f"Translation with name: {name} not found")
