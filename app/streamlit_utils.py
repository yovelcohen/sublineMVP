import streamlit as st
from tqdm.asyncio import tqdm

SUPPORTED_VID_FORMATS = ['.mp3', '.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv']


def SelectBoxColumn(label, labels):
    return st.column_config.SelectboxColumn(
        width='medium', label=label, required=False, options=labels
    )
