import random
from pathlib import Path
import streamlit as st

import yaml
from streamlit_tags import st_tags

BASE_PATH = Path(__file__).parent


def view_prompts():
    with open(BASE_PATH / 'prompts.yaml') as f:
        flows = yaml.safe_load(f)

    for flow in flows:
        version = flow['version']
        with st.expander(f'Version: {version}', expanded=False):
            st.text_area(value=flow['description'], label='desc', disabled=True, label_visibility='hidden')
            st_tags(value=flow['features'], label='Features', text='')
            st.subheader('Prompts Flow', divider='blue')
            for i, step in enumerate(flow['flow'], start=1):
                name = step['step']
                st.subheader(f'Step {i}: {name}', divider='orange')
                for p_i, prompt in enumerate(step['prompts']):
                    msg_type = prompt['writer']
                    prompt_type = prompt['type']
                    st.subheader(f'Writer: {msg_type} - Type: {prompt_type}', divider='green')
                    st.text_area(
                        label='promptText',
                        value=prompt['text'],
                        disabled=True,
                        label_visibility='hidden',
                        key=f'{version}{i}{p_i}'
                    )
