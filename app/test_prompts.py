import json_repair
import openai
import streamlit as st
from anthropic import AsyncAnthropic, Anthropic
from anthropic.types.beta import Message

from app.common.models.translation import Chunk

chunks = []
ANTHROPIC_API_KEY: str = "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"
openai_client = openai.OpenAI(api_key=st.secrets['OPENAI_KEY'])
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


def test_chunk_on_prompt(chunk: Chunk, sys_prompt, user_prompt, temperature, gpt=True):
    messages = []
    if sys_prompt:
        messages.append({'role': 'system', 'content': sys_prompt})
    messages.append({'role': 'user', 'content': user_prompt})

    model = 'gpt-4-1106-preview' if gpt else 'claude-2.1'
    method = openai_client.chat.completions.create if gpt else anthropic_client.beta.messages.create
    response = method(
        messages=messages,
        temperature=temperature,
        max_tokens=4000,
        model=model
    )

    ret = response.choices[0].message.content if gpt else response.content[0].text
    try:
        return json_repair.loads(ret)
    except ValueError:
        return ret


def prompt_tester_view():
    sys_prompt = st.text_input(label='System Prompt')
    user_prompt = st.text_input(label='User Prompt')
    temperature = st.slider(min_value=0.01, max_value=1)

    submit = st.button('submit')
    if submit:
        ...
