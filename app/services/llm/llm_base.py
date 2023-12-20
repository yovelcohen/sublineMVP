import asyncio
import copy
import logging
import time

import json_repair
import openai
import tiktoken
from httpx import Timeout
from openai import AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.edit import Choice
import streamlit as st

from common.config import settings
from common.context_vars import total_stats

encoder = tiktoken.encoding_for_model('gpt-4')


class TokenCountTooHigh(ValueError):
    pass


MODELS = {'good': 'gpt-3.5-turbo-1106', 'best': 'gpt-4-1106-preview'}
openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_KEY, timeout=Timeout(60 * 5))
azure_client = openai.AsyncAzureOpenAI(
    api_key=settings.OPENAI_KEY,
    api_version="2023-12-01-preview",
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment='glix',
)


def report_stats(openai_resp: CompletionUsage):
    stats = openai_resp.model_dump()
    val = copy.deepcopy(total_stats.get())
    val.update(stats)
    logging.info('token cost updated stats: %s', val)
    total_stats.set(val)


async def send_request(
        messages, seed, model=None, max_tokens=None, top_p=None, temperature=None, num_options=None, retry_count=1,
        **kwargs
) -> list[Choice] | ChatCompletion | AsyncStream[ChatCompletionChunk]:
    if model == 'gpt-3.5':
        model = 'gpt-3.5-turbo-1106'
        client = openai_client
    else:
        model = 'glix'
        client = azure_client
    req = dict(messages=messages, seed=seed, model=model)
    if 'tools' not in kwargs:
        req['response_format'] = {"type": "json_object"}
    if top_p:
        req['top_p'] = top_p
    if temperature:
        req['temperature'] = temperature
    if num_options:
        req['n'] = num_options
    if max_tokens:
        req['max_tokens'] = max_tokens
    if kwargs:
        req.update(kwargs)

    try:
        t1 = time.time()
        func_resp = await client.chat.completions.create(**req)
        t2 = time.time()
        logging.info('finished openai request, took %s seconds', t2 - t1)
        # report_stats(func_resp)
        return func_resp
    except openai.APITimeoutError as e:
        logging.exception('openai timeout error, sleeping for 5 seconds and retrying')
        if not st.session_state.get('openAIErrorMsg'):
            st.session_state['openAIErrorMsg'] = True
            st.toast('OpenAI is taking too long to respond, please wait...')
        await asyncio.sleep(5)
        if retry_count == 3:
            logging.exception('openai timeout error, failed after 3 retries')
            raise e
        if model == 'best':
            model = 'good'  # downgrade to avoid limit errors
        if st.session_state.get('openAIErrorMsg'):
            del st.session_state['openAIErrorMsg']
        return await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                  retry_count=retry_count + 1, num_options=num_options, max_tokens=max_tokens)


JSON_FIXER_SEED = 189


async def fix_json_request(json_str: str):
    msg = [{'role': 'system',
            'content': "You are the JSON Fixer, your job is to receive a broken JSON and return a fixed version of it. the JSON can be invalid because of a character in the middle of the JSON. RETURN ONLY THE OUTPUT FIXED JSON, MAKE SURE IT'S IN VALID JSON STRUCTURE."},
           {'role': "user", 'content': json_str}]
    resp = await send_request(messages=msg, seed=JSON_FIXER_SEED, model='gpt-3.5', temperature=.1)
    return json_repair.loads(resp.choices[0].message.content)
