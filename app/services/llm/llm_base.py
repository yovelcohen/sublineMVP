import asyncio
import copy
import logging
import time
from enum import Enum

import json_repair
import openai
import tiktoken
from httpx import Timeout
from openai import AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionNamedToolChoiceParam
from openai.types.edit import Choice
import streamlit as st

from common.config import settings
from common.context_vars import total_stats

encoder = tiktoken.encoding_for_model('gpt-4')


class TokenCountTooHigh(ValueError):
    pass


MODEL_NAME = 'glix'

azure_client = openai.AsyncAzureOpenAI(
    api_key=settings.OPENAI_KEY,
    api_version="2023-12-01-preview",
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=MODEL_NAME,
)
azure_client1 = openai.AsyncAzureOpenAI(
    api_key=settings.OPENAI_BACKUP_KEY,
    api_version="2023-12-01-preview",
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT_BACKUP,
    azure_deployment=MODEL_NAME,
)


def report_stats(openai_resp: CompletionUsage):
    stats = openai_resp.model_dump()
    val = copy.deepcopy(total_stats.get())
    val.update(stats)
    logging.info('token cost updated stats: %s', val)
    total_stats.set(val)


class Model(int, Enum):
    GPT4V = 1
    GPT4B = 2


_CLIENTS = {Model.GPT4V: azure_client, Model.GPT4B: azure_client1}


async def send_request(
        messages, seed, model: Model, max_tokens=None, top_p=None, temperature=None, num_options=None, retry_count=1,
        **kwargs
) -> list[Choice] | ChatCompletion | AsyncStream[ChatCompletionChunk]:
    req = dict(messages=messages, seed=seed, model=MODEL_NAME, max_tokens=max_tokens)
    if 'functions' not in kwargs:
        req['response_format'] = {"type": "json_object"}
    else:
        req['function_call'] = {'name': kwargs['functions'][0]['name']}
    if top_p:
        req['top_p'] = top_p
    if temperature:
        req['temperature'] = temperature
    if num_options:
        req['n'] = num_options
    if kwargs:
        req.update(kwargs)

    client = _CLIENTS[model]
    try:
        t1 = time.time()
        func_resp = await client.chat.completions.create(**req)
        t2 = time.time()
        logging.info('finished openai request, took %s seconds', t2 - t1)
        # report_stats(func_resp)
        return func_resp
    except openai.APITimeoutError as e:
        logging.exception('openai timeout error, sleeping for 5 seconds and retrying')
        st.toast('OpenAI is taking too long to respond, please wait...')
        await asyncio.sleep(5)
        if retry_count == 3:
            logging.exception('openai timeout error, failed after 3 retries')
            raise e
        return await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                  retry_count=retry_count + 1, num_options=num_options, max_tokens=max_tokens)
