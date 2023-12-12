import asyncio
import copy
import logging
import time

import json_repair
import openai
import tiktoken
from httpx import Timeout
from openai.types.edit import Choice

from app.common.config import settings
from app.common.context_vars import total_stats

encoder = tiktoken.encoding_for_model('gpt-4')


class TokenCountTooHigh(ValueError):
    pass


MODELS = {'good': 'gpt-3.5-turbo-1106', 'best': 'gpt-4-1106-preview'}
client = openai.AsyncOpenAI(api_key=settings.OPENAI_KEY, timeout=Timeout(60 * 3))


def report_stats(openai_resp):
    stats = openai_resp.usage.model_dump()
    logging.info('openai stats: %s', stats)
    val = copy.deepcopy(total_stats.get())
    val.update(stats)
    total_stats.set(val)


async def send_request(
        messages, seed, model, max_tokens=None, top_p=None, temperature=None, num_options=None, retry_count=1
) -> list[Choice]:
    req = dict(messages=messages, seed=seed, response_format={"type": "json_object"}, model=MODELS[model])
    if top_p:
        req['top_p'] = top_p
    if temperature:
        req['temperature'] = temperature
    if num_options:
        req['n'] = num_options
    if max_tokens:
        req['max_tokens'] = max_tokens
    try:
        t1 = time.time()
        func_resp = await client.chat.completions.create(**req)
        t2 = time.time()
        logging.info('finished openai request, took %s seconds', t2 - t1)
        report_stats(func_resp)
        return func_resp.choices
    except openai.APITimeoutError as e:
        logging.exception('openai timeout error, sleeping for 5 seconds and retrying')
        await asyncio.sleep(5)
        if retry_count == 3:
            logging.exception('openai timeout error, failed after 3 retries')
            raise e
        if model == 'best':
            model = 'good'  # downgrade to avoid limit errors
        return await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                  retry_count=retry_count + 1, num_options=num_options, max_tokens=max_tokens)
