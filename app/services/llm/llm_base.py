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


def validate_and_parse_answer(answer: Choice, preferred_suffix='}'):
    """
    validate the answer from openai and return the parsed answer
    """
    json_str = answer.message.content.strip()
    try:
        fixed_json = json_repair.loads(json_str)
    except ValueError as e:
        try:
            json_str = json_str.rsplit(',', 1)[0] + preferred_suffix
            json_str = json_repair.loads(json_str)
        except ValueError:
            pass
        if isinstance(json_str, dict):
            fixed_json = json_str
        else:
            if not json_str.endswith(preferred_suffix):
                last_char = json_str[-1]
                if last_char.isalnum():
                    json_str += '"}'
                elif last_char == ',':
                    json_str = json_str[:-1] + preferred_suffix
                elif json_str.endswith('"'):
                    json_str += preferred_suffix
                else:
                    raise e
                try:
                    fixed_json = json_repair.loads(json_str)
                except ValueError as e:
                    logging.exception('failed to parse answer from openai, fixed to_json manually')
                    raise e
            else:
                raise e
    except Exception as e:
        raise e

    if isinstance(fixed_json, (list, set, tuple)):
        return fixed_json, -1

    data = fixed_json['translation'] if 'translation' in fixed_json else fixed_json
    data = {str(k): v for k, v in data.items() if k.isdigit()}
    last_index = -1
    if len(data) > 1:
        # we don't trust the fixer or openai, so we remove the last index,
        # which is usually a shitty translation anyway.
        key, val = data.popitem()
        last_index = key
    try:
        last_index = int(last_index)
    except:
        raise ValueError(f'last index is not an integer, last index is {last_index}')
    return data, last_index


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


async def make_openai_request(
        messages: list[dict[str, str]],
        seed: int = 99,
        model: str = 'best',
        temperature: float | None = .1,
        top_p: float | None = None,
        max_tokens: int | None = None,
        num_options: int | None = None,
        preferred_suffix='}',
):
    """
    :param messages: list of openai message dicts
    :param seed: constant to use on the model, improves consistency
    :param model: 3.5 or 4
    :param temperature: model temp
    :param top_p: model top_p sampling, usually used when tempature isn't
    :param max_tokens: max response tokens
    :param num_options: number of answers to generate, None == 1
    :param preferred_suffix: preferred_suffix of the expected JSON structure output, used to fix broken json responses.

    :returns A dict, where keys are versions, the values are tuples of dict[row_index, translation]
             and the last valid row index that was loaded from the JSON.
    """
    choices = await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                 num_options=num_options, max_tokens=max_tokens)
    choice = choices[0]
    answer, last_idx = validate_and_parse_answer(answer=choice, preferred_suffix=preferred_suffix)
    return answer, last_idx
