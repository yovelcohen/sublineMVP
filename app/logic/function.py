import asyncio
import json
import logging
import os
import time
from typing import TypedDict, Literal

import json_repair
import openai
import tiktoken
import streamlit as st
from httpx import Timeout
from openai.types.chat.chat_completion import Choice
from srt import Subtitle, timedelta_to_srt_timestamp

logger = logging.getLogger(__name__)
encoder = tiktoken.encoding_for_model('gpt-4')


class SRTRowDict(TypedDict):
    index: int
    start: str
    end: str
    text: str
    translation: str | None
    speaker: int | None


class SRTBlock(Subtitle):

    def __init__(self, index: int, start, end, content: str, speaker: int | None = None,
                 num_tokens: int | None = None, translation: str | None = None):
        super().__init__(index=index, start=start, end=end, content=content)
        self.speaker = speaker
        self.num_tokens = num_tokens
        self.translation = translation

    def __repr__(self):
        return f'{self.index}\n{self.content}'

    def to_dict(self) -> SRTRowDict:
        return SRTRowDict(index=self.index, speaker=self.speaker, start=timedelta_to_srt_timestamp(self.start),
                          end=timedelta_to_srt_timestamp(self.end), text=self.content, translation=self.translation)


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
                    logger.exception('failed to parse answer from openai, fixed json manually')
                    raise e
            else:
                raise e
    except Exception as e:
        raise e

    if isinstance(fixed_json, (list, set, tuple)):
        return fixed_json, -1

    data = fixed_json['translation'] if 'translation' in fixed_json else fixed_json
    last_index = -1
    if len(data) > 1:
        last_index = sorted(tuple(data), reverse=True)[0]
        # we don't trust the fixer or openai, so we remove the last index,
        # which is usually a shitty translation anyway.
        data.pop(last_index)
    try:
        last_index = int(last_index)
    except:
        raise ValueError(f'last index is not an integer, last index is {last_index}')
    return data, last_index


class TokenCountTooHigh(ValueError):
    pass


try:
    api_key = st.secrets['OPENAI_KEY']
except:
    api_key = os.environ['OPENAI_KEY']

MODELS = {'good': 'gpt-3.5-turbo-1106', 'best': 'gpt-4-1106-preview'}
client = openai.AsyncOpenAI(api_key=api_key, timeout=Timeout(60 * 5))


async def make_openai_request(messages: list[dict[str, str]], seed: int = 99, model: str = 'best',
                              temperature: float = .1, max_tokens: int | None = None, retry_count=1,
                              preferred_suffix='}'):
    req = dict(messages=messages, seed=seed, response_format={"type": "json_object"}, model=MODELS[model],
               temperature=temperature, max_tokens=max_tokens)
    try:
        t1 = time.time()
        func_resp = await client.chat.completions.create(**req)
        t2 = time.time()
        logger.info('finished openai request, took %s seconds', t2 - t1)
        ret = func_resp.choices[0]
        answer, last_idx = validate_and_parse_answer(answer=ret, preferred_suffix=preferred_suffix)
        return answer, last_idx
    except openai.APITimeoutError as e:
        logger.exception('openai timeout error, sleeping for 5 seconds and retrying')
        await asyncio.sleep(5)
        if retry_count == 3:
            st.error('openai API timed out 3 times, please try again later')
            raise e
        st.text('openai API timed out, sleeping for 5 seconds and retrying')
        return await make_openai_request(messages=messages, seed=seed, model=model, temperature=temperature,
                                         retry_count=retry_count + 1)
    except Exception as e:
        raise e


async def translate_via_openai_func(
        rows: list[SRTBlock],
        target_language: str,
        tokens_safety_buffer: int = 400,
        model: Literal['best', 'good'] = 'best'
) -> tuple[dict[str, str], int]:
    """
    :param model: GPT Model to use
    :param rows: list of srt rows to translate
    :param target_language: translation's target language
    :param tokens_safety_buffer: a safety buffer to remove from max tokens, in order to avoid openai token limit errors

    :returns a dict from row index to its translation and the last index that was NOT translated,
             if -1 then all rows were translated.
    """
    if not rows:
        return {}, -1
    messages = [
        {"role": 'system',
         'content': "You are a proficient TV shows translator, you notice subtle target language nuances like names, slang... and you are able to translate them. Make sure you output a valid JSON, no matter what!"},
        {'role': 'user',
         'content': f'Translate the each row of this subtitles to {target_language}, And return the translated rows by their corresponding index as valid JSON structure. \nRows:\n {json.dumps({row.index: row.content for row in rows})}'},
    ]
    data_token_cost = len(encoder.encode(json.dumps(messages)))
    max_completion_tokens = 16000 - (data_token_cost + tokens_safety_buffer)
    if max_completion_tokens < data_token_cost:
        raise TokenCountTooHigh(f'openai token limit is 16k, and the data token cost is {data_token_cost}, '
                                f'please reduce the number of rows')
    return await make_openai_request(messages=messages, seed=99, model=model, temperature=.1)
