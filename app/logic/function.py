import json
import logging
import os
import time
from datetime import datetime
from json import JSONDecodeError
from typing import TypedDict, Literal

import json_repair
import openai
import tiktoken
from openai.types.chat.chat_completion import Choice
from srt import Subtitle, timedelta_to_srt_timestamp

logger = logging.getLogger(__name__)
OPENAI_KEY: str = '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'
encoder = tiktoken.encoding_for_model('gpt-4')
async_openai_client = openai.AsyncOpenAI(api_key=OPENAI_KEY)


def parse_time(seconds: float) -> datetime:
    """Convert SRT time format to datetime object."""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return datetime(1, 1, 1, hours, minutes, secs, millis * 1000)
    except:
        raise ValueError(f"Invalid SRT time format: {seconds}")


def _format_timestamp(seconds: float, separator: str) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


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


def get_openai_translation_func(srt_items: list[SRTBlock]):
    indices = [str(item.index) for item in srt_items]
    return [
        {
            "name": "translate_srt_file",
            "description": "Translate the each row of this subtitles And return the translated rows by their corresponding index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "translations": {
                        "type": "object",
                        "properties": {idx: {'type': 'string'} for idx in indices},
                        'description': "translation of each row based on it's leading index",
                        "required": [*indices]
                    }
                }
            }
        }
    ]


def validate_and_parse_answer(answer: Choice, last_idx: int):
    """
    validate the answer from openai and return the parsed answer
    """
    json_str = answer.message.content.strip()
    try:
        fixed_json = json_repair.loads(json_str)
    except ValueError as e:
        if not json_str.endswith('}'):
            last_char = json_str[-1]
            if last_char.isalnum():
                json_str += '"}'
            elif last_char == ',':
                json_str = json_str[:-1] + '}'
            elif json_str.endswith('"'):
                json_str += '}'
            else:
                raise e
            try:
                fixed_json = json_repair.loads(json_str)
            except ValueError as e:
                raise e
            logger.exception('failed to parse answer from openai, fixed json manually')
        else:
            raise e

    data = fixed_json['translation'] if 'translation' in fixed_json else fixed_json
    last_index = -1
    if len(data) > 1:
        last_index = sorted(tuple(data), reverse=True)[0]
        # we don't trust the fixer or openai, so we remove the last index,
        # which is usually a shitty translation anyway.
        data.pop(last_index)
    return data, last_index


class TokenCountTooHigh(ValueError):
    pass


def prepare_text_data(rows, target_language):
    msg = [
        {'role': 'user',
         'content': f'Translate the each row of this subtitles to {target_language}, And return the translated rows by their corresponding index:\n {json.dumps({row.index: row.content for row in rows})}'},
    ]
    text_data = dict(messages=msg, functions=get_openai_translation_func(rows))
    return text_data


MODELS = {'good': 'gpt-3.5-turbo-1106', 'best': 'gpt-4-1106-preview'}


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
        {'role': 'user',
         'content': f'Translate the each row of this subtitles to {target_language}, And return the translated rows by their corresponding index as valid JSON structure.:\n {json.dumps({row.index: row.content for row in rows})}'},
    ]
    data_token_cost = len(encoder.encode(json.dumps(messages)))
    max_completion_tokens = 16000 - (data_token_cost + tokens_safety_buffer)
    if max_completion_tokens < data_token_cost:
        raise TokenCountTooHigh(f'openai token limit is 16k, and the data token cost is {data_token_cost}, '
                                f'please reduce the number of rows')
    req = dict(messages=messages, seed=99, response_format={"type": "json_object"}, model=MODELS[model], temperature=.2)
    try:
        t1 = time.time()
        func_resp = await async_openai_client.chat.completions.create(**req)
        t2 = time.time()
        logger.info('finished openai request, took %s seconds', t2 - t1)
        ret = func_resp.choices[0]
        answer, last_idx = validate_and_parse_answer(answer=ret, last_idx=rows[-1].index)
        return answer, last_idx
    except Exception as e:
        raise e
