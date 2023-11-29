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


def language_rules():
    return """You are a proficient TV shows English to Hebrew Translator, Make sure you follow these rules and output a valid JSON, no matter what! 

- Modern Slang Translation: Translate English slang into contextually appropriate Hebrew slang.
Example 1: "Chill out" → Hebrew equivalent of "Relax" or "Take it easy."
Example 2: "Ghosting" → Hebrew slang for ignoring or avoiding someone.

- Gender-Specific Terms: Accurately translate gender-specific terms based on the context.
Example 1: "He is a doctor" → "הוא רופא" (for male) or "היא רופאה" (for female).
Example 2: "His book" → "הספר שלו" (for male) or "הספר שלה" (for female).

- Idiomatic Expressions: Find Hebrew equivalents for English idioms.
Example 1: "Piece of cake" → Hebrew idiom for something very easy.
Example 2: "Break a leg" → Hebrew idiom for good luck.

- Syntax Adaptation: Adjust sentence structure for Hebrew syntax.
Example 1: "She loves dogs" (Subject-Verb-Object in English) → "היא אוהבת כלבים" (Hebrew structure).
Example 2: "I am reading a book" → Adjusted to fit Hebrew verb-subject-object order.

- Root System Usage: Apply the Hebrew root system in translations.
Example 1: English "write," "writer" → Hebrew roots for "write" (כתב) and related forms.
Example 2: English "run," "runner" → Hebrew roots for "run" (רוץ) and related forms.

- Consistent Transliteration: Keep transliterations of names and terms consistent.
Example 1: "John" always transliterated the same way in Hebrew.
Example 2: Technical terms like "Internet" consistently transliterated.

- Nikkud for Clarity: Use Nikkud where necessary for clear understanding.
Example 1: Using Nikkud to distinguish between "בָּנִים" (sons) and "בָּנִין" (building).
Example 2: Clarifying "חָמוֹר" (donkey) vs. "חָמוּר" (severe) with Nikkud.

- Subtitle Length and Timing: Keep subtitles concise and timed well.
Example 1: Shortening a lengthy English sentence to fit the Hebrew screen time.
Example 2: Dividing a long English dialogue into shorter, readable Hebrew subtitles.

- Feedback Integration: Continuously improve based on viewer feedback.
Example 1: Adjusting translations that viewers find unclear or inaccurate.
Example 2: Updating phrases based on recurring audience suggestions.

- Verb Conjugation Accuracy: Correctly conjugate Hebrew verbs.
Example 1: Translating "I ate" to "אכלתי" (past tense, first person singular).
Example 2: "They will go" to "הם ילכו" (future tense, third person plural)."""


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
    data = {k:v for k,v in data.items() if k.isdigit()}
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


try:
    api_key = st.secrets['OPENAI_KEY']
except:
    api_key = os.environ['OPENAI_KEY']

MODELS = {'good': 'gpt-3.5-turbo-1106', 'best': 'gpt-4-1106-preview'}
client = openai.AsyncOpenAI(api_key=api_key, timeout=Timeout(60 * 3))


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
            st.text('openai API timed out 3 times, please try again later')
            raise e
        return await make_openai_request(messages=messages, seed=seed, model=model, temperature=temperature,
                                         retry_count=retry_count + 1)
    except Exception as e:
        raise e


async def translate_via_openai_func(
        rows: list['SRTBlock'],
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
        {"role": 'system', 'content': language_rules()},
        {'role': 'user',
         'content': f'Translate the each row of this subtitles to {target_language}, And return the translated rows by their corresponding index as valid JSON structure. \nRows:\n {json.dumps({row.index: row.content for row in rows})}'},
    ]
    data_token_cost = len(encoder.encode(json.dumps(messages)))
    max_completion_tokens = 16000 - (data_token_cost + tokens_safety_buffer)
    if max_completion_tokens < data_token_cost:
        raise TokenCountTooHigh(f'openai token limit is 16k, and the data token cost is {data_token_cost}, '
                                f'please reduce the number of rows')
    return await make_openai_request(messages=messages, seed=99, model=model, temperature=.1)
