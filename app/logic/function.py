import asyncio
import contextvars
import copy
import json
import logging
import os
import time
from contextvars import ContextVar
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
    return """You are a proficient TV shows English to Hebrew Translator,Make sure you follow the following rules and output a valid JSON, no matter what! 

- Modern Slang Translation: Translate English slang into appropriate Hebrew slang and day to day language.
Example 1: "Chill out" → Hebrew equivalent of "Relax" or "Take it easy."
Example 2: "Ghosting" → Hebrew slang for ignoring or avoiding someone.
Example 3: "Alien" → Will probably mean "חייזר" and not ״זר״ which usually will be foreigner.
Example 5: "Lying" → שקרן instead of כוזב OR "I'm Taking a shit" → based on context such as who speaks/spoken to, אני מחרבן or אני מחרבנת , example of day to day language
Example 6: "I don't deserve this" → Hebrew equivalent of ״זה לא מגיע לי״

- Gender-Specific Terms: Accurately translate gender-specific terms based on the context.
Example 1: "He is a doctor" → "הוא רופא" (for male) or "היא רופאה" (for female).
Example 2: "His book" → "הספר שלו" (for male) or "הספר שלה" (for female).
Example 3: "You guys" -> ״אתם״ או ״חברה״
Example 4: "Allison, You're crazy"  → ״אליסון, את משוגעת״  (This rule assumes that <feminine name> followed by "you" means את), same applies for male names. 
Example 5: "I saw you Mom, you did that" → "ראיתי אותך אמא, את עשית זה" infering the use of "את" instead of "אתה" based on the word "Mom".
  
- Idiomatic Expressions: Find Hebrew equivalents for English idioms.
Example 1: "Piece of cake" → Hebrew idiom for something very easy.
Example 2: "Break a leg" → Hebrew idiom for good luck.
Example 3: Use of the world "Fuck" → Fuck You (לעזאזל איתך), usually use לעזאזל
Example 4: "You Two" → שניכם instead of אתם שניים

- Syntax Adaptation: Adjust sentence structure for Hebrew syntax.
Example 1: "She loves dogs" (Subject-Verb-Object in English) → "היא אוהבת כלבים" (Hebrew structure).
Example 2: "I am reading a book" → Adjusted to fit Hebrew verb-subject-object order.
Example 3: "I was going to do something" → ״עמדתי לעשות משהו״ instead of ״הייתי הולך ל״ - adjusted because of the "to do" intead of a place

- Root System Usage: Apply the Hebrew root system in translations_result.
Example 1: English "write," "writer" → Hebrew roots for "write" (כתב) and related forms.
Example 2: English "run," "runner" → Hebrew roots for "run" (רוץ) and related forms.

- Consistent Transliteration: Keep transliterations of names and terms consistent.
Example 1: "John" always transliterated the same way in Hebrew.
Example 2: Technical terms like "Internet" consistently transliterated.

- Subtitle Length and Timing: Keep subtitles concise and timed well.
Example 1: Shortening a lengthy English sentence to fit the Hebrew screen time.
Example 2: Dividing a long English dialogue into shorter, readable Hebrew subtitles.

- Verb Conjugation Accuracy: Correctly conjugate Hebrew verbs.
Example 1: Translating "I ate" → "אכלתי" (past tense, first person singular).
Example 2: "They will go" → "הם ילכו" (future tense, third person plural).
Example 3: "You should have never gone there" → ״לא היית צריך ללכת לשם״"""


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
    data = {k: v for k, v in data.items() if k.isdigit()}
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
total_stats = dict()


def report_stats(openai_resp):
    stats = openai_resp.usage.model_dump()
    logger.info('openai stats: %s', stats)
    for k, v in stats.items():
        new_val = total_stats.get(k, 0) + v
        total_stats[k] = new_val


async def make_openai_request(
        messages: list[dict[str, str]],
        seed: int = 99,
        model: str = 'best',
        temperature: float | None = .1,
        top_p: float | None = None,
        max_tokens: int | None = None,
        retry_count=1,
        num_options: int | None = None,
        preferred_suffix='}'
):
    """
    :param messages:  list of openai message dicts
    :param seed: constant to use on the model, improves consistency
    :param model: 3.5 or 4
    :param temperature: model temp
    :param top_p: model top_p sampling, usually used when tempature isn't
    :param max_tokens: max response tokens
    :param retry_count: how many retries should attempt
    :param num_options: number of answers to generate, None == 1
    :param preferred_suffix: preferred_suffix of the expected JSON structure output, used to fix broken json responses.

    :returns A dict, where keys are versions, the values are tuples of dict[row_index, translation]
             and the last valid row index that was loaded from the JSON.
    """
    req = dict(messages=messages, seed=seed, response_format={"type": "json_object"},
               model=MODELS[model], max_tokens=max_tokens)
    if top_p:
        req['top_p'] = top_p
    if temperature:
        req['temperature'] = temperature
    if num_options:
        req['n'] = num_options
    try:
        t1 = time.time()
        func_resp = await client.chat.completions.create(**req)
        t2 = time.time()
        logger.info('finished openai request, took %s seconds', t2 - t1)
        report_stats(func_resp)
        ret = {}
        for v, choice in enumerate(func_resp.choices, start=1):
            answer, last_idx = validate_and_parse_answer(answer=choice, preferred_suffix=preferred_suffix)
            ret[v] = (answer, last_idx)
        return ret
    except openai.APITimeoutError as e:
        logger.exception('openai timeout error, sleeping for 5 seconds and retrying')
        await asyncio.sleep(5)
        if retry_count == 3:
            st.text('openai API timed out 3 times, please try again later')
            raise e
        return await make_openai_request(messages=messages, seed=seed, model=model, temperature=temperature,
                                         retry_count=retry_count + 1, num_options=num_options)
    except Exception as e:
        raise e


def get_openai_messages(*, target_language, rows):
    messages = [
        {"role": 'system', 'content': language_rules()},
        {'role': 'user',
         'content': f'Translate the each row of this subtitles to {target_language}, And return the translated rows by their corresponding index as valid JSON structure. \nRows:\n {json.dumps({row.index: row.content for row in rows})}'},
    ]
    return messages


async def translate_via_openai_func(
        *,
        rows: list['SRTBlock'],
        target_language: str,
        tokens_safety_buffer: int = 400,
        model: Literal['best', 'good'] = 'best',
        num_options: int | None = None
) -> dict[int, tuple[dict[str, str], int]]:
    """
    :param model: GPT Model to use
    :param rows: list of srt rows to translate
    :param target_language: translation's target language
    :param tokens_safety_buffer: a safety buffer to remove from max tokens, in order to avoid openai token limit errors
    :param num_options: number of translation versions to generate

    :returns a dict from row index to its translation and the last index that was NOT translated,
             if -1 then all rows were translated.
    """
    if not rows:
        return {}
    messages = get_openai_messages(rows=rows, target_language=target_language)
    data_token_cost = len(encoder.encode(json.dumps(messages)))
    max_completion_tokens = 16000 - (data_token_cost + tokens_safety_buffer)
    if max_completion_tokens < data_token_cost:
        raise TokenCountTooHigh(f'openai token limit is 16k, and the data token cost is {data_token_cost}, '
                                f'please reduce the number of rows')
    return await make_openai_request(messages=messages, seed=99, model=model, temperature=.1, num_options=num_options)
