import json
import logging
import time
from typing import Literal

import json_repair
from openai.types.edit import Choice

from common.models.core import SRTBlock
from services.llm.llm_base import send_request

_SEED = 99


def language_rules():
    return """You are a proficient TV shows English to Hebrew Translator,
Make sure you follow the following the Rules and output a valid JSON, no matter what! 
You will be paid handsomely for every corrected/improved translation you provide.

Rules:
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
Example 5: "I saw you Mom, you did that" → "ראיתי אותך אמא, את עשית זה" inferring the use of "את" instead of "אתה" based on the word "Mom".
Example 6: "Let me tell you something, Miss." → "תני לי לספר לך משהו גברתי" inferring the use of ״תני״ instead of ״תן״ because the sentece talks to a "Miss".  

- Idiomatic Expressions: Find Hebrew equivalents for English idioms.
Example 1: "Piece of cake" → Hebrew idiom for something very easy.
Example 2: "Break a leg" → Hebrew idiom for good luck.
Example 3: Use of the world "Fuck" → Fuck You (לעזאזל איתך), usually use לעזאזל
Example 4: "You Two" → "שניכם" instead of "אתם שניים"
Example 5: "Jesus Christ" → "אלוהים ישמור" instead of "ישו משיח"

- Syntax Adaptation and proper time tenses usage, Adjust sentence structure for Hebrew syntax.
Example 1: "She loves dogs" (Subject-Verb-Object in English) → "היא אוהבת כלבים" (Hebrew structure).
Example 2: "I am reading a book" → Adjusted to fit Hebrew verb-subject-object order.
Example 3: "I was going to do something" → ״עמדתי לעשות משהו״ instead of ״הייתי הולך ל״ - adjusted because of the "to do" instead of a place
Example 4: "but you were right." → ״אבל אתה צודק״ instead of "אבל היית צודק."
Example 5: "I'm sorry I was so hard on you before." → ״מצטער שהייתי קשה איתך קודם״ instead of ״ אני מצטער שהייתי כל כך קשה עליך קודם.״
Example 6: "Who does she work for?" → "אצל מי היא עובדת?" instead of ״למי היא עובדת״

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
Example 3: "You should have never gone there" → ״לא היית צריך ללכת לשם״

- Right-To-Left Punctuation
Example 1: "Hello, World!" → "שלום, עולם!".
Example 2: "did You go There?" → "הלכת לשם?".

- Appropriate Use of Prepositions for Possessive ConstructionsEnsure accurate use of prepositions like "ב" and "ל" in Hebrew translations to reflect correct possessive forms and relationships.
Example 1: "What size Cadillac do you take?" → Correct translation: "באיזה גודל קדילק אתה לוקח?" (Avoid using "ב" incorrectly for possessive constructs).
Example 2: "Suggesting that our customers have a..." → Correct translation: "לרמוז שללקוחות שלנו יש..." (Use "ל" correctly to indicate suggestion or implication).
Example 3: "The teacher's book" → "הספר של המורה" (Correct use of "של" for showing possession).
Example 4: "In the garden" → "בגן" (Correct use of "ב" for indicating a physical location).
"""


def get_openai_messages(*, target_language, rows):
    messages = [
        {"role": 'system', 'content': language_rules()},
        {'role': 'user',
         'content': f'Translate the each row of this subtitles to {target_language}, And return the translated rows by their corresponding index as valid JSON structure. \nRows:\n {json.dumps({row.index: row.content for row in rows})}'},
    ]
    return messages


def validate_and_parse_answer(json_str: str, *, preferred_suffix='}'):
    """
    validate the answer from openai and return the parsed answer
    """
    pop_last = False
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
                    pop_last = True
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
    if len(data) > 2 and pop_last:
        # we don't trust the fixer or openai, so we remove the last index,
        # which is usually a shitty translation anyway.
        key, val = data.popitem()
        last_index = key
    try:
        last_index = int(last_index)
    except:
        raise ValueError(f'last index is not an integer, last index is {last_index}')
    return data, last_index


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
    t1 = time.time()
    response = await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                  num_options=num_options, max_tokens=max_tokens, stream=True)
    collected_chunks = []
    async for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message:
            collected_chunks.append(chunk_message)

    json_str = ''.join(collected_chunks)
    answer, last_idx = validate_and_parse_answer(json_str, preferred_suffix=preferred_suffix)
    return answer, last_idx


async def translate_via_openai_func(
        *,
        rows: list['SRTBlock'],
        target_language: str,
        model: Literal['best', 'good'] = 'best',
        **kwargs
) -> dict[str, str]:
    """
    :param model: GPT Model to use
    :param rows: list of srt rows to translate
    :param target_language: translation's target language

    :returns a dict from row index to its translation and the last index that was NOT translated,
             if -1 then all rows were translated.
    """
    t1 = time.time()
    if not rows:
        return {}
    messages = get_openai_messages(rows=rows, target_language=target_language)
    answer, last_idx = await make_openai_request(messages=messages, seed=_SEED, model=model, temperature=.1)
    t2 = time.time()
    logging.info(
        'finished translating via openai',
        extra={'took': t2 - t1, 'len_returned_rows': len(answer), 'amount_rows': len(rows)}
    )
    return answer
