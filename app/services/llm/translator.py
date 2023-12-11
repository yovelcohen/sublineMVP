import json
import logging
import time
from typing import Literal

from app.common.models.core import SRTBlock
from services.llm.llm_base import encoder, make_openai_request, TokenCountTooHigh


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
Example 3: "I was going to do something" → ״עמדתי לעשות משהו״ instead of ״הייתי הולך ל״ - adjusted because of the "to do" instead of a place

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
"""


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
) -> dict[str, str]:
    """
    :param model: GPT Model to use
    :param rows: list of srt rows to translate
    :param target_language: translation's target language
    :param tokens_safety_buffer: a safety buffer to remove from max tokens, in order to avoid openai token limit errors

    :returns a dict from row index to its translation and the last index that was NOT translated,
             if -1 then all rows were translated.
    """
    t1 = time.time()
    if not rows:
        return {}
    messages = get_openai_messages(rows=rows, target_language=target_language)
    data_token_cost = len(encoder.encode(json.dumps(messages)))
    max_completion_tokens = 16000 - (data_token_cost + tokens_safety_buffer)
    if max_completion_tokens < data_token_cost:
        raise TokenCountTooHigh(f'openai token limit is 16k, and the data token cost is {data_token_cost}, '
                                f'please reduce the number of rows')
    answer, last_idx = await make_openai_request(messages=messages, seed=99, model=model, temperature=.1)
    t2 = time.time()
    logging.info('finished translating via openai, took %s seconds', t2 - t1)
    return answer
