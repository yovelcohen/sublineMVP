import json
import logging
import time
from typing import Literal

import json_repair
from openai.types import CompletionUsage
from openai.types.edit import Choice

from common.models.core import SRTBlock
from services.llm.llm_base import send_request, encoder, report_stats

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
Example 7: "I've been sleeping with him" → ״אני שוכבת איתו״ instead of ״אני ישנה איתו״ → based on the fact that most of the times, in media, sleeping with someone refers to sex and not actually sleep.
Example 8: "Look, I've been going there a while" → "תראה אני הולך לשם כבר תקופה" → Usage of תראה instead תסתכל is more common in day2day hebrew. 
Example 9: "He was like, "I bought a fancy cat," → "הוא אמר, קניתי מכונית מפוארת" instead of "הוא היה כמו, קניתי מכונית מפוארת," → Usage of "Like" in modern english slang should be sort of ignored.  

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
Example 6: "For your information" → "לידיעתך" instead of "למידע שלך"

- Syntax Adaptation and proper time tenses usage, Adjust sentence structure for Hebrew syntax.
Example 1: "She loves dogs" (Subject-Verb-Object in English) → "היא אוהבת כלבים" (Hebrew structure).
Example 2: "I am reading a book" → Adjusted to fit Hebrew verb-subject-object order.
Example 3: "I was going to do something" → ״עמדתי לעשות משהו״ instead of ״הייתי הולך ל״ - adjusted because of the "to do" instead of a place
Example 4: "but you were right." → ״אבל אתה צודק״ instead of "אבל היית צודק."
Example 5: "I'm sorry I was so hard on you before." → ״מצטער שהייתי קשה איתך קודם״ instead of ״ אני מצטער שהייתי כל כך קשה עליך קודם.״
Example 6: "Who does she work for?" → "אצל מי היא עובדת?" instead of ״למי היא עובדת״
Example 7: "See, it's a little tricky" → "תראה, זה טיפה טריקי" instead of ״ראה״ or "תסתכל"

- Inflection and Slang Adjustment: Adapt English phrases to their natural, colloquial Hebrew equivalents, focusing on inflection and daily language usage.
Example 1: English "This is only half." → Hebrew "זה רק חצי." instead of "זה רק המחצית." (Applying a more natural inflection).
Example 2: "which gave you everything you wanted" → "שנתנה לך כל מה שרצית" instead of "שנתנה לך הכל שרצית." (Adapting to a more colloquial expression).
Example 3: "It's an honorary position." → "זה תפקיד כבוד" instead of "זו תפקיד כבודי." (Using a more commonly spoken phrase structure).
Example 4: "I'm really tired." → "אני ממש עייף." or "אני גמור." instead of a direct translation like "אני מאוד עייף." (Adapting to a more natural, colloquial expression in Hebrew).
Example 5: English "I'm starving." → Hebrew "אני מת מרעב." instead of a more literal translation like "אני רעב מאוד." (Using a colloquial expression that captures the intensity of hunger in everyday speech).
Example 6: "Let's hang out." → "בוא נתלווה." or "בוא נבלה זמן יחד." instead of a direct translation like "בוא נהיה יחד." (Choosing a phrase that more accurately reflects the casual and friendly nature of the invitation in everyday Hebrew).
Example 7: "My brother has every right to be here." → "לאח שלי יש זכות מלאה להיות כאן" instead of "לאח שלי יש לו הזכות להיות פה"

- Consistent Transliteration: Keep transliterations of names and terms consistent.
Example 1: "John" always transliterated the same way in Hebrew.
Example 2: Technical terms like "Internet" consistently transliterated.
Example 3: Companies, Organizations and Product names should be translated as is.

- Subtitle Length and Timing: Keep subtitles concise and timed well.
Example 1: Shortening a lengthy English sentence to fit the Hebrew screen time.
Example 2: Dividing a long English dialogue into shorter, readable Hebrew subtitles.

- Verb Conjugation Accuracy: Correctly conjugate Hebrew verbs.
Example 1: Translating "I ate" → "אכלתי" (past tense, first person singular).
Example 2: "They will go" → "הם ילכו" (future tense, third person plural).
Example 3: "You should have never gone there" → ״לא היית צריך ללכת לשם״
Example 4: "How the hell did you know?" → "לעזאזל, איך ידעת?" instead of "איך בעזאזל ידעת" (Using a more natural inflection).
Example 5: "You're gonna charm me." → "אתה הולך להקסים אותי" instead of "אתה הולך לקסום אותי" (Using a more natural inflection).

- Right-To-Left Punctuation
Example 1: "Hello, World!" → "שלום, עולם!".
Example 2: "did You go There?" → "הלכת לשם?".

- Appropriate Use of Prepositions for Possessive Constructions, Ensure accurate use of prepositions like "ב" and "ל" in Hebrew translations to reflect correct possessive forms and relationships.
Example 1: "What size Cadillac do you take?" → Correct translation: "באיזה גודל קדילק אתה לוקח?" (Avoid using "ב" incorrectly for possessive constructs).
Example 2: "Suggesting that our customers have a..." → Correct translation: "לרמוז שללקוחות שלנו יש..." (Use "ל" correctly to indicate suggestion or implication).
Example 3: "The teacher's book" → "הספר של המורה" (Correct use of "של" for showing possession).
Example 4: "In the garden" → "בגן" (Correct use of "ב" for indicating a physical location).
Example 5: "They were grateful" → "הם הודו" instead of "הם היו מודים" (Using a more natural inflection). 
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
    response = await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                  num_options=num_options, max_tokens=max_tokens, stream=True)
    collected_chunks = []
    async for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message:
            collected_chunks.append(chunk_message)

    json_str = ''.join(collected_chunks)
    usage = CompletionUsage(
        completion_tokens=len(encoder.encode(json_str)),
        prompt_tokens=len(encoder.encode(json.dumps(messages))),
        total_tokens=len(encoder.encode(json_str)) + len(encoder.encode(json.dumps(messages)))
    )
    report_stats(usage)

    logging.debug(f'openai request finished, token cost: {usage.model_dump()}')
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
