import json
import logging
import time
from typing import Literal

import json_repair

from common.models.core import SRTBlock
from services.llm.llm_base import send_request

_SEED = 918


def language_rules():
    return """You are a proficient TV shows English to Hebrew Translator, Your Job is to review and fix the work of your fellow translators. You will be paid handsomely for every corrected/improved translation you provide.

Rules:
Rule: Ensure the translation reflects modern Hebrew slang and day2day language if used in the English source.
Example 1: English: "That movie was lit!" | Poor Translation: "הסרט היה מואר" | Improved Translation: "הסרט היה מגניב"
Example 2: English: "She ghosted him." | Poor Translation: "היא הפכה לרוח" | Improved Translation: "היא נעלמה ממנו"
Right-to-Left Alignment

Rule: Ensure the text is correctly aligned and formatted for Hebrew’s right-to-left reading order.
Example 1: English: "I love this book." | Poor Formatting: ".אני אוהבת את הספר הזה" | Correct Formatting: "אני אוהבת את הספר הזה."
Example 2: English: "He said, 'Hello!'" | Poor Formatting: "'!שלום', אמר" | Correct Formatting: "אמר, 'שלום!'"
Proper Punctuation

Rule: Adapt punctuation to fit Hebrew grammar and usage.
Example 1: English: "What time is it?" | Poor Punctuation: "מה השעה" | Correct Punctuation: "מה השעה?"
Example 2: English: "Firstly, we need to talk." | Poor Punctuation: "ראשית, אנחנו צריכים לדבר" | Correct Punctuation: "ראשית, אנחנו צריכים לדבר."
Time Tenses Accuracy

Rule: Ensure time tenses are accurately translated to reflect the original meaning.
Example 1: English: "I will go tomorrow." | Poor Translation: "הלכתי מחר" | Improved Translation: "אני אלך מחר"
Example 2: English: "She was running." | Poor Translation: "היא רצה" | Improved Translation: "היא רצה"
Verb Conjugation Accuracy

Rule: Check that verbs are correctly conjugated according to Hebrew grammar.
Example 1: English: "They are eating." | Poor Translation: "הם אוכל" | Improved Translation: "הם אוכלים"
Example 2: English: "You (female) understood." | Poor Translation: "הבנתם" | Improved Translation: "הבנת"
Consistency in Translating Names and Proper Nouns

Rule: Maintain consistency in the translation of names and proper nouns.
Example 1: English: "I visited Berlin and Paris." | Inconsistent Translation: "ביקרתי בברלין ופריז" | Consistent Translation: "ביקרתי בברלין ובפריז"
Example 2: English: "Harry Potter is a popular character." | Inconsistent Translation: "הארי פוטר הוא דמות פופולרית" | Consistent Translation: "הארי פוטר הוא דמות פופולרית"
Cultural and Contextual Accuracy

Rule: Ensure cultural references and idioms are appropriately translated or adapted.
Example 1: English: "It's raining cats and dogs." | Literal Translation: "זה גשם חתולים וכלבים" | Contextual Translation: "יורד גשם כמו שוטפים"
Example 2: English: "Break a leg!" | Literal Translation: "שבור רגל!" | Contextual Translation: "בהצלחה!"

Rule: Make sure you follow the following rules and output a valid JSON, no matter what!
"""


def get_revisor_messages(*, rows, target_language):
    data = {row.index: {'og': row.content, 'translation': row.translations.content} for row in rows}
    messages = [
        {"role": 'system', 'content': language_rules()},
        {
            'role': 'user',
            'content': f"""Review the Translation of these subtitles from English to {target_language}, And offer a better translation based on these rules: 
- More day2day language and casual translation
- A translation which corrects errors such as gender misuse, time tenses errors, syntax adaptation and so on.. in the original translation. 

Base your answer on your Rules. Return a Valid JSON Structure Where the keys are the original keys from Rows and values are the suggested {target_language} new translation.
Rows: {json.dumps(data)}"""
        }
    ]
    return messages


async def make_request(messages, seed, model, max_tokens=None, top_p=None, temperature=None, num_options=None):
    choices = await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                 num_options=num_options, max_tokens=max_tokens)
    choice = choices[0]
    return json_repair.loads(choice.message.content)


def clean_resp(resp):
    ret = {}
    for k, v in resp.items():

        if k.isdigit() and isinstance(v, dict):
            key = [i for i in v.keys() if i.lower().strip() != 'og'][0]
            ret[k] = v[key]
        else:
            raise ValueError(f"Can't serialize value: {v}")
    return ret


def pprint_from_resp(ret, rows):
    rows = {row.content.replace('\n', ' ').strip(): row for row in rows}
    for i in ret:
        if row := rows.get(i[1]['og'].replace('\n', ' ').strip()):
            if row.translations.content != i[1]['translation']:
                print(i[1]['og'].replace('\n', ' ').strip())
                print(row.translations.content)
                print(i[1]['translation'])
                print('----')


async def review_revisions_via_openai(
        *,
        rows: list['SRTBlock'],
        target_language: str,
        model: Literal['best', 'good'] = 'best'
) -> dict[str, str]:
    """
    :param model: GPT Model to use
    :param rows: list of srt rows to translate
    :param target_language: translation's target language

    :returns a dict from row original english to its translation.
    """
    t1 = time.time()
    if not rows:
        return {}
    messages = get_revisor_messages(rows=rows, target_language=target_language)
    ret = await make_request(messages=messages, seed=_SEED, model=model, temperature=.15)
    ret = clean_resp(ret)
    t2 = time.time()
    logging.info('finished translating revisions via openai, took %s seconds', t2 - t1)
    return ret


def get_refiner_message(*, original_to_translation, target_language):
    messages = [
        {"role": 'system', 'content': language_rules()},
        {'role': 'user',
         'content': f'Review these Translations of a TV episode from English to {target_language}, your job is to find outstanding bad translations, in terms of your rules and general flow of the script. Return ONLY the rows that you deem fit to replace, by their corresponding original text mapping to the new offered translation, as valid JSON structure. \nRows:\n {json.dumps(original_to_translation)}'},
    ]
    return messages
