import asyncio
import json
import logging

from pydantic import TypeAdapter

from app.common.config import settings
from app.common.db import init_db
from app.common.models.core import SRTBlock, Translation
from app.services.llm.llm_base import send_request

_SEED = 192


def get_messages(target_language: str, rows: list[SRTBlock]):
    rows = json.dumps(
        {row.content: {"translation_1": row.translations.content, "translation_2": row.translations.revision}
         for row in rows},
        ensure_ascii=False
    )
    return [
        {'role': 'system',
         'content': f"You are proficient in Translating TV Shows from English To {target_language}. You're Job is to review translations from the User, and highlight any mistakes you find. You will be paid for each mistake you find."},
        {
            'role': 'user',
            'content': f"""Review the following subtitles translation, each key in the JSON is the original English and values 
are 2 Hebrew translations to review by your rules.
The object is ordered in terms of subtitles flow. Return a VALID JSON Object of rows you think none of the translations 
work in the context and offer a revised translation, RETURN ONLY THE ROWS YOU THINK NEED FIXING. 
Keys of the returned object should be the original English. 

You will get paid for errors found and extra for errors fixed.

Example Input/Output:
Input: {{"I was going to take it": {{ "translation 1": "אני הייתי הולך לקחת את זה", "translation 2": "עמדתי ללכת לקחת את זה" }}, "He's a real piece of shit.": {{ "translation 1": "הוא פרצוף אמיתי של חרא.", "translation 2": "הוא פרטי אמיתי של חרא."}}, "Trust me, Beth, you don't want to know": {{ "translation 1": "תסתמי, בת, את לא רוצה לדעת.", "translation 2": "תסמוך עלי, בת, את לא רוצה לדעת" }}, "I guess I take back what I said
about British cuisine.": {{ "translation 1": "אני מניח שאני מתחזק בדבריי על המטבח הבריטי.", "translation 2": "אני מניח שאני מתחזק על מה שאמרתי על המטבח הבריטי."}} }}
Output: {{ "He's a real piece of shit.": "הוא חתיכת חרא.", "Trust me, Beth, you don't want to know": "תסמכי עליי, בת, את לא רוצה לדעת", "I guess I take back what I said
about British cuisine.": "אני מניח שאקח בחזרה את מה שאמרתי על המבטח הבריטי." }}

Rows: {rows}"""
        },
    ]


async def audit_results_via_openai(rows: list[SRTBlock], target_language: str):
    messages = get_messages(target_language=target_language, rows=rows)
    ret = await send_request(seed=_SEED, model='best', messages=messages, temperature=.2)
    return ret


async def main():
    await init_db(settings, [Translation])
    with open('/Users/yovel.c/PycharmProjects/services/sublineStreamlit/app/services/results.json', 'r',
              encoding='utf-8') as f:
        data = json.loads(f.read())
        data: Translation = TypeAdapter(Translation).validate_python(data)
    resp = await audit_results_via_openai(rows=data.subtitles, target_language=data.target_language)
    return resp


def logging_setup():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(asctime)s %(name)s:%(message)s", force=True)
    logging.getLogger('httpcore').setLevel(logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)


if __name__ == '__main__':
    logging_setup()
    _ret = asyncio.run(main())
    print(_ret)
