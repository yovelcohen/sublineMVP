import asyncio
import json
import logging

import json_repair

from common.config import settings
from common.db import init_db
from common.models.core import SRTBlock, Translation
from services.llm.llm_base import send_request

_SEED = 192


def slice_dict_into_three(d):
    n = len(d)
    part_size = n // 3
    extra = n % 3

    parts = [{}, {}, {}]
    iter_items = iter(d.items())

    for i in range(3):
        for _ in range(part_size + (i < extra)):
            key, value = next(iter_items)
            parts[i][key] = value

    return parts


# Example usage
def get_messages(target_language: str, rows: dict):
    example_input_output_str = json.dumps([
        {
            "Input": {"1": {"og": "I was going to take it", "1": "אני הייתי הולך לקחת את זה",
                            "2": "עמדתי ללכת לקחת את זה"},
                      "2": {"og": "He's a real piece of shit.", "1": "הוא פרצוף אמיתי של חרא.",
                            "2": "הוא פרטי אמיתי של חרא."},
                      "3": {"og": "Trust me, Beth, you don't want to know",
                            "1": "תסתמי, בת, את לא רוצה לדעת",
                            "2": "תסמוך עלי, בת, את לא רוצה לדעת"},
                      "4": {"og": "I guess I take back what I said about British cuisine.",
                            "1": ".אני מניח שאני מתחזק בדבריי על המטבח הבריטי",
                            "2": "אני מניח שאני מתחזק על מה שאמרתי על המטבח הבריטי."}},
            "Output": {"1": "2",
                       "2": "2",
                       "3": "תסמכי עליי בת, את לא רוצה לדעת.",
                       "4": "אני מניח שזה מחזק את דבריי על המטבח הבריטי"}
        },
        {
            "Input": {
                "5": {
                    "og": "You're so busted.",
                    "1": "אתה תפוס כלכך",
                    "2": "אתה כל כך נתפס."
                },
                "6": {
                    "og": "They vanished into thin air.",
                    "1": "הם נעלמו לחלוטין.",
                    "2": "הם פשוט נעלמו באוויר הדק."
                },
                "7": {
                    "og": "That's the last straw!",
                    "1": "זה הקש האחרון!",
                    "2": "זהו הגבול!"
                },
                "8": {
                    "og": "I can't believe you just said that.",
                    "1": "אני לא מאמין שאתה פשוט אמרת את זה.",
                    "2": "אני לא מאמינה שפשוט אמרת את זה."
                },
                "9": {
                    "og": "Let's shake on it.",
                    "1": "בוא ניתן יד על זה.",
                    "2": "בואו נתחבק על זה."
                }
            },
            "Output": {
                "5": "כלכך נתפסת.",
                "6": "1",
                "7": "זהו, זה הגבול!",
                "8": "2",
                "9": "בוא נלחץ ידיים על זה."
            }
        }, {
            "Input": {
                "1": {
                    "og": "Can you believe this weather?",
                    "1": "אתה יכול להאמין למזג האוויר הזה?",
                    "2": "אפשר להאמין למזג האוויר הזה?"
                },
                "2": {
                    "og": "It's been raining for weeks!",
                    "1": "זה גשם כבר שבועות!",
                    "2": "הגשם נמשך שבועות!"
                },
                "3": {
                    "og": "I forgot what the sun looks like.",
                    "1": "שכחתי איך השמש נראית.",
                    "2": "שכחתי איך נראית השמש."
                },
                "4": {
                    "og": "At least the plants are happy.",
                    "1": "לפחות הצמחים מרוצים.",
                    "2": "לפחות הצמחים שמחים."
                }
            },
            "Output": {
                "1": "1",
                "2": "1",
                "3": "2",
                "4": "לפחות הצמחים מרוצים."
            }
        }, {
            "Input": {
                "15": {"og": "Look at your face, you're so fucked.", '1': 'מבט אל פניך, אתה כל כך נעקץ.',
                       "2": "אתה נתפס בפשע. מבט אל פניך, אתה נתפס בפשע."},
                "16": {
                    "og": "Whistler! What do you got to tag and bag these bastards?",
                    "1": "שרירות! מה יש לך לתייג ולארוז את הבגדנים האלה?",
                    "2": None
                },
                "17": {
                    "og": "This is beyond my wildest dreams!",
                    "1": "זה מעבר לחלומות הכי פרועים שלי!",
                    "2": "זה מעבר לדמיון הכי פראי שלי!"
                }
            },
            "Output": {"15": "תראה את המבט שלך, כלכך נדפקת.",
                       "16": "שורק! מה יש לך לתייג ולקחת את הממזרים האלה?",
                       "17": "2"}
        }, {
            "Input": {
                "1": {
                    "og": "She's been nailing every presentation lately.",
                    "1": "היא הצליחה בכל הצגה לאחרונה.",
                    "2": "היא דוקרת כל הצגה בזמן האחרון."
                },
                "2": {
                    "og": "Yeah, she's on fire!",
                    "1": "כן, היא בוערת!",
                    "2": "כן, היא על אש!"
                },
                "3": {
                    "og": "I'll ask her for some tips before the next meeting.",
                    "1": "אני אשאל אותה לטיפים לפני הפגישה הבאה.",
                    "2": "אני אבקש ממנה עצות לפני הפגישה הבאה."
                },
                "4": {
                    "og": "Make sure you do, she's got the Midas touch.",
                    "1": "תוודא שאתה עושה את זה, יש לה מגע של מידס.",
                    "2": "ודא שתעשה את זה, היא עם מגע של מידס."
                },
                "5": {
                    "og": "If only it were that easy!",
                    "1": "חבל שזה לא כל כך פשוט!",
                    "2": "אילו זה היה כל כך פשוט!"
                }
            },
            "Output": {
                "1": "1",
                "2": "כן, היא לוהטת",
                "3": "2",
                "4": "1",
                "5": "2"
            }
        }], ensure_ascii=False, indent=2)

    system_msg = f"""You are proficient in Translating TV Shows from English To {target_language}. 
You'll be given a JSON object of Subtitles by row index and two translations choices.
The translations were made by your co-workers
Your Job is to pick the better option between the two translations, But, If both translations are wrong according to your rules,
you should offer a new revised translation based on your Rules and the context of the row (use the rows above and below it to understand the flow).
 
Rules:
- Keys of the returned object should be the row index, Values are one of ["1", "2", "<new translation offered by you>"]
- If a 2nd translation version does not exist, offer an alternative translation yousrself, and then choose the best one.
- How To Measure a translation's quality
 - Correct Gender Translation  
 - Correct Use Of Time tenses and rules
 - Usage of modern day2day 21st language and slang
 - Correct translation of curses and idioms
 - Correct translation of names and proper nouns
 - Correct punctuation and alignment (right-to-left in Hebrew)
 
You will get paid for errors found and extra for errors fixed.

---
Example Input/Output:
{example_input_output_str}
---"""
    return [
        {'role': 'system', 'content': system_msg},
        {'role': 'user', 'content': json.dumps({'Input': rows}, ensure_ascii=False, indent=2)}
    ]


async def audit_results_via_openai(rows: set[SRTBlock], target_language: str):
    rows = {
        str(row.index): {"og": row.content, "1": row.translations.content, '2': row.translations.revision}
        for row in rows
    }
    msg = [get_messages(target_language=target_language, rows=chunk) for chunk in slice_dict_into_three(rows)]
    res = []
    for i, m in enumerate(msg):
        ret = await send_request(seed=_SEED, model='best', messages=m, temperature=.2)
        res.append(json_repair.loads(ret[0].message.content))
    return res


async def main():
    await init_db(settings, [Translation])
    translation_obj = await Translation.get('65795f75371873eeaae5270f')
    copy = translation_obj.model_copy(deep=True)
    resp = await audit_results_via_openai(rows=copy.subtitles, target_language=copy.target_language)
    return resp


def logging_setup():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(asctime)s %(name)s:%(message)s", force=True)
    logging.getLogger('httpcore').setLevel(logging.INFO)
    # logging.getLogger('openai').setLevel(logging.INFO)


if __name__ == '__main__':
    logging_setup()
    _ret = asyncio.run(main())
    print(_ret)
