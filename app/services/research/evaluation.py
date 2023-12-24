import asyncio
import json
import logging

from beanie.odm.operators.find.comparison import In

from common.config import settings
from common.db import init_db
from common.models.core import TranslationStates, Translation
from services.llm.llm_base import send_request, Model

func = [
    {
        'name': 'evaluate_translations',
        'description': """Please score the following translated TV subtitles from English to Hebrew with respect to the original English script on a continuous scale from 0 to 100. 
Return a JSON mapping from row index to its translation score.

A score of zero means "no meaning preserved" and a score of one hundred means "perfect meaning and grammar", Based on the following rules:

- Simplify Complex Sentences: If the English sentence is complex, break it into simpler sentences in Hebrew.
English: "Despite the rain, the event, which was attended by many, was successful."
Hebrew: "למרות הגשם, האירוע היה מוצלח. רבים השתתפו בו."

- Word Order: Maintain the original word order except when changing it improves clarity in Hebrew.
English: "The quick brown fox jumps over the lazy dog."
Hebrew: "השועל החום המהיר קופץ מעל הכלב העצלן."

- Impersonal Pronouns: Replace 'it' with specific nouns or phrases.
English: "It's important to consider all options."
Hebrew: "חשוב לשקול את כל האופציות."

- Minimize 'You' and 'Your': Use these less frequently in Hebrew.
English: "Once you’ve chosen your items, proceed to checkout."
Hebrew: "לאחר בחירת הפריטים, המשך לתשלום."

- Verbs over Nouns: Use verbs to convey actions.
English: "The cancellation of the event was unexpected."
Hebrew: "ביטול האירוע היה בלתי צפוי."

- Consistent Bulleted Lists: Match verb form and structure to the introductory phrase.
English: "To prepare, you should: gather materials, set up the workspace, and review the instructions."
Hebrew: "להכנה יש ל: אסוף חומרים, להכין את מקום העבודה ולעיין בהוראות."

- Gender-Neutral Forms: Use plural or impersonal forms for imperative sentences.
English: "Remember to save your work."
Hebrew: "זכרו לשמור את העבודה שלכם."

- Active vs. Passive Voice: Prefer active voice in Hebrew.
English: "The novel was written by the author in one month."
Hebrew: "הסופר כתב את הרומן בחודש אחד."

- Verb Tenses: Adjust tenses appropriately for Hebrew.
English: "They have lived in Paris for three years."
Hebrew: "הם גרים בפריז שלוש שנים."

- Construct States and Modifiers: Pay attention to noun forms in Hebrew.
English: "The manager's decision"
Hebrew: "החלטת המנהל" """,
        'parameters': {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "object",
                    "description": "Mapping from row index to its translation score"
                }
            }
        }
    }
]

_SEED = 7693


async def score_one(translation: Translation):
    logging.info('scoring %s', translation.name)
    rows = sorted(list(translation.subtitles), key=lambda x: x.index)
    rows = {
        row.index: {'en': row.content, 'he': row.translations.content} for row in rows if row.translations is not None
    }
    messages = [{'role': 'user', 'content': json.dumps(rows)}]
    ret = await send_request(messages=messages, model=Model.GPT4_32K, temperature=.5, functions=func, seed=_SEED)
    return translation.name, ret


async def evaluate_subtitles(names: list[str] = None):
    await init_db(settings, [Translation])
    q = Translation.find_many({'state': TranslationStates.DONE.value})
    if names:
        q = q.find_many(In(Translation.name, names))
    translations = await q.to_list()
    results = await asyncio.gather(*[score_one(t) for t in translations])
    return dict(results)


def logging_setup():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s %(name)s:%(message)s",
        force=True,
    )  # Change these settings for your own purpose, but keep force=True at least.
    logging.getLogger('httpcore').setLevel(logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)


if __name__ == '__main__':
    logging_setup()
    _ret = asyncio.run(evaluate_subtitles())
    print(_ret)
