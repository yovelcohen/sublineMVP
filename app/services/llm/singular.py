import asyncio
import json
import logging

from services.llm.llm_base import send_request

_SEED = 563

translation_obj = lambda prefix: {
    "type": "object",
    "description": f"{prefix} New Translation Revision and its score",
    "properties": {"content": {"type": "string"}, "score": {"type": "number"}}
}

tools = [
    {'type': 'function',
     'function': {
         'name': 'correct_bad_subtitle_translation',
         "description": """Given an English subtitle sentence and its bad Hebrew translation, Offer 3 improved translations.
Make Sure to fix the following issues: 
- Misspelling.
- Wrong punctuation.
- Made Up words.
- Language Too Formal, not using slang and day2day language.

You will be paid for corrections you provide, You need to score yourself on a scale of 1-10, on how good your translation is.
""",
         "parameters": {
             "type": "object",
             "properties": {
                 "new_translation_1": translation_obj('1st'),
                 "new_translation_2": translation_obj('2nd'),
                 "new_translation_3": translation_obj('3rd')
             }
         }
     }}
]


async def get_suggestions(source_sentence, translation):
    r1 = await send_request(messages=[{"role": "user", "content": json.dumps({'source_sentence': source_sentence,
                                                                              'translation': translation},
                                                                             ensure_ascii=False)}],
                            tools=tools,
                            tool_choice={"type": "function", "function": {"name": "correct_bad_subtitle_translation"}},
                            temperature=.3, model='gpt-4', seed=_SEED)
    smart_lower = r1[0].message.tool_calls[0].function.arguments
    return json.loads(smart_lower)


def logging_setup():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(asctime)s %(name)s:%(message)s", force=True)
    logging.getLogger('httpcore').setLevel(logging.INFO)
    # logging.getLogger('openai').setLevel(logging.INFO)


if __name__ == '__main__':
    logging_setup()
    ret = asyncio.run(
        get_suggestions('and now she running shit just like that?', 'ועכשיו היא מנהלת את העסק כל כך בטחונית?')
    )
    print(ret)
