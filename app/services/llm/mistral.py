import logging
import time

import json_repair
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage

api_key = 'UHSKXOyzblj7hjwh2BczNEmbZjbzyghm'
model = "mistral-medium"

client = MistralAsyncClient(api_key=api_key, timeout=60 * 5)

_SEED = 523


async def fix_json(json_str: str) -> dict | list:
    messages = [ChatMessage(role='system',
                            content="You are the JSON Fixer, your job is to receive a broken JSON and return a fixed version of it. the JSON can be invalid because of a character in the middle of the JSON. RETURN ONLY THE OUTPUT FIXED JSON, MAKE SURE IT'S IN VALID JSON STRUCTURE."),
                ChatMessage(role="user", content=json_str)]
    t1 = time.time()
    chat_response = await client.chat(
        model=model,
        messages=messages,
        max_tokens=4096,
        random_seed=_SEED,
        temperature=0.1
    )
    ret = json_repair.loads(chat_response.choices[0].message.content)
    t2 = time.time()
    logging.info(f'Fixed JSON in {t2 - t1} seconds')
    return ret
