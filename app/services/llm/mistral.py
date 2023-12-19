# import json
#
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
#
# api_key = 'UHSKXOyzblj7hjwh2BczNEmbZjbzyghm'
# model = "mistral-medium"
#
# client = MistralClient(api_key=api_key)
#
#
# def translate_mixtral(sentences: list[str]):
#     messages = [
#         ChatMessage(role="system",
#                     content="You are a TV subtitles translator from English To Hebrew. Your job is to translate given subtitles with Best quality inflection, time tenses, using context to understand proper gender terms and so on.."),
#         ChatMessage(role="user",
#                     content=f"Translate the following sentences to English, Return a valid JSON OBJECT Mapping from sentence index to it's translation.\nSentences: {json.dumps({i: s for i, s in enumerate(sentences, start=1)})}")
#     ]
#     _SEED = 189
#     # No streaming
#     chat_response = client.chat(
#         model=model,
#         messages=messages,
#         max_tokens=4096,
#         random_seed=_SEED,
#         temperature=0.4
#     )
#     return chat_response
#
#
# if __name__ == '__main__':
#     s = [
#         "Because I clean it so well.",
#         "clean the candles, turn the lights on to the right setting,",
#         "You have a big day tomorrow.",
#         '''Has anyone ever come up to you and said, "You're not creative.'''
#     ]
#     d = translate_mixtral(s)
#     print(d)