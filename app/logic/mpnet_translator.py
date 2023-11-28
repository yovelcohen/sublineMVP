# import re
#
# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
#
# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# pattern = re.compile(r'(\d+)\s(.*?)(?=\s\d+\s|\s*$)')
# lang_conv = {'he': 'he_IL', 'en': 'en_XX', 'ar': 'ar_AR', 'fr': 'fr_XX', 'es': 'es_XX',
#              'ru': 'ru_RU', 'de': 'de_DE', 'zh': 'zh_CN'}
#
#
# def translate_local(text, source_lang, target_lang):
#     source_lang = lang_conv[source_lang]
#     target_lang = lang_conv[target_lang]
#     tokenizer.src_lang = source_lang
#     model_inputs = tokenizer(text, return_tensors="pt")
#     generated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
#     ok = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#     pairs = pattern.findall(ok[0])
#     return dict(pairs)
