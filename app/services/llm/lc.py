from langchain.chat_models import ChatOpenAI
from langchain_experimental.smart_llm import SmartLLMChain

chain = SmartLLMChain(llm=ChatOpenAI(openai_api_key='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',
                                     model_name='', temperature=.2, max_tokens=4096),
                      return_intermediate_steps=True, verbose=True)
