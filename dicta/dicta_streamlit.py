import streamlit as st
from streamlit_chat import message
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

random.seed(None)


@st.cache_resource
def load_model(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                  quantization_config=quantization_config)
    return _model, _tokenizer


def extend(input_text, max_size=20, top_k=50, top_p=0.95, tmp=0.75):
    if len(input_text) == 0:
        input_text = ""
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        kwargs = dict(
            inputs=tokenizer(input_text, return_tensors='pt').input_ids.to(model.device),
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=tmp,
            max_length=max_size,
            min_new_tokens=5
        )
        answer = (tokenizer.batch_decode(model.generate(**kwargs), skip_special_tokens=True))
    return answer


st.title("dictaLM")
pre_model_path = "dicta-il/dictalm-7b-instruct"
model, tokenizer = load_model(pre_model_path)

np.random.seed(None)
random_seed = np.random.randint(10000, size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = 0 if torch.cuda.is_available() == False else torch.cuda.device_count()

torch.manual_seed(random_seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(random_seed)

st.sidebar.subheader("Configurable parameters")
_max_len = st.sidebar.slider("Max-Length", 0, 192, 96, help="The maximum length of the sequence to be generated.")
_top_k = st.sidebar.slider("Top-K", 0, 100, 40,
                           help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
_top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 0.92,
                           help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.75, help="TO COME")


def on_input_change():
    user_input = st.session_state.user_input
    result = extend(
        input_text=user_input,
        top_k=int(_top_k),
        top_p=float(_top_p),
        max_size=int(_max_len),
        tmp=float(temperature)
    )
    print(result)
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        message(user_input, is_user=True)
        message(result[0])


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


st.session_state.setdefault(
    'past',
    []
)
st.session_state.setdefault(
    'generated', []
)

st.markdown(
    """hebrew chat based on dictaLM"""
)

with st.container():
    st.text_input("User Input:", on_change=on_input_change, key="user_input")
st.button("Clear message", on_click=on_btn_click)
