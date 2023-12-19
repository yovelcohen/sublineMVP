from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = 'google/madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name)


def main():
    text = "<2he> These here. See that? Becks."
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids)
    t = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return t


if __name__ == '__main__':
    main()
