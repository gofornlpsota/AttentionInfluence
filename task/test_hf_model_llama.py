import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/opt/tiger/ckpt/Llama-2-7b-hf'
tokenizer_path = '/opt/tiger/ckpt/Llama-2-7b-hf'

model = AutoModelForCausalLM.from_pretrained(model_path,  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = False)
tokenizer.pad_token = '[PAD]'
tokenizer.pad_token_id = 3
model.eval()

model_input = tokenizer(
    "Hello, how are you?", 
    return_tensors="pt",
    max_length=20,
    truncation=True
)
model_input["input_ids"] = model_input["input_ids"].to("cuda")
model_input["attention_mask"] = model_input["attention_mask"].to("cuda")
model_output = model.generate(model_input['input_ids'], max_new_tokens=50)
print(model_output)
output_string = tokenizer.batch_decode(model_output)[0]
print("Output with `.generate()`:\n" + output_string)
print("\n")

model_output = model(**model_input)
print(model_output.logits.shape)
output_string = tokenizer.decode(torch.argmax(model_output.logits.squeeze(), -1))
print("Output with `.forward()`:\n" + output_string)