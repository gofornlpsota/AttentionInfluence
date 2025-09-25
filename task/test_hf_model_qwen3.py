import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer

# pip3 install transformers==4.56.1
# pip3 install accelerate==1.10.1
# model_path = 'Qwen/Qwen3-1.7B-Base'
# tokenizer_path = 'Qwen/Qwen3-1.7B-Base'

model_path = 'Qwen/Qwen3-0.6B-Base'
tokenizer_path = 'Qwen/Qwen3-0.6B-Base'

model = AutoModelForCausalLM.from_pretrained(model_path,  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = False)
model.eval()
device = model.device

model_input = tokenizer(
    "Hello, how are you?", 
    return_tensors="pt",
    max_length=20,
    truncation=True
)
# model_input["input_ids"] = model_input["input_ids"].to("cuda")
model_input["input_ids"] = model_input["input_ids"].to(device) # 不只在一张卡上，所以得和model的device对齐
model_input["attention_mask"] = model_input["attention_mask"].to(device)
# generate
model_output = model.generate(model_input['input_ids'], max_new_tokens=50)
print(model_output)
output_string = tokenizer.batch_decode(model_output)[0]
print("Output with `.generate()`:\n" + output_string)
print("\n")

model_output = model(**model_input)
print(model_output.logits.shape)
output_string = tokenizer.decode(torch.argmax(model_output.logits.squeeze(), -1))
print("Output with `.forward()`:\n" + output_string) # context里面做argmax，不会非常高比例和context一致，这样解码出来的结果会有问题
print(model_output.logits)

loss_fn = partial(torch.nn.functional.cross_entropy, ignore_index=-100, reduction='none')
def cal_loss_from_logits(logits, labels):
    logits = logits[:, :-1, :] # [b, s, v]
    labels = labels[:, 1:] # [b, s]
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)
    loss = loss_fn(logits, labels)
    loss = loss.reshape(bsz, seq_len)
    print(loss.shape)
    print(loss)
    loss = loss.detach().cpu().numpy().tolist()
    return loss

loss = cal_loss_from_logits(model_output.logits, model_input['input_ids'])
full_str = ''
print(model_input['input_ids'])
print(loss)
for example_token_ids, example_token_loss_list in zip(model_input['input_ids'], loss):
    for token_id, token_loss in zip(example_token_ids, example_token_loss_list):
        full_str += f'{tokenizer.decode(token_id)}: {token_loss}\n'
    full_str += '\n===\n'
print(full_str)