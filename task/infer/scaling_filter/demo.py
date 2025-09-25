import random
from tqdm import tqdm
import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.data import load_data_from_file, save_data_to_json

# pip3 install transformers==4.56.1
# pip3 install accelerate==1.10.1
big_model_path = 'Qwen/Qwen3-1.7B-Base'
big_model_tokenizer_path = 'Qwen/Qwen3-1.7B-Base'
big_model = AutoModelForCausalLM.from_pretrained(big_model_path,  device_map="auto")
big_model.eval()
big_model_tokenizer = AutoTokenizer.from_pretrained(big_model_tokenizer_path, use_fast = False)

small_model_path = 'Qwen/Qwen3-0.6B-Base'
small_model_tokenizer_path = 'Qwen/Qwen3-0.6B-Base'

small_model = AutoModelForCausalLM.from_pretrained(small_model_path,  device_map="auto")
small_model_tokenizer = AutoTokenizer.from_pretrained(small_model_tokenizer_path, use_fast = False)
small_model.eval()
print(len(big_model_tokenizer))
print(len(small_model_tokenizer))
print(small_model_tokenizer.pad_token)
print(small_model_tokenizer.pad_token_id)

loss_fn = partial(torch.nn.functional.cross_entropy, ignore_index=-100, reduction='none')
# batch
def cal_loss_from_tokens(model_input, model, device, loss_fn, pad_token_id):
    model_input["input_ids"] = model_input["input_ids"].to(device)
    labels = model_input['input_ids']
    with torch.no_grad():
        model_output = model(**model_input)
        logits = model_output.logits
        return cal_loss_from_logits(logits, labels, loss_fn, pad_token_id)

def cal_loss_from_logits(logits, labels, loss_fn, pad_token_id):
    logits = logits[:, :-1, :] # [b, s, v]
    labels = labels[:, 1:] # [b, s]
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)
    token_loss = loss_fn(logits, labels)
    # mask
    # token_loss = token_loss.where(labels != pad_token_id, 0)
    mask = (labels != pad_token_id).float() # 有效位置为1，padding位置为0
    mask = mask.reshape(bsz, seq_len)
    token_loss = token_loss.reshape(bsz, seq_len)
    token_loss = token_loss * mask
    # print(token_loss.shape)
    mean_seq_loss = token_loss.sum(dim=1) / mask.sum(dim=1)
    mean_seq_loss = mean_seq_loss.detach().cpu().numpy().tolist()
    token_loss = token_loss.detach().cpu().numpy().tolist() # [b, s]
    return token_loss, mean_seq_loss

def process_texts(texts, tokenizer, max_seq_len):
    encoded_texts = [tokenizer(text) for text in texts]

    input_ids_list = [encoded["input_ids"] for encoded in encoded_texts]
    token_counts = [len(ids) for ids in input_ids_list]

    max_token_cnt = min(max(token_counts) if token_counts else 0, max_seq_len)

    processed_input_ids = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for ids in input_ids_list:
        if len(ids) > max_seq_len:
            processed = ids[:max_seq_len]
        else:
            pad_length = max_seq_len - len(ids)
            processed = ids + [pad_token_id] * pad_length
        
        processed_input_ids.append(processed)

    model_input = {
        "input_ids": torch.tensor(processed_input_ids)
    }

    return model_input, token_counts

def cal_loss_diff_from_texts(texts, max_seq_len, small_model, big_model, tokenizer, pad_token_id,loss_fn):
    model_input = small_model_tokenizer(
        texts, 
        return_tensors="pt",
        max_length=max_seq_len,
        padding=True, truncation=True
    )
    token_counts = [-1] * len(texts)

    # model_input, token_counts = process_texts(texts, tokenizer, max_seq_len) # 太慢了

    b_input_ids = model_input['input_ids']
    if random.random() < 0.001:
        print(f'input_ids shape: {b_input_ids.shape}')
    small_model_device = small_model.device
    b_token_loss_1, b_mean_seq_loss_1 = cal_loss_from_tokens(model_input, small_model, small_model_device, loss_fn, pad_token_id)
    big_model_device = big_model.device
    b_token_loss_2, b_mean_seq_loss_2 = cal_loss_from_tokens(model_input, big_model, big_model_device, loss_fn, pad_token_id)
    return b_input_ids, token_counts, b_token_loss_1, b_mean_seq_loss_1, b_token_loss_2, b_mean_seq_loss_2

def print_diff_loss(texts, small_model, big_model, tokenizer):
    max_seq_len = 100
    pad_token_id = tokenizer.pad_token_id
    b_input_ids, token_counts, b_token_loss_1, b_mean_seq_loss_1, b_token_loss_2, b_mean_seq_loss_2 = cal_loss_diff_from_texts(texts, max_seq_len, small_model, big_model, tokenizer, pad_token_id, loss_fn)
    print(b_input_ids.shape)
    print(b_token_loss_1)
    print(b_token_loss_2)
    print(b_mean_seq_loss_1)
    print(b_mean_seq_loss_2)
    for example_input_ids, example_loss_1, example_loss_2, example_mean_seq_loss_1, example_mean_seq_loss_2 in zip(b_input_ids, b_token_loss_1, b_token_loss_2, b_mean_seq_loss_1, b_mean_seq_loss_2):
        for token_id, token_loss_1, token_loss_2 in zip(example_input_ids, example_loss_1, example_loss_2):
            print(f'{tokenizer.decode(token_id)}: {token_loss_1}\t{token_loss_2}\t{token_loss_1 - token_loss_2}')
        print(f'mean seq loss: {example_mean_seq_loss_1}\t{example_mean_seq_loss_2}\t{example_mean_seq_loss_1 - example_mean_seq_loss_2}')
        print('\n===\n')

def process_examples(examples, bsz = 1, max_seq_len = 8192, text_field = 'content_split', limit_example_cnt = None):
    if len(examples) == 0:
        return []
    bsz = min(bsz, len(examples))
    pad_token_id = small_model_tokenizer.pad_token_id
    batch_cnt = len(examples) // bsz + (1 if len(examples) % bsz else 0)
    prefix = ''
    model_output_field = 'sf_model_output'
    for i in tqdm(range(0, len(examples), bsz), total = batch_cnt):
        batch_examples = examples[i:i+bsz]
        raw_idxs = []
        raw_idxs_is_valid = []
        raw_idxs_to_batch_idxs = []
        real_batch_texts = []
        for raw_idx, example in enumerate(batch_examples):
            text = example.get(text_field, None)
            raw_idxs.append(raw_idx)
            if text is not None and text != '':
                real_batch_texts.append(text)
                raw_idxs_is_valid.append(True)
                raw_idxs_to_batch_idxs.append(len(real_batch_texts) - 1)
            else:
                raw_idxs_is_valid.append(False)
                raw_idxs_to_batch_idxs.append(-1)
        b_input_ids, token_counts, b_token_loss_1, b_mean_seq_loss_1, b_token_loss_2, b_mean_seq_loss_2 = cal_loss_diff_from_texts(real_batch_texts, max_seq_len, small_model, big_model, small_model_tokenizer, pad_token_id, loss_fn)
        for raw_idx, raw_idx_is_valid, raw_idx_to_batch_idx in zip(raw_idxs, raw_idxs_is_valid, raw_idxs_to_batch_idxs):
            example = batch_examples[raw_idx]
            ret = {}
            if raw_idx_is_valid:
                input_ids, token_cnt, token_loss_1, mean_seq_loss_1, token_loss_2, mean_seq_loss_2 = b_input_ids[raw_idx_to_batch_idx], token_counts[raw_idx_to_batch_idx], b_token_loss_1[raw_idx_to_batch_idx], b_mean_seq_loss_1[raw_idx_to_batch_idx], b_token_loss_2[raw_idx_to_batch_idx], b_mean_seq_loss_2[raw_idx_to_batch_idx]
                # ret[f'{prefix}input_ids'] = input_ids.tolist()
                # ret[f'{prefix}token_loss_1'] = token_loss_1
                ret[f'{prefix}loss_raw'] = mean_seq_loss_1
                # ret[f'{prefix}token_loss_2'] = token_loss_2
                ret[f'{prefix}loss_new'] = mean_seq_loss_2
                ret[f'{prefix}loss_delta_abs'] = mean_seq_loss_1 - mean_seq_loss_2
                if mean_seq_loss_2 == 0:
                    ret[f'{prefix}loss_delta_rel'] = -100
                else:
                    ret[f'{prefix}loss_delta_rel'] = mean_seq_loss_1 / mean_seq_loss_2 # small model loss / large model loss
                if mean_seq_loss_1 == 0:
                    ret[f'{prefix}loss_delta_rel_my'] = -100
                else:
                    ret[f'{prefix}loss_delta_rel_my'] = (mean_seq_loss_1 - mean_seq_loss_2) / mean_seq_loss_1 # 等价
                ret[f'{prefix}is_valid'] = 1
                ret[f'{prefix}token_cnt'] = input_ids.shape[0]
                # ret[f'{prefix}token_cnt'] = token_cnt
            else:
                # ret[f'{prefix}input_ids'] = []
                ret[f'{prefix}loss_raw'] = -100
                ret[f'{prefix}loss_new'] = -100
                ret[f'{prefix}loss_delta_abs'] = -100
                ret[f'{prefix}loss_delta_rel'] = -100
                ret[f'{prefix}loss_delta_rel_my'] = -100
                ret[f'{prefix}is_valid'] = 0
                ret[f'{prefix}token_cnt'] = 0
            example[model_output_field] = ret

        total_cnt = i + len(batch_examples)
        if limit_example_cnt is not None and total_cnt >= limit_example_cnt:
            examples = examples[:limit_example_cnt]
            break

    return examples

def process_file(input_path, output_path, bsz = 1, max_seq_len = 8192, text_field = 'content_split', limit_example_cnt = None):
    examples = load_data_from_file(input_path)
    process_examples(examples, bsz = bsz, max_seq_len = max_seq_len, text_field = text_field, limit_example_cnt = limit_example_cnt)
    save_data_to_json(examples, output_path, pretty=True)

if __name__ == '__main__':
    texts = ['Hello, how are you?',
             '自然语言认知和理解是让电脑把输入的语言变成结构化符号与语义关系，然后根据目的再处理。自然语言生成系统则是把计算机数据转化为自然语言。\n自然语言处理要研制表示语言能力和语言应用的模型, 建立计算框架来实现并完善语言模型，并根据语言模型设计各种实用系统及探讨这些系统的评测技术。']
    print_diff_loss(texts, small_model, big_model, small_model_tokenizer)

    # max_seq_len = 4096
    # bsz=512 80GB OOM
    # bsz = 256

    max_seq_len = 8192
    # bsz = 2
    bsz = 1 # 比bsz快一点，虽然GPU mem利用率低，但因为没有pad了，还是会更快
    limit_example_cnt = 1000
    process_file(input_path = '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.gz.parquet',
    output_path ='7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json',
    bsz = bsz, max_seq_len = max_seq_len, text_field = 'content_split', limit_example_cnt = limit_example_cnt)