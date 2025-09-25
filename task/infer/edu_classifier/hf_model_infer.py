from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === global vars ===
score_field = 'edu_score'
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model.cuda()
# ===================

def test_model_infer():
    text = "This is a test sentence." # 0.07964444160461426
    # text = "This is a test sentence." * 100000 # 0.8143334984779358
    # 发现这个分类器可以被repeat hack，重复后，0.07964444160461426->0.8143334984779358，虽然attention influence也会受此影响，但没这么明显
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    print(len(inputs['input_ids'][0])) # 会自动截断，最大截断到512个token，应该是从左往右截断
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()
    result = {
        # "text": text,
        "score": score,
        "int_score": int(round(max(0, min(score, 5)))),
    }
    # print(result)
    print(result['score'])
    print(result['int_score'])
    # {'text': 'This is a test sentence.', 'score': 0.07964489609003067, 'int_score': 0}

def data_to_gpu(input_dict):
    for k, v in input_dict.items():
        input_dict[k] = v.cuda()

def test_model_batch_infer():
    text = "This is a test sentence."
    texts = []
    for _ in range(2):
        texts.append(text)
    inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
    print(len(inputs['input_ids'][0]))
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    scores = logits.tolist()
    print(scores)

def test_model_batch_infer_gpu_util():
    text = "This is a test sentence." * 100
    texts = []
    batch_size = 128
    for _ in range(batch_size):
        texts.append(text)
    inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
    # print(inputs)
    model.cuda()
    data_to_gpu(inputs)
    total = 10**9
    with tqdm(total=total, desc='processing') as pbar:
        while True:
            outputs = model(**inputs) # 默认是单卡infer
            pbar.update(batch_size)

def hf_infer_examples(examples):
    global tokenizer, model
    texts = []
    for example in examples:
        texts.append(example['content_split'])
    inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
    data_to_gpu(inputs)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()
    scores = logits.tolist()
    assert len(scores) == len(examples)
    for example, score in zip(examples, scores):
        example[score_field] = score
    return examples

if __name__ == "__main__":
    # test_model_infer()
    # test_model_batch_infer()
    test_model_batch_infer_gpu_util()