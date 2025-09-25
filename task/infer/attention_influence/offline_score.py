from shared_util import *
from answer_parsing import *

def score_example_key_passage_retrieval_plain(example):
    score = 0
    gen_len = len(example['model_output'])
    if example['model_output'] == example['answer'][:gen_len]:
        score = 1
    return score

def score_example_needle_retrieval_recall(example):
    score = 0
    model_output = example['model_output']
    hit_cnt = 0
    for word in model_output.split():
        if word in example['answer']:
            hit_cnt += 1
    total_cnt = len(example['answer'].split())
    score = hit_cnt / total_cnt
    return score

def score_example_gsm8K(example):
    model_output_dict = parse_cot_output(example['model_output'])
    score = 0
    try:
        score = gsm8k_metric(example['question'], model_output_dict['answer'], example['answer'])
    except Exception as e:
        print(traceback.format_exc())
        print('parsing error')
    return score

def score_file(src_path, score_fn, limit_cnt = None, tgt_path = None):
    print(f'src_path: {src_path}')
    examples = load_data_from_json(src_path)
    scores = []
    for idx, example in tqdm(enumerate(examples)):
        score = score_fn(example)
        scores.append(score)
        if limit_cnt is not None and (idx+1) >= limit_cnt:
            break
    mean_score = mean(scores)
    print(f'mean_score: {mean_score}')

def cal_heads_overlap(src_path_1, src_path_2):
    top_heads_1 = score_file_head_retrieval_score(src_path_1, return_top_heads = True)
    top_heads_2 = score_file_head_retrieval_score(src_path_2, return_top_heads = True)
    total_cnt = len(top_heads_1)
    hit_cnt = 0
    for head in top_heads_1:
        if head in top_heads_2:
            hit_cnt += 1
    print(f'hit_cnt: {hit_cnt}, total_cnt: {total_cnt}, hit_ratio: {hit_cnt/total_cnt}')

if __name__ == '__main__':
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_output.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_output.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_text_level_20241023.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file_head_retrieval_score(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_text_level_20241023.json', tgt_path=None)
    score_file_head_retrieval_score(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023.json', tgt_path=None) # 和text level相比，数值有0.0x的diff，但趋势完全一样，之后用token level即可
    score_file_head_retrieval_score(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json', tgt_path=None)
    score_file_head_retrieval_score(src_path='tmp/1.3B_llama2_like_1499B_key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json', tgt_path=None)
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023.json', score_fn=score_example_key_passage_retrieval_plain, limit_cnt = 400, tgt_path=None)
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_mask_top_retrieval_heads_5%.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file(src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K_mask_non_top_retrieval_heads_random_5%.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file(src_path='tmp/GSM8K_official_processed_20241024_test_P4_1.3B_1499B_output.json', score_fn=score_example_gsm8K, tgt_path=None)
    score_file(src_path='tmp/GSM8K_official_processed_20241024_test_P4_1.3B_1499B_mask_top_retrieval_heads_5%_output.json', score_fn=score_example_gsm8K, tgt_path=None)
    score_file(src_path='tmp/GSM8K_official_processed_20241024_test_P4_1.3B_1499B_mask_non_top_retrieval_heads_random_5%_output.json', score_fn=score_example_gsm8K, tgt_path=None)

    score_file(src_path='tmp/reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file(src_path='tmp/reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', score_fn=score_example_needle_retrieval_recall, tgt_path=None)
    score_file_head_retrieval_score(src_path='tmp/reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', tgt_path=None)

    cal_heads_overlap('tmp/1.3B_llama2_like_1499B_key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json', 'tmp/reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json')

    score_file(src_path='tmp/reasoning_needles_eval_4K_20241224_attn_output_soft_retrieval_scores_token_level_20241224_two_pass.json', score_fn=score_example_needle_retrieval_recall, tgt_path=None)
    score_file_head_retrieval_score(src_path='tmp/reasoning_needles_eval_4K_20241224_attn_output_soft_retrieval_scores_token_level_20241224_two_pass.json', tgt_path=None)
    cal_heads_overlap('tmp/1.3B_llama2_like_1499B_key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json', 'tmp/reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json')

    score_file(src_path='tmp/1.3B_llama2_like_1499B_plain_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file_head_retrieval_score(src_path='tmp/1.3B_llama2_like_1499B_plain_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', tgt_path=None)

    score_file(src_path='tmp/1.3B_llama2_like_1499B_reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', score_fn=score_example_key_passage_retrieval_plain, tgt_path=None)
    score_file_head_retrieval_score(src_path='tmp/1.3B_llama2_like_1499B_reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', tgt_path=None)

    cal_heads_overlap('tmp/1.3B_llama2_like_1499B_key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json', 
                      'tmp/1.3B_llama2_like_1499B_plain_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json')
    cal_heads_overlap('tmp/1.3B_llama2_like_1499B_key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json', 
                      'tmp/1.3B_llama2_like_1499B_reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json')
    cal_heads_overlap('tmp/1.3B_llama2_like_1499B_plain_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json', 
                      'tmp/1.3B_llama2_like_1499B_reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json')