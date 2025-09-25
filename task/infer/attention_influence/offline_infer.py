from x_shared_util import *
from x.utils import print_rank_0, get_tokenizer, custom_input
from x import Runner
from x.learner.learner import hf_model_forward

def preprocess_example_key_passage_retrieval(example):
    if 'model_input' not in example:
        example['model_input'] = example['query']

def infer_example(example, max_new_tokens = 16, verbose = False, **kwargs):
    global runner
    preprocess_example_key_passage_retrieval(example)
    model_input = example['model_input']
    model_output = runner.run_generate(prompt=model_input, do_sample=False, max_new_tokens=max_new_tokens)
    example['model_output'] = model_output
    if verbose:
        print_rank_0(f'# model_input:\n{model_input}')
        print_rank_0(f'# model_output:\n{model_output}')

def preprocess_example_GSM8K(example):
    model_input_1 = gen_model_input_1_fewshot_GSM8K(example['question'])
    example['model_input'] = model_input_1

def infer_example_GSM8K(example, max_new_tokens = 256, verbose = False, **kwargs):
    global runner
    preprocess_example_GSM8K(example)
    model_input = example['model_input']
    model_output = runner.run_generate(prompt=model_input, do_sample=False, max_new_tokens=max_new_tokens)
    example['model_output'] = model_output

def preprocess_example_pretrain_data(example):
    if 'content_split' in example:
        example['model_input'] = example['content_split']
    elif 'prompt_with_cot_and_answer' in example:
        example['model_input'] = example['prompt_with_cot_and_answer']

def preprocess_example_pretrain_data_v2(example):
    if 'prompt_with_cot_and_answer' in example:
        example['content_split'] = example['prompt_with_cot_and_answer']

def cal_retrieval_score_plain_text(top_attended_text, answer):
    answer_char_recall_cnt = 0
    top_attended_text_char_list = [ch for ch in top_attended_text]
    for ch in answer:
        if ch in top_attended_text_char_list:
            answer_char_recall_cnt += 1
            the_index = top_attended_text_char_list.index(ch)
            top_attended_text_char_list.pop(the_index)
    ret = answer_char_recall_cnt / len(answer)
    return ret

def cal_retrieval_score_plain_token(top_attended_token_ids, answer_token_ids):
    answer_token_recall_cnt = 0
    for token_id in answer_token_ids:
        if token_id in top_attended_token_ids:
            answer_token_recall_cnt += 1
            the_index = top_attended_token_ids.index(token_id)
            top_attended_token_ids.pop(the_index)
    ret = answer_token_recall_cnt / len(answer_token_ids)
    return ret

def cal_retrieval_scores(attention_scores, tokenizer, generated_text, input_token_cnt, answer, verbose = False):
    # attention_scores: (n_layer, gen_seq_len, batch_size, n_head, total_seq_len, total_seq_len)
    answer_token_ids = tokenizer.encode(answer)
    all_token_ids = tokenizer.encode(generated_text)
    # print(len(attention_scores[0]))
    # print(f'attention_scores shape: {attention_scores[0][-1].shape}')
    batch_size = attention_scores[-1][0].shape[0]
    n_layer = len(attention_scores)
    n_head = attention_scores[-1][0].shape[1]
    retrieval_scores = {}
    for layer_index in range(n_layer):
        if verbose:
            print(f'layer {layer_index}')
        the_attention_scores = attention_scores[layer_index][-1][0]
        retrieval_scores[layer_index] = []
        for i in range(n_head):
            single_example_single_layer_single_head_attn_probs = the_attention_scores[i] # (attn_seq_len, attn_seq_len)
            # 只关注生成的token的attn分布
            single_example_single_layer_single_head_attn_probs = single_example_single_layer_single_head_attn_probs[-(len(all_token_ids) - input_token_cnt):, :]
            max_token_ids_indices = torch.argmax(single_example_single_layer_single_head_attn_probs, dim = -1)
            max_token_ids = []
            for token_id_index in max_token_ids_indices:
                if token_id_index <= len(all_token_ids) - 1:
                    max_token_ids.append(all_token_ids[token_id_index])
                else: # 生成的token的某个token的attention score最高
                    max_token_ids.append(1)
            tokens = tokenizer.convert_ids_to_tokens(max_token_ids)
            top_attended_text = tokenizer.convert_tokens_to_string(tokens)
            if verbose:
                print(f'head {i}')
                print('|'.join(tokens))
                print(top_attended_text)
            # retrieval_score = cal_retrieval_score_plain_text(top_attended_text, answer)
            retrieval_score = cal_retrieval_score_plain_token(max_token_ids, answer_token_ids)
            retrieval_scores[layer_index].append(retrieval_score) # 存为json后会把int的key自动转为str
    if verbose:
        for layer_index in range(n_layer):
            print_str = '\t'.join(retrieval_scores[layer_index])
            print(print_str)
    return retrieval_scores

def cal_soft_retrieval_scores(attention_scores, tokenizer, generated_text, input_token_cnt, answer, verbose = False):
    # attention_scores: (n_layer, gen_seq_len, batch_size, n_head, total_seq_len, total_seq_len)
    answer_token_ids = tokenizer.encode(answer)
    all_token_ids = tokenizer.encode(generated_text)
    # print(len(attention_scores[0]))
    # print(f'attention_scores shape: {attention_scores[0][-1].shape}')
    batch_size = attention_scores[-1][0].shape[0]
    n_layer = len(attention_scores)
    n_head = attention_scores[-1][0].shape[1]
    retrieval_scores = {}
    for layer_index in range(n_layer):
        if verbose:
            print(f'layer {layer_index}')
        the_attention_scores = attention_scores[layer_index][-1][0]
        retrieval_scores[layer_index] = []
        for i in range(n_head):
            single_example_single_layer_single_head_attn_probs = the_attention_scores[i] # (attn_seq_len, attn_seq_len)
            # 只关注生成的token的attn分布
            single_example_single_layer_single_head_attn_probs = single_example_single_layer_single_head_attn_probs[-(len(all_token_ids) - input_token_cnt):, :]
            if verbose:
                print(f'single_example_single_layer_single_head_attn_probs: {single_example_single_layer_single_head_attn_probs.shape}')
            max_token_ids_indices = torch.argmax(single_example_single_layer_single_head_attn_probs, dim = -1)
            max_token_ids = []
            prob_score_sum = 0
            for query_token_index, key_token_id_index in enumerate(max_token_ids_indices):
                if verbose:
                    print(f'query_token_index: {query_token_index}')
                    print(f'key_token_id_index: {key_token_id_index}')
                if key_token_id_index <= len(all_token_ids) - 1:
                    max_token_id = all_token_ids[key_token_id_index]
                    max_token_ids.append(max_token_id)
                    if max_token_id in answer_token_ids:
                        prob_score_sum += single_example_single_layer_single_head_attn_probs[query_token_index][key_token_id_index].item()
                else:
                    max_token_ids.append(1)
            tokens = tokenizer.convert_ids_to_tokens(max_token_ids)
            top_attended_text = tokenizer.convert_tokens_to_string(tokens)
            if verbose:
                print(f'head {i}')
                print('|'.join(tokens))
                print(top_attended_text)
            retrieval_score = prob_score_sum / len(answer_token_ids)
            retrieval_scores[layer_index].append(retrieval_score) # 存为json后会把int的key自动转为str
    if verbose:
        for layer_index in range(n_layer):
            print_str = '\t'.join(retrieval_scores[layer_index])
            print(print_str)
    return retrieval_scores

# 这种方式没有考虑到答案的word出现多次且在错误的原文的span中出现的情况
def cal_topk_soft_retrieval_scores(attention_scores, tokenizer, generated_text, input_token_cnt, answer, verbose = False):
    # attention_scores: (n_layer, gen_seq_len, batch_size, n_head, total_seq_len, total_seq_len)
    answer_token_ids = tokenizer.encode(answer)
    topk_val = len(answer_token_ids)
    all_token_ids = tokenizer.encode(generated_text)
    # print(len(attention_scores[0]))
    # print(f'attention_scores shape: {attention_scores[0][-1].shape}')
    batch_size = attention_scores[-1][0].shape[0]
    n_layer = len(attention_scores)
    n_head = attention_scores[-1][0].shape[1]
    retrieval_scores = {}
    for layer_index in range(n_layer):
        if verbose:
            print(f'layer {layer_index}')
        the_attention_scores = attention_scores[layer_index][-1][0]
        retrieval_scores[layer_index] = []
        for i in range(n_head):
            single_example_single_layer_single_head_attn_probs = the_attention_scores[i] # (attn_seq_len, attn_seq_len)
            # 只关注生成的token的attn分布
            single_example_single_layer_single_head_attn_probs = single_example_single_layer_single_head_attn_probs[-(len(all_token_ids) - input_token_cnt):, :]
            if verbose:
                print(f'single_example_single_layer_single_head_attn_probs: {single_example_single_layer_single_head_attn_probs.shape}')
            topk_token_ids_probs, topk_token_ids_indices = torch.topk(single_example_single_layer_single_head_attn_probs, k=topk_val, dim = -1)
            q_len = topk_token_ids_probs.shape[0]
            the_token_ids = []
            prob_score_sum = 0
            for query_token_index in range(q_len):
                key_topk_token_indices = topk_token_ids_indices[query_token_index]
                key_topk_token_probs = topk_token_ids_probs[query_token_index]
                for prob, key_token_id_index in zip(key_topk_token_probs, key_topk_token_indices):
                    if key_token_id_index <= len(all_token_ids) - 1:
                        the_token_id = all_token_ids[key_token_id_index]
                        the_token_ids.append(the_token_id)
                        if the_token_id in answer_token_ids:
                            prob_score_sum += prob.item()
                    else:
                        the_token_ids.append(1)
            tokens = tokenizer.convert_ids_to_tokens(the_token_ids)
            top_attended_text = tokenizer.convert_tokens_to_string(tokens)
            if verbose:
                print(f'head {i}')
                print('|'.join(tokens))
                print(top_attended_text)
            retrieval_score = prob_score_sum / topk_val / len(answer_token_ids)
            retrieval_scores[layer_index].append(retrieval_score) # 存为json后会把int的key自动转为str
    if verbose:
        for layer_index in range(n_layer):
            print_str = '\t'.join(retrieval_scores[layer_index])
            print(print_str)
    return retrieval_scores

def find_needle_idx(real_needle_token_ids, input_token_ids):
    span_len = len(real_needle_token_ids)
    max_overlap = 0
    start = end = 0
    for i in range(len(input_token_ids)):            
        token_span = input_token_ids[i: i + span_len]
        span_ids = set(token_span)
        overlap = float(len(span_ids.intersection(set(real_needle_token_ids)))) / len(set(real_needle_token_ids))
        if overlap >= max_overlap:
            max_overlap = overlap
            start = i
            end = i + span_len
            if max_overlap == 1:
                break
    return start, end

def cal_needle_span_topk_soft_retrieval_scores(attention_scores, tokenizer, generated_text, input_token_cnt, answer, example, verbose = False):
    # attention_scores: (n_layer, gen_seq_len, batch_size, n_head, total_seq_len, total_seq_len)
    answer_token_ids = tokenizer.encode(answer)
    all_token_ids = tokenizer.encode(generated_text)
    real_needle = example['real_needle']
    real_needle_token_ids = tokenizer.encode(real_needle)
    input_token_ids = all_token_ids[:input_token_cnt]
    # 有可能因为切词，needle token ids span不全出现在input_token_ids中
    needle_start_idx, needle_end_idx = find_needle_idx(real_needle_token_ids, input_token_ids)
    topk_val = needle_end_idx - needle_start_idx
    if topk_val == 0:
        return 0
    # print(len(attention_scores[0]))
    # print(f'attention_scores shape: {attention_scores[0][-1].shape}')
    batch_size = attention_scores[-1][0].shape[0]
    n_layer = len(attention_scores)
    n_head = attention_scores[-1][0].shape[1]
    retrieval_scores = {}
    for layer_index in range(n_layer):
        if verbose:
            print(f'layer {layer_index}')
        the_attention_scores = attention_scores[layer_index][-1][0]
        retrieval_scores[layer_index] = []
        for i in range(n_head):
            single_example_single_layer_single_head_attn_probs = the_attention_scores[i] # (attn_seq_len, attn_seq_len)
            # 只关注生成的token的attn分布
            single_example_single_layer_single_head_attn_probs = single_example_single_layer_single_head_attn_probs[-(len(all_token_ids) - input_token_cnt):, :]
            if verbose:
                print(f'single_example_single_layer_single_head_attn_probs: {single_example_single_layer_single_head_attn_probs.shape}')
            topk_token_ids_probs, topk_token_ids_indices = torch.topk(single_example_single_layer_single_head_attn_probs, k=topk_val, dim = -1)
            q_len = topk_token_ids_probs.shape[0]
            the_token_ids = []
            prob_score_sum = 0
            for query_token_index in range(q_len):
                key_topk_token_indices = topk_token_ids_indices[query_token_index]
                key_topk_token_probs = topk_token_ids_probs[query_token_index]
                for prob, key_token_id_index in zip(key_topk_token_probs, key_topk_token_indices):
                    if needle_start_idx <= key_token_id_index and key_token_id_index < needle_end_idx:
                        the_token_id = all_token_ids[key_token_id_index]
                        the_token_ids.append(the_token_id)
                        prob_score_sum += prob.item()
                    else:
                        the_token_ids.append(1)
            tokens = tokenizer.convert_ids_to_tokens(the_token_ids)
            top_attended_text = tokenizer.convert_tokens_to_string(tokens)
            if verbose:
                print(f'head {i}')
                print('|'.join(tokens))
                print(top_attended_text)
            retrieval_score = prob_score_sum / topk_val / len(answer_token_ids)
            retrieval_scores[layer_index].append(retrieval_score) # 存为json后会把int的key自动转为str
    if verbose:
        for layer_index in range(n_layer):
            print_str = '\t'.join(retrieval_scores[layer_index])
            print(print_str)
    return retrieval_scores

def infer_example_all_head_top_attended_tokens(example, answer = None, max_new_tokens = 16, verbose = False, **kwargs):
    global runner
    tokenizer = runner.tokenizer
    preprocess_example_key_passage_retrieval(example)
    model_input = example['model_input']
    model_input_ids = tokenizer(model_input)['input_ids']
    input_token_cnt = len(model_input_ids)
    generated_text, output = runner.run_generate(model_input, do_sample=False, max_new_tokens = max_new_tokens, output_attentions=True, return_dict_in_generate=True, return_output_only = False)
    attention_scores = output.attentions
    if verbose:
        print('generated_text:')
        print(generated_text) # input + output
    answer = example['answer']
    example['model_output'] = generated_text[len(model_input):]
    retrieval_scores = cal_retrieval_scores(attention_scores, tokenizer, generated_text, input_token_cnt, answer, verbose)
    example['retrieval_scores'] = retrieval_scores
    return retrieval_scores

def infer_example_all_head_top_attended_tokens_two_pass(example, answer = None, max_new_tokens = 16, verbose = False, **kwargs):
    global runner
    tokenizer = runner.tokenizer
    preprocess_example_key_passage_retrieval(example)
    model_input = example['model_input']
    model_input_ids = tokenizer(model_input)['input_ids']
    input_token_cnt = len(model_input_ids)
    if verbose:
        print(f'input_token_cnt: {input_token_cnt}')
    with torch.inference_mode():
        full_generated_text = runner.run_generate(model_input, do_sample=False, max_new_tokens = max_new_tokens, output_attentions=False, return_dict_in_generate=False, return_output_only = False)
    if verbose:
        print('full_generated_text:')
        print(full_generated_text)
    full_model_input_dict = tokenizer(full_generated_text)
    input_ids = torch.tensor([full_model_input_dict['input_ids'][:-1]]).cuda().contiguous()
    attention_mask = torch.tensor([full_model_input_dict['attention_mask'][:-1]]).cuda()
    max_len = input_ids.shape[1]
    position_ids = torch.arange(max_len, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    batch_input_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
    with torch.inference_mode():
        if runner.model_type == 'hf':
            output = runner.model.forward(**batch_input_dict, output_attentions=True)
        else:
            output = runner.model.forward(batch_input_dict, output_attentions=True)
    attention_scores = output['attentions']
    n_layer = len(attention_scores)
    new_attention_scores = []
    for i in range(n_layer):
        new_attention_scores.append(attention_scores[i].unsqueeze(0)) # pad, 和generation的维度对齐
    attention_scores = tuple(new_attention_scores)
    # print(attention_scores[0][0].shape)
    answer = example['answer']
    example['model_output'] = full_generated_text[len(model_input):]
    retrieval_scores = cal_retrieval_scores(attention_scores, tokenizer, full_generated_text, input_token_cnt, answer, verbose) # plain needle
    # retrieval_scores = cal_soft_retrieval_scores(attention_scores, tokenizer, full_generated_text, input_token_cnt, answer, verbose) # plain needle改进的尝试
    # retrieval_scores = cal_topk_soft_retrieval_scores(attention_scores, tokenizer, full_generated_text, input_token_cnt, answer, verbose) # plain needle改进的尝试
    # retrieval_scores = cal_needle_span_topk_soft_retrieval_scores(attention_scores, tokenizer, full_generated_text, input_token_cnt, answer, example, verbose) # # plain needle改进的尝试，即reasoning needle
    example['retrieval_scores'] = retrieval_scores
    return retrieval_scores

def get_mask_head_ids_from_score_file(score_path, head_topn_percent = 0.05, skip_layers_str = '', **kwargs):
    mean_retrieval_scores_2d_list = score_file_head_retrieval_score(src_path = score_path, return_2d_list = True, **kwargs)
    all_skip_layers_2d = []
    if skip_layers_str:
        all_skip_layers_2d = [eval(skip_layers) for skip_layers in skip_layers_str.split('_')]
    all_skip_layers = []
    for skip_layers in all_skip_layers_2d:
        for layer_idx in range(skip_layers[0], skip_layers[1] + 1):
            all_skip_layers.append(layer_idx)
    print(f'all_skip_layers: {all_skip_layers}')
    # all_scores = [score for scores in mean_retrieval_scores_2d_list for score in scores]
    all_scores = []
    for layer_idx, scores in enumerate(mean_retrieval_scores_2d_list):
        if layer_idx in all_skip_layers:
            continue
        for score in scores:
            all_scores.append(score)
    all_scores.sort(reverse=True)
    topn_percent_th = all_scores[int(len(all_scores) * head_topn_percent)]
    if head_topn_percent > 0:
        print(f'mask top {head_topn_percent} head')
        mask_head_ids = []
        for layer_idx, scores in enumerate(mean_retrieval_scores_2d_list):
            if layer_idx in all_skip_layers:
                continue
            for head_idx, score in enumerate(scores):
                if score >= topn_percent_th:
                    mask_head_ids.append((layer_idx, head_idx))
    elif head_topn_percent == 0:
        print('no mask')
    else:
        all_candidates = []
        print(f'mask non-top random {head_topn_percent} head')
        for layer_idx, scores in enumerate(mean_retrieval_scores_2d_list):
            if layer_idx in all_skip_layers:
                continue
            for head_idx, score in enumerate(scores):
                if score < topn_percent_th:
                    all_candidates.append((layer_idx, head_idx))
        random.shuffle(all_candidates)
        mask_head_ids = all_candidates[:int(len(all_scores) * (-1 * head_topn_percent))]
    print('load masked heads from file:')
    print(mask_head_ids)
    return mask_head_ids

def get_cached_masked_head_ids(fn, score_path, head_topn_percent, **kwargs):
    mask_head_ids = None
    if not hasattr(fn, 'mask_head_ids'):
        if score_path is not None:
            mask_head_ids = get_mask_head_ids_from_score_file(score_path, head_topn_percent, **kwargs)
            fn.mask_head_ids = mask_head_ids
            print(mask_head_ids)
        else:
            fn.mask_head_ids = None
    else:
        mask_head_ids = fn.mask_head_ids
    return mask_head_ids

def infer_example_mask_selected_heads_generate(example, preprocess_fn = preprocess_example_key_passage_retrieval, score_path = None, mask_head_ids = None, max_new_tokens = 16, verbose = False, head_topn_percent = 0.05, skip_layers_str = '', **kwargs):
    global runner
    preprocess_fn(example)
    model_input = example['model_input']
    mask_head_ids = get_cached_masked_head_ids(fn = infer_example_mask_selected_heads_generate, score_path = score_path, head_topn_percent = head_topn_percent, skip_layers_str = skip_layers_str)
    if runner.model_type == 'hf':
        runner.model.transformer.mask_head_ids = mask_head_ids
    else:
        runner.model.model.transformer.mask_head_ids = mask_head_ids
    model_output = runner.run_generate(prompt=model_input, mask_head_ids=mask_head_ids, max_new_tokens=max_new_tokens, return_output_only = True)
    example['model_output'] = model_output
    if verbose:
        print_rank_0(f'# model_input:\n{model_input}')
        print_rank_0(f'# model_output:\n{model_output}')

def infer_example_mask_selected_heads_delta_loss(example, preprocess_fn = preprocess_example_pretrain_data, score_path = None, mask_head_ids = None, max_new_tokens = 16, verbose = False, head_topn_percent = 0.05, **kwargs):
    global runner
    tokenizer = runner.tokenizer
    data_parser = runner.data_parser
    preprocess_fn(example)
    model_input = example['model_input']
    if not model_input:
        model_output = {
            'loss_raw': 100,
            'loss_new': 100,
            'loss_delta_abs': 0,
            'loss_delta_rel': 0,
            'token_cnt': 0
        }
        example['model_output'] = model_output
        return

    mask_head_ids = get_cached_masked_head_ids(fn = infer_example_mask_selected_heads_delta_loss, n_layer = runner.n_layer, n_head = runner.n_head, score_path = score_path, head_topn_percent = head_topn_percent)
    model_input_ids = tokenizer(model_input)['input_ids']    
    model_input_ids = model_input_ids[:data_parser.max_length]
    batch_input = data_parser.collate([model_input_ids])
    with torch.no_grad():
        if runner.model_type == 'hf':
            result_dict = hf_model_forward(runner.model, batch_input, pad_token_id=runner.tokenizer.pad_token_id, vocab_size=runner.tokenizer.raw_vocab_size, output_attentions = False, reduce_into_1d_list = False, mask_head_ids=None)
        else:
            result_dict = runner.model.forward(batch_input, output_attentions=False, mask_head_ids=None)
        loss_raw = result_dict['loss'].detach().item()
        if runner.model_type == 'hf':
            result_dict = hf_model_forward(runner.model, batch_input, pad_token_id=runner.tokenizer.pad_token_id, vocab_size=runner.tokenizer.raw_vocab_size, output_attentions = False, reduce_into_1d_list = False, mask_head_ids=mask_head_ids)
        else:
            result_dict = runner.model.forward(batch_input, output_attentions=False, mask_head_ids=None)
        loss_new = result_dict['loss'].detach().item()
    loss_delta_abs = loss_new - loss_raw
    loss_delta_rel = loss_delta_abs / loss_raw
    model_output = {
        'loss_raw': loss_raw,
        'loss_new': loss_new,
        'loss_delta_abs': loss_delta_abs,
        'loss_delta_rel': loss_delta_rel,
        'token_cnt': len(model_input_ids)
    }
    example['model_output'] = model_output
    if verbose:
        print_rank_0(f'# model_input:\n{model_input}')
        print_rank_0(f'# model_output:\n{model_output}')

def infer_example_mask_selected_heads_delta_loss_batch(examples, preprocess_fn = preprocess_example_pretrain_data, score_path = None, mask_head_ids = None, max_new_tokens = 16, verbose = False, head_topn_percent = 0.05, **kwargs):
    global runner
    tokenizer = runner.tokenizer
    data_parser = runner.data_parser
    token_ids_batch = []
    raw_token_cnt_list = []
    for example in examples:
        preprocess_fn(example)
        model_input = example['model_input']
        token_ids = tokenizer(model_input)['input_ids']
        raw_token_cnt = len(token_ids)
        raw_token_cnt_list.append(raw_token_cnt)
        token_ids = token_ids[:data_parser.max_length]
        if len(token_ids) < data_parser.max_length:
            token_ids = token_ids + [tokenizer.pad_token_id] * (data_parser.max_length - raw_token_cnt)
        token_ids_batch.append(token_ids)
    mask_head_ids = get_cached_masked_head_ids(fn = infer_example_mask_selected_heads_delta_loss, score_path = score_path, head_topn_percent = head_topn_percent)
    batch_input = data_parser.collate(token_ids_batch)
    loss_raw_list = []
    loss_new_list = []
    # with torch.no_grad(): # 仍保留中间的激活值
    with torch.inference_mode():
        result_dict = runner.model.forward(batch_input, output_attentions=False, mask_head_ids=None, reduce_into_1d_list = True)
        loss_raw_list = result_dict['loss'].detach().tolist()
        del result_dict
        result_dict = runner.model.forward(batch_input, output_attentions=False, mask_head_ids=mask_head_ids, reduce_into_1d_list = True)
        loss_new_list = result_dict['loss'].detach().tolist()
        del result_dict
    for raw_token_cnt, loss_raw, loss_new, example in zip(raw_token_cnt_list, loss_raw_list, loss_new_list, examples):
        loss_delta_abs = loss_new - loss_raw
        loss_delta_rel = loss_delta_abs / loss_raw
        model_output = {
            'loss_raw': loss_raw,
            'loss_new': loss_new,
            'loss_delta_abs': loss_delta_abs,
            'loss_delta_rel': loss_delta_rel,
            'token_cnt': raw_token_cnt
        }
        example['model_output'] = model_output
    if verbose:
        print_rank_0(f"# model_input:\n{examples[0]['model_input']}")
        print_rank_0(f"# model_output:\n{examples[0]['model_output']}")
    return examples

infer_example_mask_selected_heads_key_passage_retrieval = partial(infer_example_mask_selected_heads_generate, preprocess_fn = preprocess_example_key_passage_retrieval)
infer_example_mask_selected_heads_GSM8K = partial(infer_example_mask_selected_heads_generate, preprocess_fn = preprocess_example_GSM8K)

def infer_example_mask_selected_heads_delta_loss_gpt_perf(example, preprocess_fn = preprocess_example_pretrain_data_v2, score_path = None, mask_head_ids = None, max_new_tokens = 16, verbose = False, head_topn_percent = 0.05, tag_output_source = True, **kwargs):
    global runner
    max_prompt_length = runner.data_parser.max_length
    tokenizer = runner.tokenizer
    preprocess_fn(example)
    with torch.no_grad():
        if example['content_split'] == '':
            print('======== WARNING NULL DATA ========')
            print(example)
            print('===================================')
            loss_raw = 100
            loss_new = 100
            token_cnt = 0
        else:
            loss_raw, token_cnt = infer_single_example_gpt_perf(example, runner.inferencer, tokenizer, max_prompt_length = max_prompt_length, return_loss = True)
            loss_new, _ = infer_single_example_gpt_perf(example, runner.inferencer2, tokenizer, max_prompt_length = max_prompt_length, return_loss = True)
    loss_delta_abs = loss_new - loss_raw
    loss_delta_rel = loss_delta_abs / loss_raw

    if tag_output_source and 'meta' in example and 'source' in example['meta']['source']:
        example['meta']['source'] = example['meta']['source'] + '<sep_token>' + 'gpt_perf_infer_single_example'
    model_output = {
        'loss_raw': loss_raw,
        'loss_new': loss_new,
        'loss_delta_abs': loss_delta_abs,
        'loss_delta_rel': loss_delta_rel,
        'token_cnt': token_cnt
    }
    example['model_output'] = model_output
    if verbose:
        if 'content_split' in example:
            print_rank_0(f"# model_input:\n{example['content_split']}")
        print_rank_0(f'# model_output:\n{model_output}')

def infer_example_mask_selected_heads_delta_loss_gpt_perf_two_stage(example, preprocess_fn = preprocess_example_pretrain_data_v2, score_path = None, mask_head_ids = None, max_new_tokens = 16, verbose = False, head_topn_percent = 0.05, tag_output_source = True, **kwargs):
    global runner
    max_prompt_length = runner.data_parser.max_length
    tokenizer = runner.tokenizer
    preprocess_fn(example)
    stage = runner.args.stage
    with torch.no_grad():
        if example['content_split'] == '':
            print('======== WARNING NULL DATA ========')
            print(example)
            print('===================================')
            loss_raw = 100
            loss_new = 100
            token_cnt = 0
        else:
            if stage == '1':
                loss_raw, token_cnt = infer_single_example_gpt_perf(example, runner.inferencer, tokenizer, max_prompt_length = max_prompt_length, return_loss = True)
            elif stage == '2':
                loss_new, _ = infer_single_example_gpt_perf(example, runner.inferencer2, tokenizer, max_prompt_length = max_prompt_length, return_loss = True)

    if stage == '1':
        model_output = {
            'loss_raw': loss_raw,
            'loss_new': -1,
            'loss_delta_abs': -1,
            'loss_delta_rel': -1,
            'token_cnt': token_cnt
        }
        example['model_output_stage1'] = model_output
    elif stage == '2':
        if tag_output_source and 'meta' in example and 'source' in example['meta']['source']:
            example['meta']['source'] = example['meta']['source'] + '<sep_token>' + 'gpt_perf_infer_single_example'
        model_output = example['model_output_stage1'] # 引用会修改
        loss_raw = model_output['loss_raw']
        model_output['loss_new'] = loss_new
        try:
            loss_delta_abs = loss_new - loss_raw
            loss_delta_rel = loss_delta_abs / loss_raw
        except:
            loss_delta_abs = -100
            loss_delta_rel = -100
            print('error data:')
            print(f'loss_raw: {loss_raw}')
            print(f'loss_new: {loss_new}')
        model_output['loss_delta_abs'] = loss_delta_abs
        model_output['loss_delta_rel'] = loss_delta_rel
        example['model_output'] = model_output

    if verbose:
        if 'content_split' in example:
            print_rank_0(f"# model_input:\n{example['content_split']}")
        print_rank_0(f'# model_output:\n{model_output}')

from task.cal_attention_distance import infer_single_example_torch as infer_example_attention_distance
def infer_example_attention_distance_wrap(example, **kwargs):
    global runner
    tokenizer = runner.tokenizer
    data_loader = runner.data_parser
    model = runner.model
    model_type = runner.model_type
    return infer_example_attention_distance(model, tokenizer, data_loader, example, model_type = model_type, return_entropy = True, verbose = False, runner = runner, **kwargs)

infer_fn_dict = {
    'infer_example': infer_example,
    'infer_example_all_head_top_attended_tokens': infer_example_all_head_top_attended_tokens,
    'infer_example_all_head_top_attended_tokens_two_pass': infer_example_all_head_top_attended_tokens_two_pass,
    'infer_example_mask_selected_heads_generate': infer_example_mask_selected_heads_key_passage_retrieval,
    'infer_example_mask_selected_heads': infer_example_mask_selected_heads_key_passage_retrieval,
    'infer_example_mask_selected_heads_delta_loss': infer_example_mask_selected_heads_delta_loss,
    'infer_example_mask_selected_heads_delta_loss_batch': infer_example_mask_selected_heads_delta_loss_batch,
    'infer_example_mask_selected_heads_delta_loss_gpt_perf': infer_example_mask_selected_heads_delta_loss_gpt_perf,
    'infer_example_mask_selected_heads_delta_loss_gpt_perf_two_stage': infer_example_mask_selected_heads_delta_loss_gpt_perf_two_stage,
    'infer_example_GSM8K': infer_example_GSM8K,
    'infer_example_mask_selected_heads_GSM8K': infer_example_mask_selected_heads_GSM8K,
    'infer_example_attention_distance': infer_example_attention_distance_wrap
}

def is_infer_done(example):
    if 'model_output' in example:
        return True
    else:
        return False

def infer_file(src_path, tgt_path, infer_fn_name = 'infer_example', max_new_tokens = 16, save_interval = 10, args = None, save_fn = save_data_to_json):
    global infer_fn_dict
    global runner
    n_layer = runner.config['model']['network']['n_layer']
    n_head = runner.config['model']['network']['n_head']
    try:
        examples = load_data_from_file(src_path)
    except:
        print(f'read error file: {src_path}')
        return 1
    infer_fn = infer_fn_dict[infer_fn_name]
    if args and args.score_path is not None:
        infer_fn = partial(infer_fn, score_path = args.score_path, head_topn_percent = args.head_topn_percent, skip_layers_str = args.skip_layers_str)
    if os.path.exists(tgt_path):
        print('found cache')
        tgt_examples = load_data_from_file(tgt_path)
        new_examples = tgt_examples + examples[len(tgt_examples):]
        assert len(new_examples) == len(examples)
        examples = new_examples
    for idx, example in tqdm(enumerate(examples)):
        if is_infer_done(example):
            continue
        example['idx'] = idx
        infer_fn(example, max_new_tokens = max_new_tokens, n_layer = n_layer, n_head = n_head)
        if (idx+1) % save_interval == 0:
            print_rank_0(example)
            save_fn(examples[:idx+1], tgt_path)
    save_fn(examples, tgt_path)
    return 0

def infer_file_batch(src_path, tgt_path, infer_fn_name = 'infer_example', max_new_tokens = 16, save_interval = 10, args = None, save_fn = save_data_to_json):
    global infer_fn_dict
    try:
        examples = load_data_from_file(src_path)
    except:
        print(f'read error file: {src_path}')
        return 1
    infer_fn = infer_fn_dict[infer_fn_name]
    if tgt_path.endswith('.json'):
        save_fn = partial(save_data_to_json, pretty=True)
    if args and args.score_path is not None:
        infer_fn = partial(infer_fn, score_path = args.score_path, head_topn_percent = args.head_topn_percent)
    if os.path.exists(tgt_path):
        print('found cache')
        tgt_examples = load_data_from_file(tgt_path)
        new_examples = tgt_examples + examples[len(tgt_examples):]
        assert len(new_examples) == len(examples)
        examples = new_examples

    batch_examples = []
    batch_size = args.batch_size
    new_examples = []
    print_batch_idx = 0
    for idx, example in tqdm(enumerate(examples)):
        example['idx'] = idx
        if is_infer_done(example):
            new_examples.append(example)
            print_rank_0(f'load cache example cnt: {len(new_examples)}')
            continue
        batch_examples.append(example)
        if len(batch_examples) >= batch_size or idx == len(examples) - 1:
            results = infer_fn(batch_examples)
            batch_examples = []
            new_examples += results
            if (idx+1) >= save_interval * (print_batch_idx + 1):
                print_batch_idx += 1
                print_rank_0(results[-1])
                new_examples.sort(key = lambda x : x['idx'], reverse=False)
                print_rank_0(f'infer examples cnt: {idx+1}')
                save_fn(new_examples[:idx+1], tgt_path)
    new_examples.sort(key = lambda x : x['idx'], reverse=False)
    save_fn(new_examples, tgt_path)
    return 0

def multi_node_infer_hdfs_dir(src_path, tgt_path, infer_fn_name = 'infer_example', max_new_tokens = 16, save_interval = 10, args = None):
    global infer_fn_dict
    total_workers = int(int(os.environ["WORKER_NUM"]) * int(os.environ["WORKER_GPU"]) / args.n_gpus_for_one_model)
    cur_worker_global_idx = int(int(os.environ["ID"]) * int(os.environ["WORKER_GPU"]) / args.n_gpus_for_one_model + args.local_cur_worker_id)
    input_path_list = get_hdfs_file_path_list(src_path, '.parquet')
    output_path_list = get_hdfs_file_path_list(tgt_path, '.parquet')
    deduped_input_path_list = []
    output_file_name_list = [x.split('/')[-1] for x in output_path_list]
    output_file_name_set = set(output_file_name_list)
    for path in input_path_list:
        input_file_name = path.split('/')[-1]
        if input_file_name not in output_file_name_set:
            deduped_input_path_list.append(path)
    file_index_interval = args.file_index_interval
    if file_index_interval != '':
        file_index_intervals = eval(file_index_interval)
        valid_part_ids = set()
        for start_index, end_index in file_index_intervals:
            for i in range(start_index, end_index+1):
                valid_part_ids.add(i)
        new_deduped_input_path_list = []
        for path in deduped_input_path_list:
            part_id = path.split('/')[-1].split('-')[1]
            part_id = part_id.lstrip('0')
            try:
                part_id = int(part_id)
            except:
                print('parse error')
                print(f'part_id: {part_id}')
                print(f'path: {path}')
            if part_id in valid_part_ids:
                new_deduped_input_path_list.append(path)
        deduped_input_path_list = new_deduped_input_path_list
    if args.reverse_file_list: # 双trial同时刷数据
        deduped_input_path_list.reverse()
    print(f'worker info: {cur_worker_global_idx}/{total_workers}')
    path_list_cur_worker = [deduped_input_path_list[i] for i in range(cur_worker_global_idx, len(deduped_input_path_list), total_workers)]
    print(f'path_list_cur_worker:\n{path_list_cur_worker}')
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    file_cnt = len(path_list_cur_worker)
    for idx, path in tqdm(enumerate(path_list_cur_worker)):
        print(f'file info: {idx}/{file_cnt}')
        file_name = path.split('/')[-1]
        local_file_tgt_path = f'tmp/{file_name}'
        hdfs_file_tgt_path = tgt_path + '/' + file_name
        if 'batch' in infer_fn_name:
            ret = infer_file_batch(path, local_file_tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, save_fn = save_examples_to_parquet, args = args)
        else:
            ret = infer_file(path, local_file_tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, save_fn = save_examples_to_parquet, args = args)
        if ret == 0: # succeed
            os.system(f'hdfs dfs -put {local_file_tgt_path} {hdfs_file_tgt_path}')
            os.system(f'rm {local_file_tgt_path}')
        elif ret == 1:
            print(f'error file: {path}')
        print(f'processed {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/model_config_1B3_M4_4k.json', required=True, help='configs')
    parser.add_argument("--data_path", type=str, default=None, required=False, help="The input path")
    parser.add_argument("--model_name", type=str, default=None, required=False, help="model_name")
    parser.add_argument("--ckpt_path", type=str, default='merged_ckpts/megatron_merge_states.pt', required=False, help="the ckpt path")
    parser.add_argument("--ckpt_path2", type=str, default='', required=False, help="the ckpt path")
    parser.add_argument("--tokenizer_path", type=str, default=None, required=True, help="tokenizer path")
    parser.add_argument("--batch_size", type=int, default=1, required=True, help="infer batch_size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False, help="learning rate")
    parser.add_argument("--max_length", type=int, default=None, required=False, help="model max context length")
    parser.add_argument("--model_type", type=str, default='M4', required=True, help="model architecture")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, required=False, help="model gradient checkpointing")
    parser.add_argument("--src_path", type=str, default=None, required=True)
    parser.add_argument("--tgt_path", type=str, default=None, required=True)
    parser.add_argument("--infer_fn_name", type=str, default='infer_example', required=True)
    parser.add_argument("--score_path", type=str, default=None, required=False)
    parser.add_argument("--head_topn_percent", type=float, default=0.05, required=False)
    parser.add_argument("--skip_layers_str", type=str, default='', required=False)
    parser.add_argument("--max_new_tokens", type=int, default=128, required=False)
    parser.add_argument("--save_interval", type=int, default=100, required=False)
    parser.add_argument("--use_accelerate", type=bool, default=False, required=False)
    parser.add_argument("--use_plain_model_parallel", type=bool, default=False, required=False)
    parser.add_argument("--n_gpus_for_one_model", type=int, default=2, required=False)
    parser.add_argument("--multi_node_infer", type=bool, default=False, required=False)
    parser.add_argument("--local_cur_worker_id", type=int, default=0, required=False)
    parser.add_argument("--reverse_file_list", type=bool, default=False, required=False)
    parser.add_argument("--file_index_interval", type=str, default='', required=False)
    parser.add_argument("--n_layer", type=int, default=16, required=False)
    parser.add_argument("--n_head", type=int, default=20, required=False)
    parser.add_argument("--stage", type=str, default='1', required=False)
    args = parser.parse_args()

    try:
        torch.set_default_device("cuda")
    except:
        print('torch.set_default_device("cuda") failed, the version of torch is not compatible')
    if 'gpt_perf' in args.infer_fn_name:
        from task.test_loss_gpt_perf import infer_single_example_gpt_perf
    runner = Runner(args=args)

    # infer_file(args.src_path, args.tgt_path)
    if not args.multi_node_infer:
        if args.batch_size > 1:
            infer_file_batch(args.src_path, args.tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, save_fn = save_examples_to_parquet, args = args)
        else:
            infer_file(args.src_path, args.tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, args = args)
    else:
        multi_node_infer_hdfs_dir(args.src_path, args.tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, args = args)