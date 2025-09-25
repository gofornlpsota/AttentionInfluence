import os
import argparse
from tqdm import tqdm
from functools import partial
from util.data import load_data_from_file, save_data_to_json, save_examples_to_parquet, get_hdfs_file_path_list
from util.infer_util import is_infer_done
from util.torch_util import print_rank_0
from demo import process_examples

infer_fn_dict = {
    'ruled_based_filter_infer_examples_batch': process_examples
}

def infer_file_batch(src_path, tgt_path, infer_fn_name = 'ruled_based_filter_infer_examples_batch', max_new_tokens = 16, print_interval = 1000, save_interval = 1000, args = None, save_fn = save_data_to_json):
    global infer_fn_dict
    try:
        examples = load_data_from_file(src_path)
    except:
        print(f'read error file: {src_path}')
        return 1
    infer_fn = infer_fn_dict[infer_fn_name]
    if tgt_path.endswith('.json'):
        save_fn = partial(save_data_to_json, pretty=True)
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
    save_batch_idx = 0
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
            if (idx+1) >= print_interval * (print_batch_idx + 1):
                print_batch_idx += 1
                print_rank_0(results[-1])
                print_rank_0(f'infer examples cnt: {idx+1}')
            if (idx+1) >= save_interval * (save_batch_idx + 1):
                save_batch_idx += 1
                print_rank_0(results[-1])
                print_rank_0(f'infer examples cnt: {idx+1}')
                new_examples.sort(key = lambda x : x['idx'], reverse=False)
                save_fn(new_examples[:idx+1], tgt_path)
    new_examples.sort(key = lambda x : x['idx'], reverse=False)
    save_fn(new_examples, tgt_path)
    return 0

def multi_node_infer_hdfs_dir(src_path, tgt_path, infer_fn_name = 'ruled_based_filter_infer_examples_batch', max_new_tokens = 16, save_interval = 1000, args = None):
    global infer_fn_dict
    LOCAL_WORKER_NUM = int(os.environ.get("LOCAL_WORKER_NUM", 1))
    total_workers = int(int(os.environ["ARNOLD_WORKER_NUM"]) * LOCAL_WORKER_NUM) / args.n_gpus_for_one_model
    total_workers = int(total_workers)
    cur_worker_global_idx = int(int(os.environ["ARNOLD_ID"]) * LOCAL_WORKER_NUM) / args.n_gpus_for_one_model + args.local_cur_worker_id
    cur_worker_global_idx = int(cur_worker_global_idx)
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
                if part_id == '':
                    part_id = 0
                else:
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
            ret = infer_file_batch(path, local_file_tgt_path, infer_fn_name = args.infer_fn_name, save_interval = args.save_interval, save_fn = save_examples_to_parquet, args = args)
        else:
            ret = infer_file(path, local_file_tgt_path, infer_fn_name = args.infer_fn_name, save_interval = args.save_interval, save_fn = save_examples_to_parquet, args = args)
        if ret == 0: # succeed
            os.system(f'hdfs dfs -put {local_file_tgt_path} {hdfs_file_tgt_path}')
            os.system(f'rm {local_file_tgt_path}')
        elif ret == 1:
            print(f'error file: {path}')
        print(f'processed {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=None, required=False, help="model_name")
    parser.add_argument("--ckpt_path", type=str, default='merged_ckpts/megatron_merge_states.pt', required=False, help="the ckpt path")
    parser.add_argument("--tokenizer_path", type=str, default=None, required=False, help="tokenizer path")
    parser.add_argument("--batch_size", type=int, default=2, required=False, help="infer batch_size")
    parser.add_argument("--src_path", type=str, default=None, required=True)
    parser.add_argument("--tgt_path", type=str, default=None, required=True)
    parser.add_argument("--infer_fn_name", type=str, default='ruled_based_filter_infer_examples_batch', required=True)
    parser.add_argument("--n_gpus_for_one_model", type=int, default=1, required=False)
    parser.add_argument("--score_path", type=str, default=None, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=128, required=False)
    parser.add_argument("--save_interval", type=int, default=10000000, required=False)
    parser.add_argument("--multi_node_infer", type=bool, default=False, required=False)
    parser.add_argument("--local_cur_worker_id", type=int, default=0, required=False)
    parser.add_argument("--reverse_file_list", type=bool, default=False, required=False)
    parser.add_argument("--file_index_interval", type=str, default='', required=False)
    args = parser.parse_args()

    try:
        torch.set_default_device("cuda")
    except:
        print('torch.set_default_device("cuda") failed, the version of torch is not compatible')

    # infer_file(args.src_path, args.tgt_path)
    if not args.multi_node_infer:
        if args.batch_size > 1:
            infer_file_batch(args.src_path, args.tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, save_fn = save_examples_to_parquet, args = args)
        else:
            infer_file(args.src_path, args.tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, args = args)
    else:
        multi_node_infer_hdfs_dir(args.src_path, args.tgt_path, infer_fn_name = args.infer_fn_name, max_new_tokens = args.max_new_tokens, save_interval = args.save_interval, args = args)