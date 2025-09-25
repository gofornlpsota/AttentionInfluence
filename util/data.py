import os
import json
import yaml
from tqdm import tqdm
import random
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from subprocess import Popen, PIPE
import io
import zstandard as zstd
from util.time import gen_time_str

def yaml_multiline_string_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        data = '\n'.join([line.rstrip() for line in data.strip().splitlines()])
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def load_data_from_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

def save_data_to_json(data, path, pretty = False):
    dir_path = os.path.dirname(path)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'w', encoding = 'utf8') as f:
        if pretty:
            json.dump(data, f, ensure_ascii = False, indent = 4)
        else:
            json.dump(data, f, ensure_ascii = False)
    print(f'save data into {path}')

def save_data_to_json_string_txt(data, path, pretty = False):
    dir_path = os.path.dirname(path)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    def print_item_to_file(d, f):
        if isinstance(d, list):
            for x in d:
                print_item_to_file(x, f)
        elif isinstance(d, dict):
            for k,v in d.items():
                print(f'{k}: {v}', file=f)
                print('-'*10, file=f)
        else:
            print(d, file=f)
            print('-'*10, file=f)
    with open(path, 'w', encoding = 'utf8') as f:
        if pretty:
            print_item_to_file(data, f)
        else:
            s = json.dumps(data, ensure_ascii = False, indent = 4)
            f.write(s)
    print(f'save data into {path}')

def save_data_to_json_for_debug(data):
    path = f'debug_{gen_time_str()}.json'
    save_data_to_json(data, path)

def load_data_from_yaml(path):
    data = yaml.load(open(path, "r"), Loader = yaml.FullLoader)
    return data

def save_data_to_yaml(data, path, clear_format = False):
    with open(path, 'w') as f:
        if clear_format:
            yaml.add_representer(str, yaml_multiline_string_presenter)
            yaml.dump(data, f, default_style='|', default_flow_style = False, sort_keys=False, encoding = 'utf8', allow_unicode = True)
        else:
            yaml.dump(data, f, default_flow_style = False, sort_keys=False, encoding = 'utf8', allow_unicode = True)

def load_examples_from_jsonl(path):
    if 'hdfs' in path:
        local_path = os.path.join('/tmp', path.split('/')[-1])
        if os.path.exists(local_path):
            os.remove(local_path)
        os.system(f'hdfs dfs -get {path} {local_path}')
        path = local_path
    with open(path, 'r', encoding = 'utf8') as f:
        # examples = []
        # for line in f:
        #     example = json.loads(line)
        #     examples.append(example)
        #     break
        examples = [json.loads(line) for line in f]
    return examples

def save_examples_to_jsonl(examples, path, ensure_ascii=False):
    dir_path = os.path.dirname(path)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'w', encoding = 'utf8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=ensure_ascii) + '\n')
    print(f'save examples into {path}')

def load_examples_from_xlsx(path, skip_first_row = False):
    examples = []
    df = pd.read_excel(path, index_col = None)
    df = df.fillna('null')
    for idx, row in df.iterrows():
        if skip_first_row and idx == 0:
            continue
        examples.append(row)
    return examples

def load_examples_from_txt_or_csv(path, skip_first_line = False, read_csv_by_pandas = True, verbose = False):
    examples = []
    if path.endswith('.csv') and read_csv_by_pandas:
        df = pd.read_csv(path, header = None, verbose = verbose)
        for index, row in df.iterrows():
            if skip_first_line and index == 0:
                continue
            examples.append(row)
    else:
        with open(path, 'r', encoding = 'utf8') as f:
            if skip_first_line:
                f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(line)
    return examples

def load_examples_from_parquet(path):
    if 'hdfs' in path:
        examples = load_examples_from_hdfs_parquet(path)
    else:
        examples = pq.read_table(path).to_pylist()
    return examples

def tran_nested_example_to_parquet_saved_format(d, depth = 0):
    if isinstance(d, dict):
        for sub_k,sub_v in d.items():
            tran_nested_example_to_parquet_saved_format(sub_v, depth+1)
            if depth != 0:
                sub_v = json.dumps(sub_v, ensure_ascii=False)
                d[sub_k] = sub_v
    elif isinstance(d, list) or isinstance(d, tuple):
        for x in d:
            tran_nested_example_to_parquet_saved_format(x, depth+1)
    else:
        pass

def save_few_examples_to_parquet_pandas(examples, path):
    data = tran_list_d_to_d_list(examples)
    df = pd.DataFrame(data=data)
    df.to_parquet(path)

def save_few_examples_to_parquet(data_dict, path, complex_format = False):
    if isinstance(data_dict, list):
        data_dict = tran_list_d_to_d_list(data_dict, complex_format)
    df = pd.DataFrame(data=data_dict)
    table = pa.Table.from_pandas(df)
    dir_path = os.path.dirname(path)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    pq.write_table(table, path)
    print(f'save examples into {path}')

def save_many_examples_to_parquet(examples, path_prefix, multiple = 10, use_slash = False):
    example_cnt = len(examples)
    print(example_cnt)
    batch_size = example_cnt // multiple + 1
    # 样本数超过了20w，pyarrow就会报错存不了了，不是数据的问题
    for i in range(0, example_cnt, batch_size):
        part_index = i // batch_size + 1
        batch_examples = examples[i:i+batch_size]
        if use_slash:
            path = f'{path_prefix}/part{part_index}.parquet'
        else:
            path = f'{path_prefix}_part{part_index}.parquet'
        save_examples_to_parquet(batch_examples, path)
        print(f'save {len(batch_examples)} examples into {path}')

def load_examples_from_hdfs_parquet(path):
    fs = pa.hdfs.connect(path)
    with fs.open(path) as f:
        examples = pq.read_table(path).to_pylist()
    fs.close()
    return examples

def load_data_from_file(path, skip_first_line = False, read_csv_by_pandas = True, verbose = False):
    suffix = path.split('.')[-1]
    if suffix == 'jsonl':
        return load_examples_from_jsonl(path)
    elif suffix == 'json':
        return load_data_from_json(path)
    elif suffix in ['txt', 'csv']:
        return load_examples_from_txt_or_csv(path, skip_first_line, read_csv_by_pandas, verbose)
    elif suffix == 'xlsx':
        return load_examples_from_xlsx(path, skip_first_line)
    elif suffix == 'parquet':
        return load_examples_from_parquet(path)

def tran_jsonl_to_json(src_path, tgt_path, format = True):
    examples = load_examples_from_jsonl(src_path)
    with open(tgt_path, 'w', encoding = 'utf8') as f:
        if format:
            json.dump(examples, f, indent = 4, ensure_ascii = False)
        else:
            json.dump(examples, f, ensure_ascii = False)

def tran_json_to_parquet(src_path, tgt_path, multiple = 10):
    examples = load_data_from_json(src_path)
    save_examples_to_parquet(examples, tgt_path, multiple)

def save_examples_to_parquet(examples, tgt_path, multiple = 10, is_large_file = False):
    print(len(examples))
    if is_large_file or len(examples) <= 200000:
        save_few_examples_to_parquet(examples, tgt_path)
    else:
        # https://issues.apache.org/jira/browse/ARROW-17137
        print('too many examples ...')
        save_many_examples_to_parquet(examples, tgt_path, multiple)

def tran_txt_to_str(path):
    with open(path, 'r', encoding = 'utf8') as f:
        s = f.read()
    return s

def pretty_print(items):
    for item in items:
        print(item)

def deduplicate_list(items):
    ret = []
    appeared = set()
    for item in items:
        item_cmp = json.dumps(item)
        if item_cmp not in appeared:
            ret.append(item)
            appeared.add(item_cmp)
    return ret

def tran_list_d_to_d_list(list_d, complex_format = False):
    d_list = defaultdict(list)
    for d in list_d:
        if complex_format:
            tran_nested_example_to_parquet_saved_format(d)
        for k,v in d.items():
            d_list[k].append(v)
    return d_list

def merge_examples_file(*paths, save_format = 'jsonl', pretty = True):
    assert len(paths) >= 3
    src_paths = paths[:-1]
    tgt_path = paths[-1]
    examples = []
    for src_path in src_paths:
        examples += load_data_from_file(src_path)
    if save_format == 'jsonl':
        save_examples_to_jsonl(examples, tgt_path)
    else:
        save_data_to_json(examples, tgt_path, pretty)

def align_example_field_file(src_path_1, src_path_2, field, tgt_path):
    examples_1 = load_examples_from_jsonl(src_path_1)
    examples_2 = load_examples_from_jsonl(src_path_2)
    for example_1, examples_2 in zip(examples_1, examples_2):
        example_1[field] = examples_2[field]
    save_examples_to_jsonl(examples_1, tgt_path)

def flat_2d_list(l):
    ret = [x for sub_l in l for x in sub_l]
    return ret

def get_hdfs_file_path_list(path, suffix):
    file_paths = []
    if path.startswith("hdfs"):
        pipe = Popen((f"hdfs dfs -ls {path}"), shell=True, stdout=PIPE)
        for line in pipe.stdout:
            line = line.strip()
            if len(line.split()) < 5:
                continue
            line_str = line.split()[-1].decode("utf8")
            if line_str.endswith(suffix):
                file_paths.append(line_str)
        pipe.stdout.close()
    else:
        raise NotImplementedError
    
    return file_paths

def load_all_examples_from_hdfs_dir_parquet(path):
    file_paths = get_hdfs_file_path_list(path, '.parquet')
    examples = []
    for file_path in tqdm(file_paths):
        examples += load_examples_from_parquet(file_path)
    return examples

def load_all_examples_from_local_dir(dir_path, suffix):
    file_paths = []
    for path in os.listdir(dir_path):
        if path.endswith(suffix):
            path = os.path.join(dir_path, path)
            file_paths.append(path)
    examples = []
    for file_path in tqdm(file_paths):
        examples += load_data_from_file(file_path)
    return examples

def load_data_from_file_or_dir(path):
    if 'hdfs' in path and not path.endswith('.parquet'):
        examples = load_all_examples_from_hdfs_dir_parquet(path)
    else:
        examples = load_data_from_file(path)
    return examples

def sample_file(src_file_path, tgt_file_path, sample_num = 100):
    examples = load_examples_from_jsonl(src_file_path)
    random.shuffle(examples)
    sampled_examples = examples[:sample_num]
    save_examples_to_jsonl(sampled_examples, tgt_file_path)
    save_examples_to_parquet(sampled_examples, tgt_file_path.replace('.jsonl', '.parquet'))

def tran_parquet_to_jsonl(src_path, tgt_path):
    examples = load_examples_from_parquet(src_path)
    save_examples_to_jsonl(examples, tgt_path)

def tran_parquet_to_json(src_path, tgt_path, sample_cnt=None, pretty=True):
    examples = load_examples_from_parquet(src_path)
    for example in examples:
        example.pop('model_input_ids', None)
        example.pop('chunk_text', None)
    if sample_cnt is not None:
        random.shuffle(examples)
        examples = examples[:sample_cnt]
    save_data_to_json(examples, tgt_path, pretty=pretty)

def load_field_and_vals_from_txt(path):
    lines = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    field = lines[0]
    d = {field: lines[1:]}
    return d

def get_first_key_from_dict(d):
    for k in d:
        return k

def shuffle_data(path, save_format = 'jsonl'):
    examples = load_data_from_file(path)
    random.shuffle(examples)
    save_path = '.'.join(path.split('.')[:-1]) + '_shuffled' + f'.{save_format}'
    print(save_path)
    if save_format == 'json':
        save_data_to_json(examples, save_path)
    elif save_format == 'jsonl':
        save_examples_to_jsonl(examples, save_path)
    else:
        save_examples_to_parquet(examples, save_path)

def read_jsonl_zst_stream(path, max_lines=None):
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for i, line in enumerate(text_stream):
                if max_lines and i >= max_lines:
                    break
                yield json.loads(line)