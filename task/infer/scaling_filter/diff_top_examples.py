from util.data import load_data_from_json, save_data_to_json

def get_score_group(score, quantiles):
    for idx, th in enumerate(quantiles):
        if score >= th:
            continue
        else:
            return idx
    return idx+1

def group_examples(examples, model_output_field = 'model_output', group_cnt = 3):
    example_cnt = len(examples)
    print(example_cnt)
    scores = []
    for example in examples:
        output = example[model_output_field]
        if 'loss_delta_rel' in output:
            score = output['loss_delta_rel']
        else:
            score = output['final_score']

        scores.append(score)
    scores.sort()

    quantiles = []
    for i in range(1, group_cnt):
        idx = int(example_cnt * i / group_cnt)
        quantiles.append(scores[idx])

    for example in examples:
        output = example[model_output_field]
        if 'loss_delta_rel' in output:
            score = output['loss_delta_rel']
        else:
            score = output['final_score']
        score_group_id = get_score_group(score, quantiles)
        example['score_group_id'] = score_group_id

def compare_rank_sim(path1, path2, limit_cnt = 1000, group_cnt = 4, model_output_field_1 = 'model_output', model_output_field_2 = 'sf_model_output'):
    examples_1 = load_data_from_json(path1)[:limit_cnt]
    examples_2 = load_data_from_json(path2)[:limit_cnt]
    group_examples(examples_1, model_output_field = model_output_field_1, group_cnt = group_cnt)
    group_examples(examples_2, model_output_field = model_output_field_2, group_cnt = group_cnt)
    all_cnt = len(examples_1)
    diff_cnt = 0
    for example_1, example_2 in zip(examples_1, examples_2):
        if example_1['score_group_id'] != example_2['score_group_id']:
            diff_cnt += 1
    print('group_cnt:', group_cnt)
    print(diff_cnt, all_cnt, diff_cnt / all_cnt)

def show_top_cases(path1, path2, limit_cnt = 1000, group_cnt = 7, model_output_field_2 = 'sf_model_output'):
    examples_1 = load_data_from_json(path1)[:limit_cnt]
    examples_2 = load_data_from_json(path2)[:limit_cnt]
    group_examples(examples_1, model_output_field = 'model_output', group_cnt = group_cnt)
    group_examples(examples_2, model_output_field = model_output_field_2, group_cnt = group_cnt)
    for example_1, example_2 in zip(examples_1, examples_2):
        example_2['score_group_id_1'] = example_1['score_group_id']
        example_2['model_output'] = example_1['model_output']
    examples_2.sort(key=lambda x: x['model_output']['loss_delta_rel'], reverse=True)
    save_data_to_json(examples_2, 'tmp/cmp/v4/cmp.json', pretty=True)

if __name__ == '__main__':
    # show_top_cases('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json',
    #                '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json',
    #                limit_cnt = 1000, group_cnt = 7)

    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', limit_cnt = 1000, group_cnt = 2, model_output_field_2 = 'sf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', limit_cnt = 1000, group_cnt = 3, model_output_field_2 = 'sf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', limit_cnt = 1000, group_cnt = 4, model_output_field_2 = 'sf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', limit_cnt = 1000, group_cnt = 5, model_output_field_2 = 'sf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', limit_cnt = 1000, group_cnt = 6, model_output_field_2 = 'sf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', limit_cnt = 1000, group_cnt = 7, model_output_field_2 = 'sf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', limit_cnt = 1000, group_cnt = 8, model_output_field_2 = 'sf_model_output')

    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 2, model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 3, model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 4, model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 5, model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 6, model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 7, model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 8, model_output_field_2 = 'rbf_model_output')

    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 2, model_output_field_1 = 'sf_model_output', model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 3, model_output_field_1 = 'sf_model_output', model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 4, model_output_field_1 = 'sf_model_output', model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 5, model_output_field_1 = 'sf_model_output', model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 6, model_output_field_1 = 'sf_model_output', model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 7, model_output_field_1 = 'sf_model_output', model_output_field_2 = 'rbf_model_output')
    # compare_rank_sim('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json', '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json', limit_cnt = 1000, group_cnt = 8, model_output_field_1 = 'sf_model_output', model_output_field_2 = 'rbf_model_output')

    show_top_cases('7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_scaling_filter.json',
                   '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json',
                   limit_cnt = 1000, group_cnt = 7, model_output_field_2 = 'rbf_model_output')