```bash
fuser -v /dev/nvidia* | grep -o '[0-9]*' | xargs -I {} kill -9 {}
```

plain needle
```bash
CUDA_VISIBLE_DEVICES=0 python3 offline_infer.py --config=config/model_config_1B3_LLaMA2_4k.json --tokenizer_path=bbpe --batch_size=1 --max_length=4096 --model_type=LLaMA2 --src_path='tmp/plain_needles_eval_4K_20241224.json' --tgt_path='tmp/1.3B_LLaMA2_D70_1499B_plain_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json' --infer_fn_name='infer_example_all_head_top_attended_tokens_two_pass' --max_new_tokens=32 --save_interval=10 --ckpt_path='merged_ckpt_1.3B_LLaMA2_D70_1499B/megatron_merge_states.pt'
```

reasoning needle
```bash
CUDA_VISIBLE_DEVICES=0 python3 offline_infer.py --config=config/model_config_1B3_LLaMA2_4k.json --tokenizer_path=bbpe --batch_size=1 --max_length=4096 --model_type=LLaMA2 --src_path='tmp/reasoning_needles_eval_4K_20241224.json' --tgt_path='tmp/reasoning_needles_eval_4K_20241224_attn_output_retrieval_scores_token_level_20241224_two_pass.json' --infer_fn_name='infer_example_all_head_top_attended_tokens_two_pass' --max_new_tokens=32 --save_interval=10 --ckpt_path='merged_ckpt_1.3B_LLaMA2_D70_1499B/megatron_merge_states.pt'

CUDA_VISIBLE_DEVICES=0 python3 offline_infer.py --config=config/model_config_1B3_LLaMA2_4k.json --tokenizer_path=bbpe --batch_size=1 --max_length=4096 --model_type=LLaMA2 --src_path='tmp/reasoning_needles_eval_4K_20241224.json' --tgt_path='tmp/reasoning_needles_eval_4K_20241224_attn_output_soft_retrieval_scores_token_level_20241224_two_pass.json' --infer_fn_name='infer_example_all_head_top_attended_tokens_two_pass' --max_new_tokens=32 --save_interval=10 --ckpt_path='merged_ckpt_1.3B_LLaMA2_D70_1499B/megatron_merge_states.pt'

CUDA_VISIBLE_DEVICES=0 python3 offline_infer.py --config=config/model_config_1B3_LLaMA2_4k.json --tokenizer_path=bbpe --batch_size=1 --max_length=4096 --model_type=LLaMA2 --src_path='tmp/reasoning_needles_eval_4K_20241224.json' --tgt_path='1.3B_LLaMA2_D70_1499B_reasoning_needles_eval_4K_20241224_attn_output_needle_span_topk_soft_retrieval_scores_token_level_20241224_two_pass.json' --infer_fn_name='infer_example_all_head_top_attended_tokens_two_pass' --max_new_tokens=32 --save_interval=10 --ckpt_path='merged_ckpt_1.3B_LLaMA2_D70_1499B/megatron_merge_states.pt'
```

7B 8K
```bash
CUDA_VISIBLE_DEVICES=0 python3 offline_infer.py --config=config/model_config_7B_LLaMA2_4k.json --tokenizer_path= --batch_size=1 --max_length=4096 --model_type=LLaMA2 --src_path='tmp/key_passage_retrieval_32k_fewshot_filtered_short_4K.json' --tgt_path='tmp/7B_LLaMA2_key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json' --infer_fn_name='infer_example_all_head_top_attended_tokens_two_pass' --max_new_tokens=16 --save_interval=100 --ckpt_path='merged_ckpt_7B_LLaMA2/megatron_merge_states.pt'
```


run infer loss
```bash
cd /opt/tiger/llm-debug-train && bash init_env.sh ckpt_hdfs_path='hdfs://haruna/home/x/1.3B_dense/megatron_merge_states.pt' && bash run_multi_node_masked_loss_infer.sh offline_infer.py --config=config/model_config_1.3B_dense_4k.json  --tokenizer_path=bbpe --max_length=8192 --batch_size=3 --model_type=llama2 --src_path='hdfs://haruna/home/x/attention_influence/reasoning_data/x' --tgt_path='hdfs://haruna/home/x/attention_influence/reasoning_data/x_masked_attention_head_loss_output' --infer_fn_name='infer_example_mask_selected_heads_delta_loss_batch'  --score_path='tmp/1.3B_dense_key_passage_retrieval_32k_fewshot_filtered_short_4K_attn_output_retrieval_scores_token_level_20241023_two_pass.json' --head_topn_percent=0.05 --save_interval=10000000 --ckpt_path='merged_ckpt_1.3B_dense/megatron_merge_states.pt' --multi_node_infer=True --n_gpus_for_one_model=1
```