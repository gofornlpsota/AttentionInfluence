https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier

python3 task/infer/edu_classifier/hf_model_infer.py

multi nodes cmd

``` bash
pip3 install torch==2.4.1
pip3 install transformers==4.38.2
pip3 install seaborn==0.13.2


cd /opt/tiger/llm-debug-train
bash run_multi_node_masked_loss_infer.sh task/simple_large_scale_infer.py \
--batch_size=128 \
--src_path='hdfs://haruna/home/x/open_source/smollm/smollm_pretrain_format_train_split' \
--tgt_path='hdfs://haruna/home/x/open_source/smollm/smollm_pretrain_format_train_split_hf_edu_classifier_output_20250724' \
--infer_fn_name='hf_infer_examples_batch' \
--save_interval=10000000000 \
--multi_node_infer=True \
--n_gpus_for_one_model=1 \
&& sleep 30m
```