pip3 install transformers==4.56.1
pip3 install accelerate==1.10.1

hdfs dfs -get .../x/7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.gz.parquet

CUDA_VISIBLE_DEVICES=0 python3 task/infer/scaling_filter/demo.py

single node/multi nodes cmd
```bash
cd /opt/tiger

pip3 install pyarrow==11.0.0
pip3 install torch==2.4.1
pip3 install transformers==4.56.1
pip3 install accelerate==1.10.1

cd /opt/tiger/script
bash task/infer/run_multi_node_infer.sh task/infer/scaling_filter/simple_large_scale_infer.py \
--batch_size=1 \
--src_path='hdfs://haruna/home/x/open_source/smollm/smollm_pretrain_format_train_split' \
--tgt_path='hdfs://haruna/home/x/open_source/smollm/smollm_pretrain_format_train_split_scaling_filter_output_20250918' \
--infer_fn_name='hf_infer_examples_batch' \
--save_interval=10000000000 \
--multi_node_infer=True \
--n_gpus_for_one_model=1 \
--file_index_interval='[(0,127)]' \
&& sleep 30m
```