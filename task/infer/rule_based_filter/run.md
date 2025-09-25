```bash
pip3 install numpy==1.23.5
pip3 install spacy==3.8.7
pip3 install textstat==0.7.10
pip3 install nltk==3.9.1

python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('words')"

python3 task/infer/rule_based_filter/demo.py
```

single node/multi nodes cmd
```bash
cd /opt/tiger

pip3 install spacy==3.8.7
pip3 install textstat==0.7.10
pip3 install nltk==3.9.1
pip3 install numpy==1.23.5
pip3 install zstandard==0.24.0

python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('words')"

cd /opt/tiger/script
export LOCAL_WORKER_NUM=8
bash task/infer/run_multi_node_infer.sh task/infer/rule_based_filter/simple_large_scale_infer.py \
--batch_size=1024 \
--src_path='hdfs://haruna/home/x/open_source/smollm/smollm_pretrain_format_train_split' \
--tgt_path='hdfs://haruna/home/x/open_source/smollm/smollm_pretrain_format_train_split_rule_based_filter_output_20250921' \
--infer_fn_name='ruled_based_filter_infer_examples_batch' \
--save_interval=10000000000 \
--multi_node_infer=True \
--n_gpus_for_one_model=1 \
&& sleep 30m
```