updated_args=()
for arg in "$@"; do
    if [[ "$arg" == "--wait_job_time="* ]]; then
        wait_job_time="${arg#*=}"
    elif [[ "$arg" == "--n_gpus_for_one_model="* ]]; then
        n_gpus_for_one_model="${arg#*=}"
        updated_args+=("$arg")
    elif [[ "$arg" == "--tgt_path="* ]]; then
        output_hdfs_dir="${arg#*=}"
        updated_args+=("$arg")
    else
        updated_args+=("$arg")
    fi
done
set -- "${updated_args[@]}"

hdfs dfs -mkdir -p $output_hdfs_dir
echo $ARNOLD_WORKER_GPU
echo $n_gpus_for_one_model
if [ -n "$LOCAL_WORKER_NUM" ]; then
    local_worker_num=$LOCAL_WORKER_NUM
else
    local_worker_num=$(expr $ARNOLD_WORKER_GPU \/ $n_gpus_for_one_model)
fi
devices=(0 1 2 3 4 5 6 7)
for ((i=0; i < $local_worker_num; i=i+1)); do
    start_index=$((i * n_gpus_for_one_model))
    end_index=$((start_index + n_gpus_for_one_model - 1))
    if [ $i -eq $(($local_worker_num - 1)) ]; then
        end_index=$((8 - 1))
    fi
    worker_devices=("${devices[@]:$start_index:$((end_index - start_index + 1))}")
    joined_devices=$(IFS=, ; echo "${worker_devices[*]}")
    echo "Worker $i -- GPU decices: $joined_devices"
    export CUDA_VISIBLE_DEVICES=$joined_devices
    python3 "${updated_args[@]}" --local_cur_worker_id=$i &
    sleep 5
done

for job in `jobs -p`
do
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
    echo "All background tasks completed"
else
    echo "FAIL! ($FAIL)"
fi

sleep 60