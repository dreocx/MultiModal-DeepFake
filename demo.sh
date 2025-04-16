EXPID=your_best_model_dir_name

HOST='127.0.0.1'
PORT='1'

NUM_GPU=1

python demo.py \
--config 'configs/test.yaml' \
--output_dir 'results' \
--launcher pytorch \
--rank 0 \
--log_num 'log20250317_220331' \
--dist-url tcp://${HOST}:1003${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--test_epoch best \

