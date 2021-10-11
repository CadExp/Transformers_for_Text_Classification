"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/9/29
@description: null
"""

data_dir=None
model_type=None
model_name_or_path=None
task_name=None
output_dir=None

## Other parameters
config_name=""
tokenizer_name=""
cache_dir=""
max_seq_length=128
do_train=True
do_eval=True
do_predict=True
evaluate_during_training=True
do_lower_case=True

gradient_accumulation_steps=1
per_gpu_train_batch_size=8
per_gpu_eval_batch_size=8
learning_rate=5e-5
weight_decay=0.0
adam_epsilon=1e-8
max_grad_norm=1.0
num_train_epochs=3.0
max_steps=-1
warmup_steps=0

logging_steps=50,
save_steps=50,
eval_all_checkpoints=True
no_cuda=True
overwrite_output_dir=True
overwrite_cache=True
seed=42,

fp16=True
fp16_opt_level='O1',
local_rank=-1,

# Additional layer parameters
# CNN
filter_num=256
filter_sizes='3,4,5'

# LSTM
lstm_hidden_size=300
lstm_layers=2
lstm_dropout=0.5

# GRU
gru_hidden_size=300
gru_layers=2
gru_dropout=0.5



class Args:
    data_dir = "../dataset/policy"
    output_dir = "../results/policy/bert"
    model_name_or_path = "bert"
    model_type = "bert_base"
    task_name = "policy"

    ## Other parameters
    config_name=""
    tokenizer_name=""
    cache_dir=""
    max_seq_length=512
    do_train=True
    do_eval=True
    do_predict=True
    evaluate_during_training=True
    do_lower_case=True

    gradient_accumulation_steps=2
    per_gpu_train_batch_size=1
    per_gpu_eval_batch_size=16
    learning_rate=2-5
    weight_decay=0.0
    adam_epsilon=1e-8
    max_grad_norm=1.0
    num_train_epochs=3.0
    max_steps=-1
    warmup_steps=0

    logging_steps=14923
    save_steps=14923
    eval_all_checkpoints=False
    no_cuda=False
    overwrite_output_dir=True
    overwrite_cache=False
    seed=42

    fp16=False
    fp16_opt_level='O1'
    local_rank=-1

    # Additional layer parameters
    # CNN
    filter_num=256
    filter_sizes='3,4,5'

    # LSTM
    lstm_hidden_size=512
    lstm_layers=1
    lstm_dropout=0.1

    # GRU
    gru_hidden_size=512
    gru_layers=1
    gru_dropout=0.1
