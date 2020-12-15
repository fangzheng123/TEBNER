
# BERT词连接模型
CUDA_VISIBLE_DEVICES="0"

# 当前任务领域
TASK_NAME="bc5cdr"

# 实体类别
LABELS="chemical,disease"

# 根路径
ROOT_DIR="/data/fangzheng/bert_autoner"
# 任务根路径
TASK_DATA_DIR=$ROOT_DIR/${TASK_NAME}
# 格式化数据路径
FORMAT_DATA_DIR=$TASK_DATA_DIR/format

# 预训练模型类别，如BERT,Robert等
MODEL_TYPE="biobert-base-cased-v1.1"
# 预训练模型路径
PRE_TRAINED_MODEL_DIR=$ROOT_DIR/pre_trained_model/${MODEL_TYPE}/
# 微调模型存储路径
FINE_TUNING_MODEL_DIR=$TASK_DATA_DIR/model

# 创建相关目录
echo ${green}=== Mkdir ===${reset}
mkdir -p $FINE_TUNING_MODEL_DIR

####################用户需提供的数据#####################
# 模型训练、验证、测试文件
TRAIN_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/train_dev_distance_data
DEV_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/test_distance_data
TEST_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/test_distance_data

###################训练BERT Word模型#####################
# 日志
LOG_FILE=word_log_3

nohup python -u run_model/run_bert_word.py \
  --do_train \
  --task_name=$TASK_NAME \
  --gpu_devices=$CUDA_VISIBLE_DEVICES \
  --pre_trained_model_path=$PRE_TRAINED_MODEL_DIR \
  --model_type=$MODEL_TYPE \
  --model_dir=$FINE_TUNING_MODEL_DIR \
  --train_data_path=$TRAIN_DISTANCE_DATA_PATH \
  --dev_data_path=$DEV_DISTANCE_DATA_PATH \
  --test_data_path=$TEST_DISTANCE_DATA_PATH \
  --label_names=$LABELS \
  --require_improvement=1500 \
  --max_seq_length=256 \
  --per_eval_batch_step=40 \
  --per_gpu_train_batch_size=86 \
  --per_gpu_dev_batch_size=170 \
  --per_gpu_test_batch_size=200 \
  --num_train_epochs=20 \
  --seed=42 \
  > $LOG_FILE 2>&1 &