
# Mention分类模型
CUDA_VISIBLE_DEVICES="1"

# 当前任务领域
TASK_NAME="bc5cdr"

# 实体类别
LABELS="chemical,disease"

# 根路径
ROOT_DIR=""
# 任务根路径
TASK_DATA_DIR=$ROOT_DIR/${TASK_NAME}
# 格式化数据路径
FORMAT_DATA_DIR=$TASK_DATA_DIR/format
# 短语相关路径
PHRASE_DIR=$TASK_DATA_DIR/phrase

# 预训练模型类别，如BERT,Robert等
MODEL_TYPE="biobert-base-cased-v1.1"
# MODEL_TYPE="bert-base-cased"

# 预训练模型路径
PRE_TRAINED_MODEL_DIR=$ROOT_DIR/pre_trained_model/${MODEL_TYPE}/
# 微调模型存储路径
FINE_TUNING_MODEL_DIR=$TASK_DATA_DIR/model/dis_supervised_model

# 种子实体文件
SEED_ENTITY_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_dict.txt
# 标注实体类型
GOLD_ENTITY_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_gold_entity.txt
# 短语分数结果
PHRASE_TYPE_SCORE_PATH=$PHRASE_DIR/phrase_type_score.txt
# 短语类别预测数据
PHRASE_LABEL_PATH=$PHRASE_DIR/phrase_label.txt

# 日志目录
LOG_DIR=${TASK_NAME}_log

# 创建相关目录
echo ${green}=== Mkdir ===${reset}
mkdir -p $FINE_TUNING_MODEL_DIR
mkdir -p $LOG_DIR

####################用户需提供的数据#####################
# 模型训练、验证、测试文件
TRAIN_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/train_dev_distance_data
DEV_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/test_distance_data
TEST_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/test_distance_data
PREDICT_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/train_distance_data

# 日志
LOG_FILE=$LOG_DIR/dis_train_mention_classify_log

nohup python -u run_model/run_mention_classify.py \
  --do_train \
  --task_name=$TASK_NAME \
  --gpu_devices=$CUDA_VISIBLE_DEVICES \
  --pre_trained_model_path=$PRE_TRAINED_MODEL_DIR \
  --model_type=$MODEL_TYPE \
  --model_dir=$FINE_TUNING_MODEL_DIR \
  --seed_entity_path=$SEED_ENTITY_PATH \
  --train_data_path=$TRAIN_DISTANCE_DATA_PATH \
  --dev_data_path=$DEV_DISTANCE_DATA_PATH \
  --test_data_path=$TEST_DISTANCE_DATA_PATH \
  --pred_data_path=$PREDICT_DISTANCE_DATA_PATH \
  --gold_entity_path=$GOLD_ENTITY_PATH \
  --phrase_type_score_path=$PHRASE_TYPE_SCORE_PATH \
  --phrase_label_path=$PHRASE_LABEL_PATH \
  --label_names=$LABELS \
  --loss_type=ce \
  --require_improvement=1500 \
  --max_seq_length=256 \
  --per_eval_batch_step=40 \
  --per_gpu_train_batch_size=96 \
  --per_gpu_dev_batch_size=200 \
  --per_gpu_test_batch_size=200 \
  --num_train_epochs=20 \
  --dnn_hidden_size=256 \
  --seed=42 \
  > $LOG_FILE 2>&1 &