
# 当前任务领域
TASK_NAME="bc5cdr"

# 根路径
ROOT_DIR="/data/fangzheng/bert_autoner"
# 任务根路径
TASK_DATA_DIR=$ROOT_DIR/${TASK_NAME}
# 格式化数据路径
FORMAT_DATA_DIR=$TASK_DATA_DIR/format
# 短语相关路径
PHRASE_DIR=$TASK_DATA_DIR/phrase/

####################用户需提供的数据#####################
# 种子实体文件
SEED_ENTITY_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_dict.txt
# 挖掘出的短语路径
PHRASE_PATH=$PHRASE_DIR/${TASK_NAME}_autophrase.txt
# 训练数据
TRAIN_DATA_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_train.json
# 验证数据
DEV_DATA_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_dev.json
# 测试数据
TEST_DATA_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_test.json

####################远程打标生成的数据#####################
# 模型训练、验证、测试文件
TRAIN_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/train_distance_data
DEV_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/dev_distance_data
TEST_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/test_distance_data

####################新增分类短语#####################
# 分类短语数据
PHRASE_LABEL_PATH=$PHRASE_DIR/phrase_label.txt
# 模型训练、验证、测试文件
ADD_TRAIN_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/add_train_distance_data
ADD_DEV_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/add_dev_distance_data
ADD_TEST_DISTANCE_DATA_PATH=$FORMAT_DATA_DIR/add_test_distance_data

# 日志
LABEL_LOG_FILE=label_data_log
nohup python -u run_model/run_data_label.py \
    --do_add_distance \
    --task_name=$TASK_NAME \
    --seed_entity_path=$SEED_ENTITY_PATH \
    --phrase_path=$PHRASE_PATH \
    --phrase_label_path=$PHRASE_LABEL_PATH \
    --train_data_path=$TRAIN_DATA_PATH \
    --dev_data_path=$DEV_DATA_PATH \
    --test_data_path=$TEST_DATA_PATH \
    --train_distance_data_path=$TRAIN_DISTANCE_DATA_PATH \
    --dev_distance_data_path=$DEV_DISTANCE_DATA_PATH \
    --test_distance_data_path=$TEST_DISTANCE_DATA_PATH \
    --add_train_distance_data_path=$ADD_TRAIN_DISTANCE_DATA_PATH \
    --add_dev_distance_data_path=$ADD_DEV_DISTANCE_DATA_PATH \
    --add_test_distance_data_path=$ADD_TEST_DISTANCE_DATA_PATH \
    > $LABEL_LOG_FILE 2>&1 &
