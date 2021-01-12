
# 当前任务领域
TASK_NAME="bc5cdr"

# 根路径
ROOT_DIR="/data/fangzheng/bert_autoner"
# 任务根路径
TASK_DATA_DIR=$ROOT_DIR/${TASK_NAME}
# 格式化数据路径
FORMAT_DATA_DIR=$TASK_DATA_DIR/format
# 短语相关路径
PHRASE_DIR=$TASK_DATA_DIR/phrase

####################用户需提供的数据#####################
# 种子实体文件
SEED_ENTITY_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_dict.txt
# 挖掘出的短语路径
PHRASE_PATH=$PHRASE_DIR/${TASK_NAME}_autophrase.txt
# 全量词向量文件
WORD_VEC_PATH=$ROOT_DIR/word2vec/wikipedia-pubmed-and-PMC-w2v.bin
# 当前任务词向量
PART_WORD_VEC_PATH=$FORMAT_DATA_DIR/part_word_vec.txt

# 标注实体类型
GOLD_ENTITY_PATH=$TASK_DATA_DIR/source_data/${TASK_NAME}_gold_entity.txt

####################短语类别打标数据#####################
PHRASE_LABEL_PATH=$PHRASE_DIR/knn_phrase_label.txt

# 日志
LABEL_LOG_FILE=phrase_label_log
nohup python -u run_model/run_phrase_label.py \
    --task_name=$TASK_NAME \
    --seed_entity_path=$SEED_ENTITY_PATH \
    --phrase_path=$PHRASE_PATH \
    --word_vec_path=$WORD_VEC_PATH \
    --gold_entity_path=$GOLD_ENTITY_PATH \
    --part_word_vec_path=$PART_WORD_VEC_PATH \
    --phrase_label_path=$PHRASE_LABEL_PATH \
    > $LABEL_LOG_FILE 2>&1 &
