
# 当前任务领域
TASK_NAME="bc5cdr"

# 所有数据的根路径
SOURCE_DIR="/data/fangzheng/bert_autoner"
# 当前任务根路径
TASK_DIR=$SOURCE_DIR/$TASK_NAME

# 短语挖掘相关目录
PHRASE_DIR=$TASK_DIR/phrase
PHRASE_INTER_DATA_DIR=$PHRASE_DIR/inter_data
PHRASE_MODEL_DIR=$PHRASE_DIR/model
PHRASE_RESULT_DIR=$PHRASE_DIR/result
# 文本格式化目录
FORMAT_DIR=$TASK_DIR/format
echo ${green}=== Mkdir ===${reset}
mkdir -p $FORMAT_DIR
mkdir -p $PHRASE_DIR
mkdir -p $PHRASE_INTER_DATA_DIR
mkdir -p $PHRASE_MODEL_DIR
mkdir -p $PHRASE_RESULT_DIR

# 训练PhraseMining模型所需数据
PHRASE_TRAIN_ENTITY_PATH=$SOURCE_DIR/common_data/final_dict.txt
PHRASE_TRAIN_RAW_TEXT_PATH=$SOURCE_DIR/pubmed/pubmed_phrase_sent

###########任务相关数据###########
# 待挖掘的自由文本数据
SOURCE_DATA_PATH=$TASK_DIR/source_data/all.json
#SOURCE_DATA_PATH=$FORMAT_DIR/text_format
# 种子实体文件
SEED_ENTITY_PATH=$TASK_DIR/source_data/${TASK_NAME}_dict.txt

# 词向量文件
WORD_VEC_PATH=$SOURCE_DIR/word2vec/wikipedia-pubmed-and-PMC-w2v.bin
# 停用词
STOPWORD_PATH=$SOURCE_DIR/common_data/stopwords-en.txt
# 标点符号
SYMBOL_PATH=$SOURCE_DIR/common_data/symbol.txt
# 词性标签
POS_LABEL_PATH=$SOURCE_DIR/common_data/pos_label.txt

###########模型参数###########
# 短语最大长度(包含词语数量)
MAX_PHRASE_LEN=5
# 候选短语出现的最小频次
MIN_PHRASE_FREQ=2
# 基分类器个数
BASE_MODEL_NUM=10
# 训练集中负例数与正例数之比
NEG_POS_TIMES=8
# 中心词出现的最小频次
HEADWORD_MIN_FREQ=5

###########此步骤生成的文件###########
# 短语模型路径
PHRASE_MODEL_PATH=$PHRASE_MODEL_DIR/rf_clf_${TASK_NAME}
# 挖掘出的短语路径
PHRASE_PRED_RESULT_PATH=$PHRASE_RESULT_DIR/phrase_score_result
# 从高质量短语中选取的候选实体文件路径
CANDIDATE_PHRASE_ENTITY_PATH=$PHRASE_RESULT_DIR/candidate_phrase_entity
# 格式化自由文本数据
TEXT_FORMAT_PATH=$FORMAT_DIR/text_format

# 日志
LOG_FILE="phrase_test"

# do_write_inter_result 开关变量 表示是否将中间结果写入文本中
nohup python -u run_model/run_phrase_mining.py \
  --do_test \
  --task_name=$TASK_NAME \
  --phrase_train_entity_path=$PHRASE_TRAIN_ENTITY_PATH \
  --phrase_train_raw_text_path=$PHRASE_TRAIN_RAW_TEXT_PATH \
  --source_data_path=$SOURCE_DATA_PATH \
  --seed_entity_path=$SEED_ENTITY_PATH \
  --word_vec_path=$WORD_VEC_PATH \
  --symbol_path=$SYMBOL_PATH \
  --stopword_path=$STOPWORD_PATH \
  --pos_label_path=$POS_LABEL_PATH \
  --candidate_phrase_entity_path=$CANDIDATE_PHRASE_ENTITY_PATH \
  --text_format_path=$TEXT_FORMAT_PATH \
  --max_phrase_len=$MAX_PHRASE_LEN \
  --min_phrase_freq=$MIN_PHRASE_FREQ \
  --base_model_num=$BASE_MODEL_NUM \
  --neg_pos_times=$NEG_POS_TIMES \
  --inter_data_dir=$PHRASE_INTER_DATA_DIR \
  --model_path=$PHRASE_MODEL_PATH \
  --pred_result_path=$PHRASE_PRED_RESULT_PATH \
  > $LOG_FILE 2>&1 &