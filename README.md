# TEBNER

Code for EMNLP 2021 paper:"TEBNER: Domain Specific Named Entity Recognition with Type Expanded Boundary-aware Network"

## Required Inputs
- **Raw Texts**
  - Example: ```data/bc5cdr/source_data/bc5cdr_train.json```
- **Dictionary**
  - Example: ```data/bc5cdr/source_data/bc5cdr_dict.txt```
- **High-quality Phrase (Optional)**
  - Example: ```data/bc5cdr/phrase/bc5cdr_autophrase.txt```
    

## Dependencies
This project is based on ```python>=3.6```. The dependent package for this project is listed as below:
```
torch==1.5.1
transformers==3.3.1
numpy==1.19.2
scikit_learn==0.24.1
seqeval==1.2.2
gensim==3.8.3
```

## Training Command
1.To label data by source dictionary (e.g., bc5cdr_dict.txt)
```
add config in b_run_data_label.sh: --do_source_distance
run: sh b_run_data_label.sh
```

2.To train entity classification model
```
add config in b_run_mention_classify.sh: --do_train
run: sh b_run_mention_classify.sh
```

3.To extend the source dictionary
```
add config in b_run_mention_classify.sh: --do_predict
run: sh b_run_mention_classify.sh
```

4.Add new label entities (according to the extended dictionary)
```
add config in b_run_data_label.sh: --do_add_distance
run: sh b_run_data_label.sh
```

5.To train bert word boundary model
```
sh c_run_bert_word.sh
```

6.To train bert sent boundary model
```
sh c_run_bert_sent.sh
```

7.To combine final result
```
sh d_run_bert_pipeline_model.sh
```

**Attention**: you need to assign data dir in each script file. (e.g., "ROOT_DIR" in b_run_data_label.sh)


