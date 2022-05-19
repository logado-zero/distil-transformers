# distil-transformers

## **1. Reference**
* The solution in thís repository based on the architecture published in microsoft research : 

  [XtremeDistilTransformers: Task Transfer for Task-agnostic Distillation](https://arxiv.org/pdf/2106.04563.pdf)
  
  [XtremeDistil: Multi-stage Distillation for Massive Multilingual Models](https://arxiv.org/pdf/2004.05686.pdf)

* This repository contains solution of NER task based on PyTorch [reimplementation](https://github.com/huggingface/transformers) of 
[Microsoft's TensorFlow repository for **XtremeDistilTransformers**](https://github.com/microsoft/xtreme-distil-transformers)

* This repository use the following *task-agnostic pre-distilled checkpoints* from XtremeDistilTransformers for (only) fine-tuning on labeled data from downstream tasks:
  -  [6/256 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l6-h256-uncased)
  -  [6/384 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l6-h384-uncased)
  -  [12/384 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l12-h384-uncased)
   
## **2.Usage**
### **2.1. Prepare Dataset**
Dataset folder must have 3 files : ```train.tsv``` ```dev.tsv``` ```test.tsv``` ```labels.tsv``` ```transfer.tsv``` 

* ```train/dev/test '.tsv'``` files must be formatted in *.tsv* file with ```sep="\t"``` for token-wise tags data as follow:

-- Example: Cả nước Thái bàng hoàng . <tab> O O B-LOC O O O
* ```labels.tsv``` file containing class labels for sequence labeling, for example:
  ```
  B-ORG
  B-PER
  B-LOC
  ...
  ```
* ```transfer.tsv``` file containing unlabeled data


### **2.2. Train model**
  *Run ```PYTHONHASHSEED=42 python3 run_xtreme_distil.py``` follow this arguments*
```
  - task                        dataset folder which contains train/dev/test/label/transfer '.tsv'
  - model_dỉr                   path to store/restore student model checkpoints
  - do_NER                      set opion for sequence labeling
  - seq_len max                 length of each sequence
  - transfer_file               path to unlabeled data
  - pt_teacher                  for teacher model to distil (e.g., BertModel, RobertaModel, ElectraModel)
  - pt_teacher_checkpoint       for pre-trained teacher model checkpoints (e.g., bert-base-multilingual-cased, roberta-large)
  - teacher_model_dir           path to store/restore teacher model checkpoints
  - pt_student_checkpoint       to initialize from pre-trained small student models (e.g., MiniLM, DistilBert, TinyBert)
  - student_distil_batch_size   batch size data for distilation processing
  - student_ft_batch_size       batch size data for student model training with labeled data
  - teacher_batch_size          batch size data for teacher model training 
  - distil_chunk_size           for using transfer data in chunks dur ing distillation (reduce for OOM issues, checkpoints are saved after every distil_chunk_size steps)
  - distil_multi_hidden_states  to distil multiple hidden states from the teacher
  - distil_attention            to distil deep attention network of the teacher
  - compress_word_embedding     to initialize student word embedding with SVD-compressed teacher word embedding (useful for multilingual distillation)
  - freeze_word_embedding       to keep student word embeddings frozen during distillation (useful for multilingual distillation)
  - opt_policy                  (e.g., mixed_float16 for GPU and mixed_bfloat16 for TPU)
  - tf_model_check              if teacher model checkpoint is tensorflow model
```
  *Example*
```
  PYTHONHASHSEED=42 python3 run_xtreme_distil.py \
  --task /home/intern/Multi_NER/data_distil \
  --model_dir /home/intern/Multi_NER/model_distil_bert_12_384 \
  --seq_len 32  \
  --transfer_file /home/intern/Multi_NER/data_distil/tranfer.tsv \
  --do_NER \
  --pt_teacher BertModel \
  --pt_teacher_checkpoint bert-base-multilingual-cased \
  --student_distil_batch_size 32 \
  --student_ft_batch_size 16 \
  --teacher_batch_size 8  \
  --pt_student_checkpoint microsoft/xtremedistil-l12-h384-uncased \
  --distil_chunk_size 10000 \
  --teacher_model_dir /home/intern/Multi_NER/model_distil_bert \
  --distil_multi_hidden_states \
  --distil_attention \
  --compress_word_embedding \
  --freeze_word_embedding \
  --opt_policy mixed_float16 \
  --tf_model_check

``` 

