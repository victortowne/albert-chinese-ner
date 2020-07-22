#!/usr/bin/env bash

export LANG="en_US.UTF-8"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
export Albert_Base_Dir=/home/work/qabot/tangshengjun/deep_learning/bert_model/pre_trained_models/albert/albert_base_zh

python albert_ner.py \
	--task_name ner \
	--do_train true \
	--do_eval true \
	--data_dir data \
	--vocab_file $Albert_Base_Dir/vocab.txt \
	--bert_config_file $Albert_Base_Dir/albert_config_base.json \
	--init_checkpoint $Albert_Base_Dir/albert_model.ckpt \
	--max_seq_length 128 \
	--train_batch_size 64 \
	--learning_rate 2e-5 \
	--num_train_epochs 3 \
	--output_dir ./output
