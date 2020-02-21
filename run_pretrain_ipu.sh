export BERT_BASE_DIR=/home/chengshunx/bert/cased_L-12_H-768_A-12
python run_pretraining.py \
  --input_file=/home/chengshunx/bert/tf_examples.tfrecord \
  --output_dir=/home/chengshunx/bert/pretraining_output \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=1 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5	\
  --use_ipu=True
