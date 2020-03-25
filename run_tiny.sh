
export BERT_BASE_DIR=/home/xihuaiw/code/test/xihuai/bert/uncased_L-2_H-128_A-2
export DATA_DIR=/home/xihuaiw/code/test/xihuai/bert/data/mrpc
export POPLAR_ENGINE_OPTIONS='{"debug.loweredVarDumpFile":"vars.capnp"}'
python run_classifier.py \
	  --task_name=MRPC \
	  --do_train=true \
	  --do_eval=true \
	  --data_dir=$DATA_DIR \
	  --vocab_file=$BERT_BASE_DIR/vocab.txt \
	  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
	  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	  --max_seq_length=128 \
	  --train_batch_size=16 \
	  --learning_rate=0.01 \
	  --num_train_epochs=5.0 \
	  --output_dir=./tmp/mrpc_output \
	  --use_ipu=True \
          --use_fp16=False \
	  --ipu_profiling=true
