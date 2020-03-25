
#export BERT_BASE_DIR=/home/xihuaiw/code/bert/squad_sota-F16
#export BERT_BASE_DIR=/home/xihuaiw/code/bert/cased_L-12_H-768_A-12-F16
#export BERT_BASE_DIR=/home/xihuaiw/code/test/xihuai/bert/uncased_L-2_H-128_A-2
export BERT_BASE_DIR=/home/xihuaiw/code/test/xihuai/bert/sota-cpu_without_adam-F16
export SQUAD_DIR=/home/xihuaiw/code/bert/SQUAD_DIR
export POPLAR_ENGINE_OPTIONS='{"debug.loweredVarDumpFile":"vars.capnp"}'
python run_squad.py \
	--vocab_file=$BERT_BASE_DIR/vocab.txt \
	--bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	--do_train=True \
	--train_file=$SQUAD_DIR/train-v1.1.json \
	--do_predict=True \
	--predict_file=$SQUAD_DIR/dev-v1.1.json \
	--train_batch_size=1 \
	--learning_rate=1e-6 \
	--num_train_epochs=2.0 \
	--max_seq_length=384 \
	--doc_stride=128 \
	--output_dir=./tmp/squad_base/ \
	--do_lower_case=False	\
	--use_ipu=True \
	--use_fp16=True
