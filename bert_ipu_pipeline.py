import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils, scopes
from tensorflow.python import ipu
from tensorflow.python.ipu import loops
import modeling
import optimization
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from gcprofile import save_tf_report
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.training import gradient_descent
import tensorflow.contrib.slim as slim
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue

import argparse

parser = argparse.ArgumentParser(description='NMT model in TensorFlow to run on the IPU')
parser.add_argument('--epochs', type=int, default=1,
                            help="batch-size")
parser.add_argument('--batch-size', type=int, default=1,
                            help="Set batch-size")
parser.add_argument('--learning-rate', type=float, default=5e-5,
                            help="the initial learning rate")
parser.add_argument('--seq-length', type=int, default=128,
                            help="Size of input length (by padding or truncating)")
parser.add_argument('--vocab-size', type=int, default=3000,
                            help="Size of vocab")
parser.add_argument('--steps', type=int, default=50000,
                            help="Number of steps to complete in training")
parser.add_argument('--num-hidden-layers', type=int, default=6,
                            help="Number of hidden layers")
parser.add_argument('--num-attention-heads', type=int, default=12,
                            help="Number of attention heads")
parser.add_argument('--max_predictions_per_seq', type=int, default=20,
                            help="Maximum number of masked LM predictions per sequence.")
parser.add_argument('--hidden-size', type=int, default=768,
                            help="Number of hidden size")
parser.add_argument('--intermediate-size', type=int, default=3072,
                            help="size of intermediate")
parser.add_argument('--hidden-dropout-prob', type=float, default=0.1,
                            help="dropout prob of hidden layer")
parser.add_argument('--attention-probs-dropout-prob', type=float, default=0.1,
                            help="dropout prob of attention layer ")
parser.add_argument('--initializer-range', type=float, default=0.02,
                            help="propotion of initializer range")
parser.add_argument('--profiling', type=bool, default=True,
                            help="Whether to enable profiling")
parser.add_argument('--poplar-text-report', type=bool, default=True,
                                    help="Whether to save poplar text report rather than in Json")

args = parser.parse_args()
intermediate_act_fn=modeling.gelu

def generate_tf_dataset():
    _input_ids = np.random.randint(args.vocab_size, size=(args.batch_size, args.seq_length)).astype(np.int32)
    _input_mask = np.random.randint(args.vocab_size, size=(args.batch_size, args.seq_length)).astype(np.int32)
    _token_type_ids = np.random.randint(args.vocab_size, size=(args.batch_size, args.seq_length)).astype(np.int32)
    _masked_lm_positions = np.random.randint(args.seq_length,size=(args.batch_size,args.max_predictions_per_seq)).astype(np.int32)
    _masked_lm_ids       = np.random.randint(args.seq_length,size=(args.batch_size,args.max_predictions_per_seq)).astype(np.int32)
    _masked_lm_weights   = np.random.rand(args.batch_size,args.max_predictions_per_seq).astype(np.float32)
    _next_sentence_labels = np.random.randint(1000,size=1).astype(np.int32)
    ds = tf.data.Dataset.from_tensors((_input_ids,
                                      _input_mask,
                                      _token_type_ids,
                                      _masked_lm_positions,
                                      _masked_lm_ids,
                                      _masked_lm_weights,
                                      _next_sentence_labels))
    ds = ds.repeat()
    iterator = ds.make_initializable_iterator()
    return ds,iterator

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def calculate_required_ipu():
    num_shards = int (args.num_hidden_layers) + 2
    i = 0
    num_ipus_list = [2,4,8,16]
    for nums in num_ipus_list:
        if nums >= num_shards:
            return nums
    print ("Cannot meet the IPU resource allocation request, you required %d" % (num_shards))
    sys.exit(1)

def words_embedding_lookup(input_ids,input_mask,token_type_ids,
            masked_lm_positions,masked_lm_ids,
            masked_lm_weights,next_sentence_labels):
    embedding_output,embedding_table = modeling.embedding_lookup(
                                         input_ids = input_ids,
                                         vocab_size = args.vocab_size,
                                         embedding_size = args.hidden_size,
                                         initializer_range = args.initializer_range,
                                         word_embedding_name = "word_embeddings",
                                         use_one_hot_embeddings = False
                                         )
def positional_segment_embedding_lookup():
  pass
def attention_ff():
  pass
def mlm_loss_calc():
  pass
def nsp_loss_calc():
  pass

ds,itr = generate_tf_dataset()
infeed = ipu_infeed_queue.IPUInfeedQueue(ds,feed_name="infeed")
outfeed = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

def bert_model(input_ids,input_mask,token_type_ids,
            masked_lm_positions,masked_lm_ids,
            masked_lm_weights,next_sentence_labels):
  with scopes.ipu_shard(0):
    embedding_output,embedding_table = modeling.embedding_lookup(
                                         input_ids = input_ids,
                                         vocab_size = args.vocab_size,
                                         embedding_size = args.hidden_size,
                                         initializer_range = args.initializer_range,
                                         word_embedding_name = "word_embeddings",
                                         use_one_hot_embeddings = False
                                         )
  with scopes.ipu_shard(1):
    embedding_output = modeling.embedding_postprocessor(
                         input_tensor = embedding_output,
                         use_token_type=True,
                         token_type_ids=token_type_ids, 
                         token_type_vocab_size=2,
                         token_type_embedding_name="token_type_embeddings",
                         use_position_embeddings=True,
                         position_embedding_name="position_embeddings",
                         initializer_range=args.initializer_range,
                         max_position_embeddings=512,
                         dropout_prob=args.hidden_dropout_prob)

    attention_mask = modeling.create_attention_mask_from_input_mask(
                            input_ids, input_mask)

    if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                            "The hidden size (%d) is not a multiple of the number of attention "
                                    "heads (%d)" % (hidden_size, num_attention_heads))
    
    attention_head_size = int(args.hidden_size /args.num_attention_heads)
    input_shape = modeling.get_shape_list(embedding_output, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != args.hidden_size:
      raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
        (input_width, args.hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
                              # help the optimizer.
    prev_output = modeling.reshape_to_matrix(embedding_output)
    #embedding_output = tf.stop_gradient(embedding_output)

  for layer_idx in range(args.num_hidden_layers):
    """
        currently put one attention layer onto one IPU
        embedding on IPU subgraph #0
        loss caculation on IPU subgraph #1
    """
    with scopes.ipu_shard(int(layer_idx) + 2):
      with tf.variable_scope("layer_%d" % layer_idx):
        layer_input = prev_output
  
        with tf.variable_scope("attention"):
          attention_heads = []
          with tf.variable_scope("self"):
            attention_head = modeling.attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=args.num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                initializer_range=args.initializer_range,
                do_return_2d_tensor=True,
                batch_size=args.batch_size,
                from_seq_length=args.seq_length,
                to_seq_length=args.seq_length)
            attention_heads.append(attention_head)
  
          attention_output = None
          if len(attention_heads) == 1:
            attention_output = attention_heads[0]
          else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)
  
          # Run a linear projection of `hidden_size` then add a residual
          # with `layer_input`.
          with tf.variable_scope("output"):
            attention_output = tf.layers.dense(
                attention_output,
                args.hidden_size,
                kernel_initializer=modeling.create_initializer(args.initializer_range))
            attention_output = modeling.dropout(attention_output, args.hidden_dropout_prob)
            attention_output = modeling.layer_norm(attention_output + layer_input)
  
        # The activation is only applied to the "intermediate" hidden layer.
        with tf.variable_scope("intermediate"):
          intermediate_output = tf.layers.dense(
              attention_output,
              args.intermediate_size,
              activation=intermediate_act_fn,
              kernel_initializer=modeling.create_initializer(args.initializer_range))
        # Down-project back to `hidden_size` then add the residual.
        with tf.variable_scope("output"):
          layer_output = tf.layers.dense(
              intermediate_output,
              args.hidden_size,
              kernel_initializer=modeling.create_initializer(args.initializer_range))
          layer_output = modeling.dropout(layer_output, args.hidden_dropout_prob)
          layer_output = modeling.layer_norm(layer_output + attention_output)
          prev_output = layer_output
	
        if layer_idx == args.num_hidden_layers - 1:
          final_output = modeling.reshape_from_matrix(prev_output, input_shape)
          #since in modeling.py the transformer_model return all layer output 
          # we can comment following line 
          #sequence_output = final_output[-1]
          sequence_output = final_output

  with scopes.ipu_shard(0):
      (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
          sequence_output, embedding_table,
          masked_lm_positions, masked_lm_ids, masked_lm_weights)

  with scopes.ipu_shard(1):
      with tf.variable_scope("pooler"):
      # We "pool" the model by simply taking the hidden state corresponding
      # to the first token. We assume that this has been pre-trained
          first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
         # pooled_output = tf.stop_gradient (tf.layers.dense(
          pooled_output = tf.layers.dense(
              first_token_tensor,
              args.hidden_size,
              activation=tf.tanh,
          #    kernel_initializer=modeling.create_initializer(args.initializer_range)))
              kernel_initializer=modeling.create_initializer(args.initializer_range))

      ### caculate the loss
      (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
          pooled_output, next_sentence_labels)

      total_loss = masked_lm_loss + next_sentence_loss

  opt = sharded_optimizer.ShardedOptimizer(
                            gradient_descent.GradientDescentOptimizer(args.learning_rate))
  train_op = opt.minimize(total_loss)
        
  return outfeed.enqueue(total_loss),train_op

def get_masked_lm_output(input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=args.hidden_size,
          activation=modeling.get_activation("gelu"),
          kernel_initializer=modeling.create_initializer(
              args.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        dtype=tf.float16,
        shape=[args.vocab_size],
        initializer=tf.zeros_initializer())
    #input_tensor = tf.stop_gradient(input_tensor)
    #output_weights = tf.stop_gradient(output_weights)
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])
    label_weights = tf.cast(label_weights,dtype=tf.float16)

    one_hot_labels = tf.one_hot(
        label_ids, depth=args.vocab_size, dtype=tf.float16)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        dtype=tf.float16,
        shape=[2, args.hidden_size],
        initializer=modeling.create_initializer(args.initializer_range))
    output_bias = tf.get_variable(
        "output_bias",
        dtype=tf.float16,
        shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float16)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [args.batch_size * args.seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def bert_loop():
  return loops.repeat(100,bert_model,infeed_queue=infeed)

with ipu_scope("/device:IPU:0"):
    input_ids = tf.placeholder(tf.int32,shape=(args.batch_size, args.seq_length),name='input_ids')
    input_mask = tf.placeholder(tf.int32,shape=(args.batch_size, args.seq_length),name='input_mask')
    token_type_ids = tf.placeholder(tf.int32,shape=(args.batch_size, args.seq_length),name='token_type_ids')

    masked_lm_positions = tf.placeholder(tf.int32,shape=(args.batch_size,args.max_predictions_per_seq),name="masked_lm_positions")
    masked_lm_ids = tf.placeholder(tf.int32,shape=(args.batch_size,args.max_predictions_per_seq),name="masked_lm_ids")
    masked_lm_weights = tf.placeholder(tf.float32,shape=(args.batch_size,args.max_predictions_per_seq),name="masked_lm_weights")
    next_sentence_labels = tf.placeholder(tf.int32,shape=(1),name="next_sentence_labels")
    batch = ipu.ipu_compiler.compile(bert_loop)


_total_loss = outfeed.dequeue()
opts = utils.create_ipu_config(profiling=args.profiling,profile_execution=args.profiling,use_poplar_text_report=args.poplar_text_report)
opts = utils.set_convolution_options(opts, {"availableMemoryProportion": "0,23"})
opts = utils.set_matmul_options(opts,{"availableMemoryProportion": "0.23"})
utils.set_recomputation_options(opts, allow_recompute=True)
cfg = utils.auto_select_ipus(opts,calculate_required_ipu())
ipu.utils.configure_ipu_system(cfg)

tvars = tf.trainable_variables()
tf.logging.info("**** Trainable Variables ****")
for var in tvars:
    init_string = ""
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

with tf.device('cpu'):
    report = gen_ipu_ops.ipu_event_trace()



with tf.Session() as sess:
    model_summary()
    sess.run(tf.global_variables_initializer())
    sess.run(infeed.initializer)
    for i in range(0,args.epochs):
        sess.run(batch)
        result = sess.run(_total_loss)
        print (result)
    raw_reports = sess.run(report)
    if args.poplar_text_report:
        rep = ipu.utils.extract_all_strings_from_event_trace(raw_reports)
        with open("test_report.txt","w") as f:
            f.write(rep)
    else:
        save_tf_report(raw_reports)
