import numpy as np
import tensorflow as tf
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu
import modeling
import optimization
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from gcprofile import save_tf_report


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
parser.add_argument('--vocab-size', type=int, default=28996,
                            help="Size of vocab")
parser.add_argument('--steps', type=int, default=50000,
                            help="Number of steps to complete in training")
parser.add_argument('--num-hidden-layers', type=int, default=12,
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

args = parser.parse_args()
intermediate_act_fn=modeling.gelu

class Dataset:
    def __init__(self,num_items):
        self.num_items = num_items
        self.data = []
        for i in range(0,self.num_items):
            _input_tensor = np.random.rand(args.batch_size, args.seq_length, args.hidden_size)
            _attention_mask = np.random.randint(args.seq_length, size=(args.batch_size, args.seq_length, args.seq_length))
            _embedding_table = np.random.rand(args.vocab_size,args.hidden_size)
            _masked_lm_positions = np.random.randint(args.seq_length,size=(args.max_predictions_per_seq))
            _masked_lm_ids       = np.random.randint(args.seq_length,size=(args.max_predictions_per_seq))
            _masked_lm_weights   = np.random.rand(args.max_predictions_per_seq)
            _next_sentence_labels = np.random.randint(1000,size=1)
            self.data.append((_input_tensor,
                            _attention_mask,
                            _embedding_table,
                            _masked_lm_positions,
                            _masked_lm_ids,
                            _masked_lm_weights,
                            _next_sentence_labels))

    def __getitem__(self,index):
        return self.data[index]
    def __iter__(self):
        return (self[index] for index in range(self.num_items))
    def __len__(self):
        return self.num_items
        


def embedding_graph(input_ids):
    embedding_output,embedding_table = modeling.embedding_lookup(input_ids,args.vocab_size)
    embedding_output = modeling.embedding_postprocessor(input_tensor=embedding_output)
    return embedding_output,embedding_table

def attention(input_tensor,attention_mask,
            embedding_table,masked_lm_positions,
            masked_lm_ids, masked_lm_weights,next_sentence_labels):
    """
        input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size]
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
              seq_length], with 1 for positions that can be attended to and 0 in
                    positions that should not be.
        hidden_size: int. Hidden size of the Transformer.
    """
    input_shape = modeling.get_shape_list(input_tensor, expected_rank=3)

    layer_idx = 0
    attention_head_size = int(args.hidden_size / args.num_attention_heads)
    prev_output = modeling.reshape_to_matrix(input_tensor)
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


        final_output = modeling.reshape_from_matrix(prev_output, input_shape)
        #since in modeling.py the transformer_model return all layer output 
        # we can comment following line 
        #sequence_output = final_output[-1]
        sequence_output = final_output
    
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = tf.layers.dense(
            first_token_tensor,
            args.hidden_size,
            activation=tf.tanh,
            kernel_initializer=modeling.create_initializer(args.initializer_range))

     ### caculate the loss
    (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         sequence_output, embedding_table,
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
         pooled_output, next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    optimiser = tf.train.GradientDescentOptimizer(0.01)
    optimiser.minimize(total_loss)
        
    return layer_output,total_loss


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
        shape=[args.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=args.vocab_size, dtype=tf.float32)

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
        shape=[2, args.hidden_size],
        initializer=modeling.create_initializer(args.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
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

with ipu_scope("/device:IPU:0"):
    input_tensor = tf.placeholder(tf.float32,shape=(args.batch_size, args.seq_length, args.hidden_size),name='input_tensor')
    attention_mask = tf.placeholder(tf.int32,shape=(args.batch_size, args.seq_length, args.seq_length),name='attention_mask')
    embedding_table = tf.placeholder(tf.float32,shape=(args.vocab_size,args.hidden_size),name="embedding_table")
    masked_lm_positions = tf.placeholder(tf.int32,shape=(args.max_predictions_per_seq),name="masked_lm_positions") 
    masked_lm_ids = tf.placeholder(tf.int32,shape=(args.max_predictions_per_seq),name="masked_lm_ids")
    masked_lm_weights = tf.placeholder(tf.float32,shape=(args.max_predictions_per_seq),name="masked_lm_weights")
    next_sentence_labels = tf.placeholder(tf.int32,shape=(1),name="next_sentence_labels")
    batch = ipu.ipu_compiler.compile(attention, [input_tensor,
                                                attention_mask,
                                                embedding_table,
                                                masked_lm_positions,
                                                masked_lm_ids,
                                                masked_lm_weights,
                                                next_sentence_labels])


opts = utils.create_ipu_config(profiling=args.profiling,profile_execution=args.profiling)
cfg = utils.auto_select_ipus(opts,1)
ipu.utils.configure_ipu_system(cfg)

data = Dataset(10)

_next_sentence_labels = np.random.randint([1])
tvars = tf.trainable_variables()
tf.logging.info("**** Trainable Variables ****")
for var in tvars:
    init_string = ""
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

with tf.device('cpu'):
    report = gen_ipu_ops.ipu_event_trace()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0,args.epochs):
        j = 0
        for (_input_tensor,
            _attention_mask,
            _embedding_table,
            _masked_lm_positions,
            _masked_lm_ids,
            _masked_lm_weights,
            _next_sentence_labels)  in data:
            layer_output = sess.run(batch,feed_dict = {input_tensor:_input_tensor,
                                            attention_mask:_attention_mask,
                                            embedding_table:_embedding_table,
                                            masked_lm_positions:_masked_lm_positions,
                                            masked_lm_ids:_masked_lm_ids,
                                            masked_lm_weights:_masked_lm_weights,
                                            next_sentence_labels:_next_sentence_labels})
            print ("processing #%d" % (j))
            j+=1
    raw_reports = sess.run(report)
    save_tf_report(raw_reports)
