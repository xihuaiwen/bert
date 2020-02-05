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
parser.add_argument('--batch-size', type=int, default=1,
                            help="Set batch-size")
parser.add_argument('--embedding-size', type=int, default=32,
                            help="Size of source and target embedding")
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

"""
batch_size = 1
seq_length = 128
hidden_size = 768
vocab_size = 28996
hidden_size=768
num_hidden_layers=12
num_attention_heads=12
intermediate_size=3072
intermediate_act_fn=modeling.gelu
hidden_dropout_prob=0.1
attention_probs_dropout_prob=0.1
initializer_range=0.02
"""
intermediate_act_fn=modeling.gelu


def embedding_graph(input_ids):
    embedding_output,embedding_table = modeling.embedding_lookup(input_ids,args.vocab_size)
    embedding_output = modeling.embedding_postprocessor(input_tensor=embedding_output)
    return embedding_output,embedding_table

def attention(input_tensor,attention_mask):
    """
        input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size]
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
              seq_length], with 1 for positions that can be attended to and 0 in
                    positions that should not be.
        hidden_size: int. Hidden size of the Transformer.
    """
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
    return layer_output


with ipu_scope("/device:IPU:0"):
    input_ids = tf.placeholder(tf.int32,shape=(args.batch_size, args.seq_length),name='input_ids')
    input_tensor = tf.placeholder(tf.float32,shape=(args.batch_size, args.seq_length, args.hidden_size),name='input_tensor')
    attention_mask = tf.placeholder(tf.int32,shape=(args.batch_size, args.seq_length, args.seq_length),name='attention_mask')
    batch = ipu.ipu_compiler.compile(attention, [input_tensor,attention_mask])


opts = utils.create_ipu_config(profiling=args.profiling,profile_execution=args.profiling)
cfg = utils.auto_select_ipus(opts,1)
ipu.utils.configure_ipu_system(cfg)


input_t = np.random.rand(args.batch_size, args.seq_length).astype(np.int32)
_input_tensor = np.random.rand(args.batch_size, args.seq_length, args.hidden_size)
_attention_mask = np.random.rand(args.batch_size, args.seq_length, args.seq_length).astype(np.int32)

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
    layer_output = sess.run(batch,feed_dict = {input_tensor:_input_tensor,attention_mask:_attention_mask})
    print (layer_output)
    raw_reports = sess.run(report)
    save_tf_report(raw_reports)
