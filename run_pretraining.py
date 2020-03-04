# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import modeling
import optimization
import math
import time
import numpy as np

from collections import namedtuple,OrderedDict
import tensorflow as tf
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils, scopes
from tensorflow.python import ipu

from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu import loops, ipu_infeed_queue, ipu_outfeed_queue, ipu_compiler

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer("epochs", 1, "How many epoches for training.")

flags.DEFINE_integer("layers_per_ipu", 2 , "attention layers per IPU")

flags.DEFINE_bool("ipu_profile", False, "Whether to enable Graphcore IPU event tracing/profiling.")

flags.DEFINE_integer("ipu_log_interval", 100, "Interval at which to log progress.")

flags.DEFINE_integer("ipu_summary_interval", 1, "Interval at which to write summaries.")

GraphOps = namedtuple(
     'graphOps', ['graph',
                  'session',
                  'init',
                  'ops',
#                  'placeholders',
                  'iterator',
                  'outfeed',
                  'saver'])

def calculate_required_ipus(num_hidden_layers):
    num_shards = int (math.ceil(num_hidden_layers/2.0)) + 2
    i = 0
    num_ipus_list = [2,4,8,16]
    for nums in num_ipus_list:
        if nums >= num_shards:
            return nums
    print ("Cannot meet the IPU resource allocation request, you required %d" % (num_shards))
    sys.exit(1)

def training_process(bert_config,FLAGS):

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    total_samples = get_dataset_files_count(FLAGS.input_file)
    logs_per_epoch = 10000
    iterations_per_epoch = total_samples / FLAGS.train_batch_size
    iterations = FLAGS.epochs * iterations_per_epoch
    log_freq = iterations_per_epoch // logs_per_epoch
    iterations_per_step = 10

    # -------------- BUILD TRAINING GRAPH ----------------                    
    train = training_graph(
            bert_config=bert_config,
            input_files=FLAGS.input_file,
            learning_rate=FLAGS.learning_rate,
            iterations_per_step=10,
            )
    train.session.run(train.init)
    train.session.run(train.iterator.initializer)
    # -------------- BUILD VALIDATION GRAPH ----------------
    # -------------- SAVE AND RESTORE --------------

    print_format = ("epochs: {epoch_idx:6d}, total_loss: {total_loss_mean:6.5f}, mlm_loss: {mlm_loss_mean:6.5f}, nsp_loss: {nsp_loss_mean:6.3f}, samples/s: {avg_samples_per_sec} ")
    i = 0
    batch_times = []
    epoch_idx = 1
    while i < 100000:
        
        train.session.run(train.ops)
        start_time = time.time()
        total_loss,mlm_loss,nsp_loss = train.session.run(train.outfeed)
        batch_time = time.time() - start_time
        batch_time /= iterations_per_step 
        if i != 0:
          batch_times.append(batch_time)

        log_this_step = True 

        if log_this_step:
          if len(batch_times) != 0:
            avg_batch_time = np.mean(batch_times)
          else:
            avg_batch_time = batch_time
          avg_samples_per_sec = FLAGS.train_batch_size / avg_batch_time
          total_loss_mean = np.mean(total_loss)
          mlm_loss_mean  = np.mean(mlm_loss)
          nsp_loss_mean = np.mean(nsp_loss) 
          stats =OrderedDict([
            ("epoch_idx",epoch_idx),
            ("total_loss_mean",total_loss_mean),
            ("mlm_loss_mean",mlm_loss_mean),
            ("nsp_loss_mean",nsp_loss_mean),
            ("avg_samples_per_sec",avg_samples_per_sec)
          ])
          tf.logging.info(print_format.format(**stats))

        i += iterations_per_step

def training_graph(bert_config,input_files,learning_rate,iterations_per_step):
    train_graph = tf.Graph()
    with train_graph.as_default():
        ds = get_pretraining_dataset(FLAGS.train_batch_size,
                        FLAGS.input_file,
                        FLAGS.max_seq_length,
                        FLAGS.max_predictions_per_seq,
                        is_training=True)
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds, feed_name="infeed")
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

        with ipu_scope('/device:IPU:0'):
            train = training_step_with_infeeds_and_outfeeds(infeed_queue, outfeed_queue,
                    bert_config, learning_rate, iterations_per_step)
        outfeed = outfeed_queue.dequeue()
#        logging.print_trainable_variables(opts)
        train_saver = tf.train.Saver(max_to_keep=999999)
        ipu.utils.move_variable_initialization_to_cpu()

        train_init = tf.global_variables_initializer()

    config = utils.create_ipu_config()
    config = utils.auto_select_ipus(config,8)

    config = utils.set_compilation_options(config, {
        "device.clearAtomicFlagAfterExchange": "false",
        "prng.enable": "true",
        "target.deterministicWorkers": "false" ,
        })
    config = utils.set_convolution_options(config, {"availableMemoryProportion": "0,23"})
    config = utils.set_matmul_options(config,{"availableMemoryProportion": "0.23"})
    config = utils.set_recomputation_options(config, allow_recompute=True)
    config = utils.set_floating_point_behaviour_options(config , nanoo=True)
    ipu.utils.configure_ipu_system(config)

    train_sess = tf.Session(graph=train_graph, config=tf.ConfigProto())
    return GraphOps(train_graph, train_sess, train_init, [train], infeed_queue, outfeed, train_saver)

def training_step_with_infeeds_and_outfeeds (infeed, outfeed, bert_config,learning_rate,iterations_per_step):
  def training_step(input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels):  # pylint: disable=unused-argument
    """
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]
    """

    with scopes.ipu_shard(0):
      embedding_output,embedding_table = modeling.embedding_lookup(
                                         input_ids = input_ids,
                                         vocab_size = bert_config.vocab_size,
                                         embedding_size = bert_config.hidden_size,
                                         initializer_range = bert_config.initializer_range,
                                         word_embedding_name = "word_embeddings",
                                         use_one_hot_embeddings = False
                                         )
    with scopes.ipu_shard(1):
      embedding_output = modeling.embedding_postprocessor(
                         input_tensor = embedding_output,
                         use_token_type=True,
                         token_type_ids=segment_ids, 
                         token_type_vocab_size=2,
                         token_type_embedding_name="token_type_embeddings",
                         use_position_embeddings=True,
                         position_embedding_name="position_embeddings",
                         initializer_range=bert_config.initializer_range,
                         max_position_embeddings=512,
                         dropout_prob=bert_config.hidden_dropout_prob)

      attention_mask = modeling.create_attention_mask_from_input_mask(
                            input_ids, input_mask)

      if bert_config.hidden_size % bert_config.num_attention_heads != 0:
            raise ValueError(
                            "The hidden size (%d) is not a multiple of the number of attention "
                                    "heads (%d)" % (hidden_size, num_attention_heads))
    
      attention_head_size = int(bert_config.hidden_size /bert_config.num_attention_heads)
      input_shape = modeling.get_shape_list(embedding_output, expected_rank=3)
      batch_size = input_shape[0]
      seq_length = input_shape[1]
      input_width = input_shape[2]

      # The Transformer performs sum residuals on all layers so the input needs
      # to be the same as the hidden size.
      if input_width != bert_config.hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
        (input_width, bert_config.hidden_size))

      # We keep the representation as a 2D tensor to avoid re-shaping it back and
      # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
      # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
                              # help the optimizer.
      prev_output = modeling.reshape_to_matrix(embedding_output)
      #embedding_output = tf.stop_gradient(embedding_output)

    for layer_idx in range(bert_config.num_hidden_layers):
      #  currently put one attention layer onto one IPU
      #  embedding on IPU subgraph #0
      #  loss caculation on IPU subgraph #1

      with scopes.ipu_shard(int(math.floor(layer_idx/2.0)) + 2):
        with tf.variable_scope("layer_%d" % layer_idx):
          layer_input = prev_output
  
          with tf.variable_scope("attention"):
            attention_heads = []
            with tf.variable_scope("self"):
              attention_head = modeling.attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=bert_config.num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                initializer_range=bert_config.initializer_range,
                do_return_2d_tensor=True,
                batch_size=batch_size,
                from_seq_length=seq_length,
                to_seq_length=seq_length)
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
                bert_config.hidden_size,
                kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
              attention_output = modeling.dropout(attention_output, bert_config.hidden_dropout_prob)
              attention_output = modeling.layer_norm(attention_output + layer_input)
  
          # The activation is only applied to the "intermediate" hidden layer.
          with tf.variable_scope("intermediate"):
            intermediate_output = tf.layers.dense(
              attention_output,
              bert_config.intermediate_size,
              activation=modeling.gelu,
              kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
          # Down-project back to `hidden_size` then add the residual.
          with tf.variable_scope("output"):
            layer_output = tf.layers.dense(
              intermediate_output,
              bert_config.hidden_size,
              kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
            layer_output = modeling.dropout(layer_output, bert_config.hidden_dropout_prob)
            layer_output = modeling.layer_norm(layer_output + attention_output)
            prev_output = layer_output
	
        if layer_idx == bert_config.num_hidden_layers - 1:
          final_output = modeling.reshape_from_matrix(prev_output, input_shape)
          #since in modeling.py the transformer_model return all layer output 
          # we can comment following line 
          #sequence_output = final_output[-1]
          sequence_output = final_output

    with scopes.ipu_shard(0):
      (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
          bert_config,sequence_output, embedding_table,
          masked_lm_positions, masked_lm_ids, masked_lm_weights)

    with scopes.ipu_shard(1):
      with tf.variable_scope("pooler"):
      # We "pool" the model by simply taking the hidden state corresponding
      # to the first token. We assume that this has been pre-trained
          first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
         # pooled_output = tf.stop_gradient (tf.layers.dense(
          pooled_output = tf.layers.dense(
              first_token_tensor,
              bert_config.hidden_size,
              activation=tf.tanh,
          #    kernel_initializer=modeling.create_initializer(args.initializer_range)))
              kernel_initializer=modeling.create_initializer(bert_config.initializer_range))

      ### caculate the loss
      (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
          bert_config,pooled_output, next_sentence_labels)

      total_loss = masked_lm_loss + next_sentence_loss

    opt = sharded_optimizer.ShardedOptimizer(
                            gradient_descent.GradientDescentOptimizer(learning_rate))
    train_op = opt.minimize(total_loss)
    out = outfeed.enqueue((total_loss,masked_lm_loss,next_sentence_loss))
        
    return out,train_op
    
  def compiled_fn():
    return loops.repeat(iterations_per_step,
                training_step,
                [],
                infeed)
  return ipu_compiler.compile(compiled_fn, [])

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        dtype=tf.float16,
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])
    label_weights = tf.cast(label_weights,dtype=tf.float16)

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float16)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)

def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        dtype=tf.float16,
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", 
        dtype=tf.float16,
        shape=[2], 
        initializer=tf.zeros_initializer())

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
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def get_dataset_files_count(input_file):
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  total_samples = 0
  for input_file in input_files:
    total_samples += sum(1 for _ in tf.python_io.tf_record_iterator(input_file))
  return total_samples

def get_pretraining_dataset(batch_size,input_file,max_seq_length,max_predictions_per_seq,is_training,num_cpu_threads=4):
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  name_to_features = {
      "input_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "masked_lm_positions":
          tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_ids":
          tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_weights":
          tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
      "next_sentence_labels":
          tf.FixedLenFeature([1], tf.int64),
  }

  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  if is_training:
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files))

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(num_cpu_threads, len(input_files))

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    d = d.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=is_training,
            cycle_length=cycle_length))
    d = d.shuffle(buffer_size=100)
  else:
    d = tf.data.TFRecordDataset(input_files)
    # Since we evaluate for a fixed number of steps we don't want to encounter
    # out-of-range exceptions.
    d = d.repeat()

  # We must `drop_remainder` on training because the TPU requires fixed
  # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
  # and we *don't* want to drop the remainder, otherwise we wont cover
  # every sample.
  d = d.apply(
      tf.contrib.data.map_and_batch(
          lambda record: _decode_record(record, name_to_features),
          batch_size=batch_size,
          num_parallel_batches=num_cpu_threads,
          drop_remainder=True))
  return d

def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  if FLAGS.do_train:
    training_process(bert_config,FLAGS)

  if FLAGS.do_eval:
    pass
    

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
