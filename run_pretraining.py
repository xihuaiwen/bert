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
import tensorflow as tf
import math
import pdb

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils, scopes
from tensorflow.python import ipu

from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.ipu_pipeline_estimator import IPUPipelineEstimatorSpec, IPUPipelineEstimator
from tensorflow.python.ipu.ops import pipelining_ops

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

flags.DEFINE_integer("num_train_steps", 1000000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_ipu", False, "Whether to user Graphcore IPU.")

flags.DEFINE_integer("layers_per_ipu", 2 , "attention layers per IPU")

flags.DEFINE_bool("ipu_profile", False, "Whether to enable Graphcore IPU event tracing/profiling.")

flags.DEFINE_integer("ipu_log_interval", 100, "Interval at which to log progress.")

flags.DEFINE_integer("ipu_summary_interval", 1, "Interval at which to write summaries.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

def calculate_required_ipus(num_hidden_layers):
    num_shards = num_hidden_layers + 2
    i = 0
    num_ipus_list = [2,4,8,16]
    for nums in num_ipus_list:
        if nums >= num_shards:
            return nums
    print ("Cannot meet the IPU resource allocation request, you required %d" % (num_shards))
    sys.exit(1)

def create_estimator(FLAGS, model_fn, bert_config):
    if FLAGS.use_ipu: 
        return create_ipu_estimator(FLAGS, model_fn, bert_config)
    else:
        return create_tpu_estimator(FLAGS, model_fn,)

def create_tpu_estimator(FLAGS,model_fn):
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))
    return tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

def create_ipu_estimator(FLAGS, model_fn, bert_config):
    ipu_options = ipu.utils.create_ipu_config(
        profiling=FLAGS.ipu_profile,
        use_poplar_text_report=FLAGS.ipu_profile,
        profile_execution=FLAGS.ipu_profile
    )

    required_ipus = calculate_required_ipus(bert_config.num_hidden_layers)

    ipu_options = ipu.utils.set_convolution_options(ipu_options, {"availableMemoryProportion": "0,23"})
    ipu_options = ipu.utils.set_matmul_options(ipu_options,{"availableMemoryProportion": "0.23"})
    ipu.utils.set_recomputation_options(ipu_options, allow_recompute=True)
    
    cfg = ipu.utils.auto_select_ipus(ipu_options, num_ipus=required_ipus)

    ipu_run_config = ipu.ipu_run_config.IPURunConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        ipu_options=ipu_options,
        num_shards=required_ipus,
    )

    config = ipu.ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config,
        log_step_count_steps=FLAGS.ipu_log_interval,
        save_summary_steps=FLAGS.ipu_summary_interval,
        model_dir=FLAGS.output_dir,
    )

    return IPUPipelineEstimator(
        config=config,
        model_fn=model_fn,
        params={"learning_rate": FLAGS.learning_rate,
                "batch_size":FLAGS.train_batch_size},
    )

def model_fn_builder_ipu(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
  """Returns `model_fn` closure for IPUEstimator."""
  def model_fn(mode, params):
    def words_embedding_lookup(input_ids,input_mask,segment_ids,
            masked_lm_positions,masked_lm_ids,
            masked_lm_weights,next_sentence_labels):
      embedding_output,embedding_table = modeling.embedding_lookup(
                                         input_ids = input_ids,
                                         vocab_size = bert_config.vocab_size,
                                         embedding_size = bert_config.hidden_size,
                                         initializer_range = bert_config.initializer_range,
                                         word_embedding_name = "word_embeddings",
                                         use_one_hot_embeddings = False
                                         )
      return embedding_output,embedding_table,input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels
    
    def positional_segment_embedding_lookup(embedding_output,embedding_table,input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels):
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
      return prev_output,attention_mask,embedding_table,embedding_output,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels

    def attention_ff(prev_output,attention_mask,embedding_table,embedding_output,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels):
        layer_input = prev_output
  
        attention_head_size = int(bert_config.hidden_size /bert_config.num_attention_heads)
        input_shape = modeling.get_shape_list(embedding_output, expected_rank=3)
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
                batch_size=FLAGS.train_batch_size,
                from_seq_length=FLAGS.max_seq_length,
                to_seq_length=FLAGS.max_seq_length)
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
	
        #if layer_idx == args.num_hidden_layers - 1:
        final_output = modeling.reshape_from_matrix(prev_output, input_shape)
          #since in modeling.py the transformer_model return all layer output 
          # we can comment following line 
          #sequence_output = final_output[-1]
        sequence_output = final_output
        return sequence_output,embedding_table,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels

    def mlm_loss_calc(sequence_output,embedding_table,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels):
      (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
          bert_config,sequence_output, embedding_table,
          masked_lm_positions, masked_lm_ids, masked_lm_weights)
      return masked_lm_loss,sequence_output,next_sentence_labels

    def nsp_loss_calc(masked_lm_loss,sequence_output,next_sentence_labels):
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
      return total_loss
    
    def optimizer_function(total_loss):
        opt = gradient_descent.GradientDescentOptimizer(FLAGS.learning_rate)
        return pipelining_ops.OptimizerFunctionOutput(opt, total_loss)   
  
    return IPUPipelineEstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                                computational_stages=[words_embedding_lookup, positional_segment_embedding_lookup,attention_ff,mlm_loss_calc,nsp_loss_calc],
                                pipeline_depth=4,
                                optimizer_function=optimizer_function,
                                device_mapping=[0,1,2,3,4]
                                )
  return model_fn
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

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

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


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


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

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

  return input_fn


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

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  if FLAGS.use_ipu:
    model_fn = model_fn_builder_ipu(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps)
  else:
    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = create_estimator(FLAGS,model_fn,bert_config)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
