import numpy as np
import tensorflow as tf
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu
import modeling
import argparse
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from gcprofile import save_tf_report


parser = argparse.ArgumentParser(description='embedding layer to run on the IPU')
parser.add_argument('--batch-size', type=int, default=1,
                    help="Set batch-size")
parser.add_argument('--embedding-size', type=int, default=32,
                    help="Size of source and target embedding")
parser.add_argument('--sequence-length', type=int, default=128,
                    help="Size of input length (by padding or truncating)")
parser.add_argument('--vocab-size', type=int, default=15000,
                    help="Size of vocab")
parser.add_argument('--steps', type=int, default=50000,
                    help="Number of steps to complete in training")
parser.add_argument('--profiling', type=bool, default=True,
                    help="Whether to enable profiling on IPU")
args = parser.parse_args()

def embedding_graph(input_ids):
    embedding_output,embedding_table = modeling.embedding_lookup(input_ids,args.vocab_size)
    embedding_output = modeling.embedding_postprocessor(input_tensor=embedding_output)
    return embedding_output,embedding_table


with ipu_scope("/device:IPU:0"):
    input_ids = tf.placeholder(tf.int32,shape=(args.batch_size, args.sequence_length),name='input_ids')
    batch = ipu.ipu_compiler.compile(embedding_graph, [input_ids])

opts = utils.create_ipu_config(profiling=args.profiling,profile_execution=args.profiling)
cfg = utils.auto_select_ipus(opts,1)
ipu.utils.configure_ipu_system(cfg)

print (f"Running with batch_size: .%d sequence_length :%d" % (args.batch_size,args.sequence_length))
input_t = np.random.rand(args.batch_size, args.sequence_length).astype(np.int32)


with tf.device('cpu'):
    report = gen_ipu_ops.ipu_event_trace()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embedding_output,embedding_table = sess.run(batch,feed_dict = {input_ids: input_t})
    print (embedding_output)
    raw_reports = sess.run(report)
    save_tf_report(raw_reports)
