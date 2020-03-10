import os
from tensorflow import pywrap_tensorflow
import numpy as np
import tensorflow as tf
from shutil import copyfile
import argparse

# parser
parser = argparse.ArgumentParser(description='Convert Precision Model')
parser.add_argument('--ckpt_dir', type=str, default='',
                    help='path to convert checkpoint file directory (default:'')')
args = parser.parse_args()

def convert_ckpt_to_fp(checkpoint_path,data_type=np.float16):
    """Convert checkpoint to fp weights and return saver.
    Args:
        init_checkpoint: Path to checkpoint file.
        data_type: np.float16, np.float32, np.float64,

    """
    ckpt = '.ckpt'
    sync_file = []
    checkpoint_name = None
    os.chdir(checkpoint_path)
    for each_file in  os.listdir(os.curdir):
        if ckpt in each_file:
            checkpoint_name = each_file.split(ckpt)[0]+ckpt
            break
        
    if checkpoint_name is None:
        return
    
    curent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/'
    out_dir = curent_dir +  checkpoint_path + "-F16"+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 

    for each_file in  os.listdir(os.curdir):
        ext = os.path.splitext(each_file)[1]
        if ext in ['.txt','.json']:
            copyfile(curent_dir+checkpoint_path+'/'+each_file, out_dir+each_file)
        
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_name)
    var_to_map = reader.get_variable_to_shape_map()
    val_f = {}
    for key, dim in var_to_map.items():
        val_f[key.strip(":0")] = tf.Variable(reader.get_tensor(key).astype(data_type))
        '''
        if 'word_embeddings' in key:
            temp = reader.get_tensor(key)[:2896,:]
            val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))#119547
        if 'dense' in key:
            if len(dim)>1:
                need_split_dim1 = False
                need_split_dim2 = False
                need_split_dim1 = True if dim[0]==3072 else False
                need_split_dim2 = True if dim[1]==3072 else False
                if need_split_dim1:
                    temp = reader.get_tensor(key)[:2048,:]
                    val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
                elif need_split_dim2:
                    temp = reader.get_tensor(key)[:,:2048]
                    val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
                elif need_split_dim1 and need_split_dim2:
                    temp = reader.get_tensor(key)[:2048,:2048]
                    val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
            else:
                if dim[0]==3072:
                    temp = reader.get_tensor(key)[:2048]
                    val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
        '''
        
    #get parameters before convert
    param_log_origin=''
    for key in var_to_map:
        param_log_origin += "tensor_name: "+key+"  shape:"+str(reader.get_tensor(key).shape)+"\r\n"
        param_log_origin += str(reader.get_tensor(key))+"\r\n"  
    writer = open(out_dir+'Param-'+str(reader.get_tensor(key).dtype)+'.txt', 'w', encoding="utf-8")
    writer.write(param_log_origin)      
  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph(curent_dir + checkpoint_path +'/'+checkpoint_name+'.meta')
        new_saver.restore(sess,curent_dir + checkpoint_path +'/'+checkpoint_name)  
        saver = tf.train.Saver(val_f)
        saver.save(sess, out_dir+checkpoint_name)  

    #save parameters after convert
    reader_convert = pywrap_tensorflow.NewCheckpointReader(out_dir+checkpoint_name)
    var_to_map_convert = reader_convert.get_variable_to_shape_map()  
    param_log_convert=''
    for item in var_to_map_convert:
        param_log_convert += "tensor_name: "+item+"  shape:"+str(reader_convert.get_tensor(item).shape)+"\r\n"
        param_log_convert += str(reader_convert.get_tensor(item))+"\r\n" 
    writer = open(out_dir+'Param-'+str(reader_convert.get_tensor(item).dtype)+'.txt', 'w', encoding="utf-8")
    writer.write(param_log_convert)      
    
    print("Convert Finish!")
    print("Save to path:"+out_dir)    


if __name__=='__main__':
    convert_ckpt_to_fp(args.ckpt_dir)