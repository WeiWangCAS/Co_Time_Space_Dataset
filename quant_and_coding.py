from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import sys
import pickle
import codecs
import math

encode_app = 'Encoder_app'
encoder_cfg = 'Encoder_cfg'
quant_feat_dir = './intermediate_data/co_time_space_data/'

if not os.path.exists(quant_feat_dir):
    os.makedirs(quant_feat_dir)

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

def quant(feat, bits, feat_format='NHWC'):
    if feat_format == 'NCHW':
        feat = np.transpose(feat, (0, 2, 3, 1))
    pad_size_h = np.mod(feat.shape[1], 8)
    if pad_size_h != 0:
        pad_size_h = 8 - pad_size_h
    pad_size_w = np.mod(feat.shape[2], 8)
    if pad_size_w != 0:
        pad_size_w = 8 - pad_size_w
    data_log2 = np.log2(feat+1)
    maxLog2Val = data_log2.max()
    datalog2SampleScaled = np.rint(data_log2 * ((2**bits)-1) / maxLog2Val).astype(np.uint8)
    padded_matrix = np.pad(datalog2SampleScaled, ((0,0),(0, pad_size_h),(0,pad_size_w),(0,0)), 'constant')
    meta_data = (maxLog2Val, (pad_size_h, pad_size_w), (feat.shape[1]+pad_size_h, feat.shape[2]+pad_size_w, feat.shape[3]))
    return padded_matrix, meta_data

def exec_encode(op_path, meta_data, feat_name, Qp):
    input_yuv = os.path.join(op_path, feat_name+'.yuv')
    _, _, yuv_size = meta_data

    output_yuv = os.path.join(op_path, 'output_yuv')
    if not os.path.exists(output_yuv):
        os.makedirs(output_yuv)
    output_bin = os.path.join(op_path, 'output_bin')
    if not os.path.exists(output_bin):
        os.makedirs(output_bin)
    log_dir = os.path.join(op_path, 'encode_log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    rec_yuv = os.path.join(output_yuv, 'rec_{}_Qp{}.yuv'.format(feat_name, Qp))
    bitstream = os.path.join(output_bin, '{}_Qp{}.bin'.format(feat_name, Qp))
    log_file = os.path.join(log_dir, 'enc_{}_Qp{}.log'.format(feat_name, Qp))
    cmd = "{} -c {} -i {} -wdt {} -hgt {} -fr 30 --InputChromaFormat=400 --InternalBitDepth=8 -o {} -q {} -f {} -b {} > {} 2>/dev/null".format(
                encode_app, encoder_cfg, input_yuv, yuv_size[1], yuv_size[0], rec_yuv, Qp, yuv_size[2], bitstream, log_file)
    flag = os.system(cmd)
    print("flag is {}".format(flag))
    if flag !=0:
        print('error occured when encode {}/{} at Qp {}.'.format(op_path, feat_name, Qp))

def save_yuv(quant_data, save_dir):
    fp = open(save_dir, 'wb')
    for i in range(quant_data.shape[-1]):
        fp.write(np.squeeze(quant_data[0, :, :, i]).tobytes()) # fp.write(np.squeeze(quant_data[0, :, :, i]).tobytes())
    fp.close()

def read_log(file_name):
    total_volume = 0
    with open(file_name, 'r') as f:
        content = f.readlines()
    for str_line in content:
        if str_line.startswith('POC'):
            component_list = [j for j in str_line.split(' ') if j!='']
            bits_index = component_list.index('bits')
            bits_num = int(component_list[bits_index-1])
            total_volume += bits_num
    return total_volume #+ 6 + 32 + 24

def read_yuv(yuv_path, data_shape):
    with open(yuv_path, 'rb') as fp:
        strs = fp.read()
    data = np.frombuffer(strs, dtype=np.uint8)
    data.shape = (data_shape[2], data_shape[0], data_shape[1])
    return np.transpose(data, (1, 2, 0))

def dequant(quant_data, maxLog2Val, pad_size, bits):
    recData = np.exp2(quant_data[:quant_data.shape[0]-pad_size[0], :quant_data.shape[1]-pad_size[1], :].astype(np.float32) / ((2**bits)-1) * maxLog2Val) - 1
    return np.expand_dims(recData.astype(np.float32), axis=0)

def proc_dequant_yuv(yuv_path, maxLog2Val, pad_size, yuv_size, bits, feat_format='NCHW'):
    yuv_data = read_yuv(yuv_path, yuv_size)
    rec_data = dequant(yuv_data, maxLog2Val, pad_size, bits)
    if feat_format == 'NCHW':
        rec_data = np.transpose(rec_data, (0, 3, 1, 2))
    return rec_data

def HEVC_encoding(feat,bits,Qp):
    print(f'The current qp is {Qp}')
    print(f'the feature\'s shape is {feat.shape}')
    quant_feat, feat_meta = quant(feat, bits, feat_format='NCHW')
    print(f'the quant_feat\'s shape is {quant_feat.shape}')
    print(f'the feat_meat is {feat_meta}')
    save_yuv(quant_feat, os.path.join(quant_feat_dir, 'co_time_space_data.yuv'))
    # exec_encode(quant_feat_dir, feat_meta, 'co_time_space_data', Qp)
    log_dir = os.path.join(quant_feat_dir, 'encode_log', 'enc_{}_Qp{}.log'.format('co_time_space_data', Qp))
    compressed_bits = read_log(log_dir) / (8 * 1024)
    print("compressed_bits:{}".format(compressed_bits))
    maxLog2Val, pad_size, yuv_size = feat_meta
    ori_bits = (yuv_size[0]-pad_size[0])*(yuv_size[1]-pad_size[1])*yuv_size[2]*32/(8*1024)
    comp_rate = compressed_bits / ori_bits
    rec_yuv_dir = os.path.join(quant_feat_dir, 'output_yuv', 'rec_{}_Qp{}.yuv'.format('co_time_space_data', Qp))
    print('begin dequant')
    dequant_data = proc_dequant_yuv(rec_yuv_dir, maxLog2Val, pad_size, yuv_size, bits, feat_format='NCHW')
    print(f'the dequant_feat\'s shape is {dequant_data.shape}')
    mse = ((dequant_data - feat) ** 2).mean()
    psnr = 10 * math.log10((((2 * bits) - 1) ** 2) / mse)
    # shutil.rmtree(quant_feat_dir)
    if not os.path.exists(quant_feat_dir):
        os.makedirs(quant_feat_dir)
    # print("mse:{}".format(mse))
    # print("psnr:{}".format(psnr))
    return dequant_data, compressed_bits, ori_bits, mse, psnr, comp_rate
