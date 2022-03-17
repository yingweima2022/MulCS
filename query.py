import os
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch

import models, configs, data_loader 
from modules import get_cosine_schedule_with_warmup
from utils import similarity, normalize
from data_loader import *

import matplotlib.pyplot as plt

def test(config, model, device, query, code):
    logger.info('Test Begin...')

    model.eval()
    model.to(device)

    # load data
    data_path = args.data_path+args.dataset+'/'
    test_set = eval(config['dataset_name'])(config, data_path,
                                config['test_ir'], config['n_node'],
                                config['test_desc'], config['desc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=32,
                                        shuffle=False, drop_last=False, num_workers=1)
    # encode tokens and descs
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in data_loader:
        # batch[0:3]: init_input, adjmat, node_mask
        code_batch = [tensor.to(device) for tensor in batch[:4]]
        # batch[3:5]: good_desc, good_desc_len
        desc_batch = [tensor.to(device) for tensor in batch[4:6]]
        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32) # [poolsize x hid_size]
            # normalize when sim_measure=='cos'
            code_repr = normalize(code_repr)
            desc_repr = normalize(desc_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0) # +batch_size
    # code_reprs: [n_processed x n_hidden]
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs) 

    # calculate similarity
    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []
    test_sim_result, test_rank_result = [], []
    for i in tqdm(range(0, n_processed)):
        desc_vec = np.expand_dims(desc_reprs[i], axis=0) # [1 x n_hidden]
        sims = np.dot(code_reprs, desc_vec.T)[:,0] # [n_processed]
        negsims = np.negative(sims)
        predict = np.argsort(negsims)
        
        # SuccessRate@k
        predict_1, predict_5, predict_10 = [int(predict[0])], [int(k) for k in predict[0:5]], [int(k) for k in predict[0:10]]
        sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
        sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
        sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)
        # MRR
        predict_list = predict.tolist()
        rank = predict_list.index(i)
        sum_mrr.append(1/float(rank+1))

        # results need to be saved
        predict_20 = [int(k) for k in predict[0:20]]
        sim_20 = [sims[k] for k in predict_20]
        test_sim_result.append(zip(predict_20, sim_20))
        test_rank_result.append(rank+1)

    logger.info(f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')
    save_path = args.data_path + 'result/'
    sim_result_filename, rank_result_filename = 'sim.npy', 'rank.npy'
    np.save(save_path+sim_result_filename, test_sim_result)
    np.save(save_path+rank_result_filename, test_rank_result)


def test_query(config, model, device, query, code):
    logger.info('Test Begin...')

    model.eval()
    model.to(device)

    # load data
    data_path = args.data_path + args.dataset + '/'
    test_set = eval(config['dataset_name'])(config, data_path,
                                            config['test_ir'], config['n_node'],
                                            config['test_desc'], config['desc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=999,
                                              shuffle=True, drop_last=False, num_workers=1)
    # encode tokens and descs
    code_reprs, desc_reprs = [], []
    n_processed = 0
    final_r1 = 0
    final_r5 = 0
    final_r10 = 0
    final_mrr = 0
    cnt = 0
    print("ok")
    for batch in data_loader:
        # batch[0:3]: init_input, adjmat, node_mask
        code_batch = [tensor.to(device) for tensor in batch[:4]]
        # batch[3:5]: good_desc, good_desc_len
        desc_batch = [tensor.to(device) for tensor in batch[4:6]]
        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
            # normalize when sim_measure=='cos'
            code_repr = normalize(code_repr)
            desc_repr = normalize(desc_repr)
        code_reprs.append(code)
        desc_reprs.append(query)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)  # +batch_size

        if n_processed >= 999:
            # code_reprs: [n_processed x n_hidden]
            code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

            # calculate similarity
            sum_1, sum_5, sum_10, sum_mrr = [], [], [], []
            test_sim_result, test_rank_result = [], []
            desc_vec = np.expand_dims(desc_reprs[0], axis=0)  # [1 x n_hidden]
            sims = np.dot(code_reprs, desc_vec.T)[:, 0]  # [n_processed]
            negsims = np.negative(sims)
            predict = np.argsort(negsims)

            # SuccessRate@k
            predict_1, predict_5, predict_10 = [int(predict[0])], [int(k) for k in predict[0:5]], [int(k) for k in
                                                                                                    predict[0:10]]
            sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
            sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
            sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)
            # MRR
            predict_list = predict.tolist()
            rank = predict_list.index(i)
            sum_mrr.append(1 / float(rank + 1))

            # results need to be saved
            predict_20 = [int(k) for k in predict[0:20]]
            sim_20 = [sims[k] for k in predict_20]
            test_sim_result.append(zip(predict_20, sim_20))
            test_rank_result.append(rank + 1)
            R1 = np.mean(sum_1)
            R5 = np.mean(sum_5)
            R10 = np.mean(sum_10)
            MRR = np.mean(sum_mrr)
            final_r1 += R1
            final_r5 += R5
            final_r10 += R10
            final_mrr += MRR
            cnt += 1
            logger.info(f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')
            code_reprs, desc_reprs = [], []
            n_processed = 0
        # break
    logger.info(f'ave result')
    logger.info(f'R@1={final_r1/cnt}, R@5={final_r5/cnt}, R@10={final_r10/cnt}, MRR={final_mrr/cnt}')
    return final_r1/cnt, final_r10/cnt, final_mrr/cnt

    #save_path = args.data_path + 'result/'
    #sim_result_filename, rank_result_filename = 'sim.npy', 'rank.npy'
    #np.save(save_path + sim_result_filename, test_sim_result)
    #np.save(save_path + rank_result_filename, test_rank_result)


def txt2json_mask(args, ir_txt_file_path, ir_mask_file_path):
    mark_list = []
    start_index, end_index = [0, 0]
    ir_cnt = 1
    with open(ir_txt_file_path, 'r') as ir_txt_file:
        ir_lines = ir_txt_file.readlines()
        for i in range(0, len(ir_lines)):
            if ir_lines[i][0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
                ir_cnt += 1
        print('ir_cnt:\n', ir_cnt)
        mark_list.append([start_index, len(ir_lines)])
    
    with open(ir_mask_file_path, 'r') as ir_mask_file:
        ir_mask_lines = ir_mask_file.readlines()
        all_mask = []
        one_mask = []
        for i in range(0, len(ir_mask_lines)):
            if ir_mask_lines[i][0:6] == 'E:\\tmp' and i != 0:
                all_mask.append(one_mask)
                one_mask = []
            elif ir_mask_lines[i][0:6] != 'E:\\tmp':
                word_list = ir_mask_lines[i].split(':')
                word_id = int(word_list[0])
                word = word_list[1].strip()
                if args.with_opcode == 1:
                    one_mask.append(word_id)
                else:
                    if word not in Opcode2:
                        one_mask.append(word_id)
        all_mask.append(one_mask)
    
    dir_path = args.data_path + args.dataset
    vocab_ir_file_path = dir_path + args.vocab_ir_file
    vocab = json.loads(open(vocab_ir_file_path, 'r').readline())

    graph_dict = multidict()
    for i in range(0, ir_cnt):
        s_ind, e_ind = mark_list[i]
        #print("Graph Index: ", i)
        for j in range(s_ind+2, e_ind):
            edge_info_list = ir_lines[j].split()
            s_node_list = edge_info_list[0].split(':')
            e_node_list = edge_info_list[1].split(':')
            edge_type = edge_info_list[2]

            #print(edge_info_list)

            # 根据word_split_type的不同，选择是否要拆开词，区别在于wordid的存法不同
            if (args.word_split_type == 'split'):
                # 分别考虑起终节点，'control_label'和'return_point'不做拆分，其他存成list，这里没区分边的类型
                s_node_index = int(s_node_list[0])
                if graph_dict[i][s_node_index]['wordid'] == {}:
                    if s_node_list[1] == 'control_label' or s_node_list[1] == 'label_true' or s_node_list[1] == 'label_false':
                        graph_dict[i][s_node_index]['wordid'] = [vocab.get(s_node_list[1], UNK_ID)]
                    else:
                        graph_dict[i][s_node_index]['wordid'] = []
                        s_node_word_list = s_node_list[1].split('_')
                        for k in range(0, len(s_node_word_list)):
                            graph_dict[i][s_node_index]['wordid'].append(vocab.get(s_node_word_list[k], UNK_ID))
                    
                    if s_node_index in all_mask[i]:
                        graph_dict[i][s_node_index]['mask'] = 1
                    else:
                        graph_dict[i][s_node_index]['mask'] = 0

                    '''
                    if s_node_list[1] in Opcode:
                        graph_dict[i][s_node_index]['mask'] = 0
                    else:
                        graph_dict[i][s_node_index]['mask'] = 1
                    '''
                    #print("snode: %s index: %d" %(s_node_list[1], s_node_index))
                    #print(graph_dict[i][s_node_index]['wordid'])

                e_node_index = int(e_node_list[0])
                if graph_dict[i][e_node_index]['wordid'] == {}:
                    if e_node_list[1] == 'control_label' or e_node_list[1] == 'label_true' or e_node_list[1] == 'label_false':
                        graph_dict[i][e_node_index]['wordid'] = [vocab.get(e_node_list[1], UNK_ID)]
                    else:
                        graph_dict[i][e_node_index]['wordid'] = []
                        e_node_word_list = e_node_list[1].split('_')
                        for k in range(0, len(e_node_word_list)):
                            graph_dict[i][e_node_index]['wordid'].append(vocab.get(e_node_word_list[k], UNK_ID))
                    
                    if e_node_index in all_mask[i]:
                        graph_dict[i][e_node_index]['mask'] = 1
                    else:
                        graph_dict[i][e_node_index]['mask'] = 0
                    '''
                    if e_node_list[1] in Opcode:
                        graph_dict[i][e_node_index]['mask'] = 0
                    else:
                        graph_dict[i][e_node_index]['mask'] = 1
                    '''
                    #print("enode: %s index: %d" %(e_node_list[1], e_node_index))
                    #print(graph_dict[i][e_node_index]['wordid'])
                    
            else:
                graph_dict[i][int(s_node_list[0])]['wordid'] = vocab.get(s_node_list[1], UNK_ID)
                graph_dict[i][int(e_node_list[0])]['wordid'] = vocab.get(e_node_list[1], UNK_ID)

            if graph_dict[i][int(s_node_list[0])]['snode'] == {}: # 该节点当前还无子节点
                graph_dict[i][int(s_node_list[0])]['snode'] = [int(e_node_list[0])]
                graph_dict[i][int(s_node_list[0])]['edgetype'] = [int(edge_type)]
                #print('if \{None\}', graph_dict[i][int(s_node_list[0])]['node'])
            else: # 该节点含多个子节点
                graph_dict[i][int(s_node_list[0])]['snode'].append(int(e_node_list[0]))
                graph_dict[i][int(s_node_list[0])]['edgetype'].append(int(edge_type))
                #print('multiple sons exist in line={}, i={}, s={}, node={}'.format(j, i, s_node_list[0], graph_dict[i][int(s_node_list[0])]['node']))

    graph_dict_str = json.dumps(graph_dict)
    ir_json_file_path = ir_txt_file_path[0:-3] + 'json'
    with open(ir_json_file_path, 'w') as ir_json_file:
        ir_json_file.write(graph_dict_str)    

def generate_vectors(query, code):
    


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='IREmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='java_github', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=185, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=False, help="Visualize training status in tensorboard")

    parser.add_argument('--trainset_num', type=int, default=160000)
    parser.add_argument('--testset_num', type=int, default=20000)
    parser.add_argument('--testset_start_ind', type=int, default=160000)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config = getattr(configs, 'config_'+args.model)()
    
    ##### Define model ######
    logger.info('Constructing Model..')
    final_mrr_list = []
    final_r1_list = []
    final_r10_list = []
    # for epo_id in range(20, args.reload_from, 10):
        # print(epo_id)
    model = getattr(models, args.model)(config) # initialize the model
    ckpt=f'./output/{args.model}/{args.dataset}/models/epo{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))

    #     #test(config, model, device)
    final_r1, final_r10, final_mrr = test_batch(config, model, device)
    #     final_r1_list.append(final_r1)
    #     final_r10_list.append(final_r10)
    #     final_mrr_list.append(final_mrr)
    
    # plt.plot(final_r1_list)
    # plt.plot(final_r10_list)
    # plt.plot(final_mrr_list)
    # plt.show()

