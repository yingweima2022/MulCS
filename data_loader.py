import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle

from utils import PAD_ID, UNK_ID, indexes2sent
import configs
from util_ir import get_one_ir_npy_info

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, config, data_dir, f_irs=None, max_node_num=None, f_descs=None, max_desc_len=None):
    
        self.max_node_num = max_node_num
        self.max_desc_len = max_desc_len

        self.n_edge_types = config['n_edge_types']
        self.state_dim = config['state_dim']
        self.max_word_num = config['max_word_num']
        self.pooling_type = config['pooling_type']
        self.annotation_dim = config['annotation_dim']
        self.is_partial_attn = config['is_partial_attn']

        print("Loading Data...")

        if f_descs is None:
            self.graph_dict = json.loads(open(data_dir+f_irs, 'r').readline()) # 读入ir对应的json文件
            
            self.dataload_type = 0 # 注释为空，只取IR，针对user study中的整个code base
            self.data_len = len(self.graph_dict)
            print("{} entries".format(self.data_len))

        elif f_irs is None:
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]
            
            self.dataload_type = 1 # IR为空，只取注释，针对user study中选取的少量query
            self.data_len = self.idx_descs.shape[0]
            print("{} entries".format(self.data_len))

        else:
            self.graph_dict = json.loads(open(data_dir+f_irs, 'r').readline()) # 读入ir对应的json文件
            
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]
            
            self.dataload_type = 2 # 均不为空，注释和IR成对同时取，针对训练过程
            assert len(self.graph_dict)==self.idx_descs.shape[0]
            self.data_len = self.idx_descs.shape[0]
            print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq = seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):
        
        # anno:[n_node], adjmat:[n_node x (n_node*n_edge_types*2)], node_mask:[n_node]
        # node_num:[1], word_num: [n_node]

        if self.dataload_type == 0: # 注释为空，只取IR，针对user study中的整个code base
            anno, adjmat, node_mask, word_mask = get_one_ir_npy_info(self.graph_dict[str(offset)], 
                                self.max_node_num, self.n_edge_types, self.max_word_num, self.pooling_type, self.annotation_dim, self.is_partial_attn)

            anno = torch.from_numpy(anno).type(torch.LongTensor)
            adjmat = torch.from_numpy(adjmat).type(torch.FloatTensor)
            node_mask = torch.Tensor(node_mask)
            word_mask = torch.from_numpy(word_mask).type(torch.FloatTensor)

            return anno, adjmat, node_mask, word_mask

        elif self.dataload_type == 1: # IR为空，只取注释，针对user study中选取的少量query
            len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
            good_desc_len = min(int(len), self.max_desc_len)
            good_desc = self.descs[pos: pos+good_desc_len]
            good_desc = self.pad_seq(good_desc, self.max_desc_len)

            return good_desc, good_desc_len

        else: # 均不为空，注释和IR成对同时取，针对训练过程
            anno, adjmat, node_mask, word_mask = get_one_ir_npy_info(self.graph_dict[str(offset)], 
                                self.max_node_num, self.n_edge_types, self.max_word_num, self.pooling_type, self.annotation_dim, self.is_partial_attn)

            anno = torch.from_numpy(anno).type(torch.LongTensor)
            adjmat = torch.from_numpy(adjmat).type(torch.FloatTensor)
            node_mask = torch.Tensor(node_mask)
            word_mask = torch.from_numpy(word_mask).type(torch.FloatTensor)

            len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
            good_desc_len = min(int(len), self.max_desc_len)
            good_desc = self.descs[pos: pos+good_desc_len]
            good_desc = self.pad_seq(good_desc, self.max_desc_len)
            
            rand_offset = random.randint(0, self.data_len-1)
            len, pos = self.idx_descs[rand_offset][0], self.idx_descs[rand_offset][1]
            bad_desc_len = min(int(len), self.max_desc_len)
            bad_desc = self.descs[pos: pos+bad_desc_len]
            bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

            return anno, adjmat, node_mask, word_mask, good_desc, good_desc_len, bad_desc, bad_desc_len
    
    def __len__(self):
        return self.data_len

def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 


if __name__ == '__main__':
    device = 'cpu'
    config = getattr(configs, 'config_IREmbeder')()
    input_dir = './data/github_user/'

    test_set = CodeSearchDataset(config, input_dir, f_irs=config['all_ir'], max_node_num=config['n_node'])
    #test_set = CodeSearchDataset(config, input_dir, f_descs=config['query_desc'], max_desc_len=config['desc_len'])
    test_data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=1)
    print('number of batch:\n', len(test_data_loader))
    print('============ Train Data ================')
    k = 0
    '''
    for batch in test_data_loader:
        #print(batch)
        anno, adjmat, node_mask, node_num, word_num, good_desc, good_desc_len, bad_desc, bad_desc_len = [tensor.to(device) for tensor in batch]
        #print(anno)
        print(adjmat)
        for i in range(0, 160):
            for j in range(0, 320):
                if adjmat[0][i][j] == 1:
                    print(i, j)
        #print(node_num)
        #print(word_num)
        k+=1
        if k>0: break
    '''
    
