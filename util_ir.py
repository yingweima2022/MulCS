import numpy as np
import configs
import argparse
from collections import Counter
import json
from utils import PAD_ID, UNK_ID

Opcode1 = ['ret', 'add', 'sub', 'mul', 'div', 'rem', 'fneg', 'getelementptr', 'select', 'shl', 'lshr', 'ashr', 
            'and', 'or', 'xor', 'cmp_gt', 'cmp_ge', 'cmp_lt', 'cmp_le', 'cmp_eq', 'cmp_ne', 'cmp_no']
Opcode2 = ['add', 'sub', 'mul', 'div', 'rem', 'fneg', 'getelementptr', 'select', 'shl', 'lshr', 'ashr', 
            'and', 'or', 'xor', 'cmp_gt', 'cmp_ge', 'cmp_lt', 'cmp_le', 'cmp_eq', 'cmp_ne', 'cmp_no']

# 输入单个ir图对应的json_dict，从dataloader输出对应anno，adjmat，node_mask
def get_one_ir_npy_info(json_graph_dict, n_node, n_edge_types, max_word_num, pooling_type, annotation_dim, is_partial_attn):
    
    node_num = min(len(json_graph_dict), n_node)
    save_edge_digit_list = []
    word_num = []
    anno = np.zeros([n_node, max_word_num])
    word_mask = np.zeros([n_node, max_word_num, annotation_dim])
    #print(json_graph_dict)

    for i in range(0, node_num):#what happened???
        word_list = json_graph_dict[str(i)]['wordid']
        word_num_this_node = len(word_list)

        if pooling_type == 'max_pooling':
            for j in range(word_num_this_node, max_word_num):
                word_mask[i][j] = -10000

        else: # avg_pooling
            for j in range(0, word_num_this_node):
                word_mask[i][j] = 8.0 / word_num_this_node
        
        for j in range(0, word_num_this_node):
            anno[i][j] = word_list[j]

        if 'snode' in json_graph_dict[str(i)].keys():
            snode_list = json_graph_dict[str(i)]['snode']
            edgetype_list = json_graph_dict[str(i)]['edgetype']
            for j in range(0, len(snode_list)):
                snode = snode_list[j]
                edgetype = edgetype_list[j]
                if snode < n_node: # 超出设定的最大节点数的节点舍弃
                    is_control_edge = int(edgetype)
                    if is_control_edge == 1:
                        save_edge_digit_list.append([i, snode, 1]) # 1代表控制边，0代表数据边
                    else:
                        save_edge_digit_list.append([i, snode, 0])
                    
    # adjmat: [n_node x (n_node * n_edge_types * 2)]
    adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
    
    # node_mask: [n_node]
    node_mask = [1 if k < node_num else 0 for k in range(0, n_node)]
    if is_partial_attn:
        for i in range(0, node_num):
            if json_graph_dict[str(i)]['mask'] == 0:
                node_mask[i] = 0

    return anno, adjmat, node_mask, word_mask


def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
    a = np.zeros([n_node, n_node * n_edge_types * 2])

    for edge in save_edge_digit_list:
        src_idx = edge[0]
        tgt_idx = edge[1]
        e_type = edge[2]

        a[tgt_idx][(e_type) * n_node + src_idx] = 1
        a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1

    return a
    
def construct_shuffle_data(args):
    index = np.load(args.shuffle_index_file)
    
    dir_path = args.data_path + args.dataset
    all_ir_file_path = dir_path + args.all_ir_file
    shuffle_all_ir_file_path = dir_path + args.shuffle_all_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ir_file_path, 'r') as all_ir_file, open(shuffle_all_ir_file_path, 'w') as shuffle_all_ir_file:
        lines = all_ir_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
        print('all_num of ir:\n', len(mark_list))

        for i in range(0, 41100):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                shuffle_all_ir_file.write(lines[j])



# 把数据集按shuffle_index.npy分成训练集和测试集，保持和desc一样的顺序
def split_data(args):
    index = np.load(args.shuffle_index_file)
    
    dir_path = args.data_path + args.dataset
    all_ir_file_path = dir_path + args.all_ir_file
    train_ir_file_path = dir_path + args.train_ir_file
    test_ir_file_path = dir_path + args.test_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ir_file_path, 'r') as all_ir_file:
        lines = all_ir_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
    print('all_num of ir:\n', len(mark_list))

    with open(train_ir_file_path, 'w') as train_ir_file,  open(test_ir_file_path, 'w') as test_ir_file:
        for i in range(0, args.trainset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                train_ir_file.write(lines[j])
        for i in range(args.testset_start_ind, args.testset_start_ind+args.testset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                test_ir_file.write(lines[j])


def generate_test_data(args):
    index = np.load(args.shuffle_index_file)
    
    dir_path = args.data_path + args.dataset
    all_ir_file_path = dir_path + args.all_ir_file
    
    test_ir_file_path = dir_path + args.test_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ir_file_path, 'r') as all_ir_file:
        lines = all_ir_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
    print('all_num of ir:\n', len(mark_list))

    with open(test_ir_file_path, 'w') as test_ir_file:
        for i in range(args.testset_start_ind, args.testset_start_ind+args.testset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                test_ir_file.write(lines[j])

def split_mask_data(args):
    index = np.load(args.shuffle_index_file)
    
    dir_path = args.data_path + args.dataset
    all_ir_file_path = dir_path + args.all_mask_ir_file
    train_ir_file_path = dir_path + args.train_mask_ir_file
    test_ir_file_path = dir_path + args.test_mask_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ir_file_path, 'r') as all_ir_file:
        lines = all_ir_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:6] == 'E:\\tmp' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
    print('all_num of ir:\n', len(mark_list))

    with open(train_ir_file_path, 'w') as train_ir_file,  open(test_ir_file_path, 'w') as test_ir_file:
        for i in range(0, args.trainset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                train_ir_file.write(lines[j])
        for i in range(args.testset_start_ind, args.testset_start_ind+args.testset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                test_ir_file.write(lines[j])

def transform_edge_to_node(args):
    dir_path = args.data_path + args.dataset
    origin_ir_file_path = dir_path + args.origin_ir_file
    all_ir_file_path = dir_path + args.all_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(origin_ir_file_path, 'r') as origin_ir_file:
        ir_lines = origin_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])

    with open(all_ir_file_path, 'w') as all_ir_file:
        for i in range(0, len(mark_list)):
            s_ind = mark_list[i][0]
            e_ind = mark_list[i][1]
            all_ir_file.write(ir_lines[s_ind])
    
            ir_graph_info_list = ir_lines[s_ind+1].split()
            node_num = int(ir_graph_info_list[0])
            edge_num = int(ir_graph_info_list[1])
            all_ir_file.write(str(node_num+edge_num) + ' ' + ir_graph_info_list[1] + '\n')

            for j in range(s_ind+2, e_ind):
                line = ir_lines[j]
                edge_info_list = line.split()

                if len(edge_info_list) == 2:
                    all_ir_file.write(line)
                else:
                    all_ir_file.write(edge_info_list[0] + ' ' + str(node_num) + ':' + edge_info_list[2] + '\n')
                    all_ir_file.write(str(node_num) + ':' + edge_info_list[2] + ' ' + edge_info_list[1] + '\n')
                    node_num += 1


# 辅助函数，用来观察数据特征
def observe_data(args):
    dir_path = args.data_path + args.dataset
    all_ir_file_path = dir_path + args.all_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ir_file_path, 'r') as all_ir_file:
        ir_lines = all_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])

    max_word_num = 0
    for i in range(0, len(mark_list)):
        s_ind = mark_list[i][0]
        e_ind = mark_list[i][1]

        for j in range(s_ind+2, e_ind):
            line = ir_lines[j]
            edge_info_list = line.split()

            start_node_list = edge_info_list[0].split(':')
            end_node_list = edge_info_list[1].split(':')
            s_word = start_node_list[1]
            len1 = len(s_word.split('_'))
            e_word = end_node_list[1]
            len2 = len(e_word.split('_'))
            if len1 > max_word_num:
                max_word_num = len1
                print(s_word)
            if len2 > max_word_num:
                max_word_num = len2
                print(e_word)
    print(max_word_num)


# 对ir二元组中的节点内容进行清洗
from tqdm import tqdm
def preprocess_origin_ir(args):
    dir_path = args.data_path + args.dataset
    origin_ir_file_path = dir_path + args.origin_ir_file
    all_ir_file_path = dir_path + args.all_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(origin_ir_file_path, 'r') as origin_ir_file:
        ir_lines = origin_ir_file.readlines()
        for i in tqdm(range(0, len(ir_lines))):
            line = ir_lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])

    with open(all_ir_file_path, 'w') as all_ir_file:
        for i in tqdm(range(0, len(mark_list))):
            s_ind = mark_list[i][0]
            e_ind = mark_list[i][1]
            all_ir_file.write(ir_lines[s_ind])
            all_ir_file.write(ir_lines[s_ind+1])

            for j in range(s_ind+2, e_ind):
                line = ir_lines[j]
                edge_info_list = line.split()

                start_node_list = edge_info_list[0].split(':')
                s_node = start_node_list[1]

                end_node_list = edge_info_list[1].split(':')
                e_node = end_node_list[1]
                
                all_ir_file.write(start_node_list[0]+':'+clean_node(s_node)+' '+
                                    end_node_list[0]+':'+clean_node(e_node)+' '+
                                    edge_info_list[2]+'\n')


# 对节点内容做清洗，主要针对带下划线的词，包括变量和函数名        
def clean_node(node_str):
    # 如果是数字节点，直接返回
    if node_str.isdigit():
        return 'ID'

    # 改掉一些特殊情况
    if node_str[0:9] == '__func__.':
        node_str = 'func_' + node_str[9:]
    if node_str[0:13] == '__FUNCTION__.':
        node_str = 'function_' + node_str[13:]
    if node_str[0:6] == 'FLAC__':
        node_str = 'flac_' + node_str[6:]
    #print(node_str)

    # 去掉一些特殊字符，包括'.'，数字，大写字母
    new_node_str = ''
    for i in range(0, len(node_str)):
        if node_str[i] == '.':
            new_node_str += '_'
        elif node_str[i] >= '0' and node_str[i] <= '9':
            continue
        elif node_str[i] >= 'A' and node_str[i]  <= 'Z':
            new_node_str += node_str[i].lower()
        else:
            new_node_str += node_str[i]
    #print(new_node_str)
    '''
    new_node_str = new_node_str.strip('_') # 得先去一次，防止出现'i_'这种情况，不过可能还是会有'a_b'这种情况没法处理，会变成空字符串
    # 处理'a_b'以及单字符字符串情况
    if len(new_node_str) == 3:
        if new_node_str[1] == '_':
            return new_node_str
    '''
    if len(new_node_str) == 1:
        return new_node_str
    
    # 去掉下滑线间的所有单字母字符，但是可能会出现去掉单字符后字符串为空的情况
    new2_node_str = ''
    for i in range(0, len(new_node_str)):
        if i == 0:
            if new_node_str[i+1] == '_':
                new2_node_str += '_'
            else:
                new2_node_str += new_node_str[i]
        elif i == len(new_node_str)-1:
            if new_node_str[i-1] == '_':
                new2_node_str += '_'
            else:
                new2_node_str += new_node_str[i]
        else:
            if new_node_str[i-1] == '_' and new_node_str[i+1] == '_':
                new2_node_str += '_'
            else:
                new2_node_str += new_node_str[i]
    # 如果字符串变成全'_'了，就不做这步处理
    flag = 0
    for i in range(0, len(new2_node_str)):
        if new2_node_str[i] != '_':
            flag = 1
    if flag == 0:
        new2_node_str = new_node_str
    #print(new2_node_str)

    # 去掉字符串的头尾'_'以及连续'_'
    new2_node_str = new2_node_str.strip('_')
    new3_node_str = ''
    for i in range(0, len(new2_node_str)):
        if i == len(new2_node_str)-1:
            new3_node_str += new2_node_str[i]
        elif new2_node_str[i] == '_' and new2_node_str[i+1] == '_':
            continue
        else:
            new3_node_str += new2_node_str[i]
    #print(new3_node_str)

    # 超出5个词的部分删除
    cnt_num = 0
    for i in range(0, len(new3_node_str)):
        if new3_node_str[i] == '_':
            cnt_num += 1
        if cnt_num == 5:
            new3_node_str = new3_node_str[0:i]
            break
    #print(new3_node_str)

    return new3_node_str
        

# 词汇中有'_'的 先/不 拆开再统计，只根据训练集中的词汇建词表
def create_dict_file(args):
    dir_path = args.data_path + args.dataset
    train_ir_file_path = dir_path + args.train_ir_file

    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()

    mark_list = []
    start_index, end_index = [0, 0]
    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])    

    ir_words = []
    for i in range(0, len(mark_list)):
        s_ind = mark_list[i][0]
        e_ind = mark_list[i][1]
        for j in range(s_ind+2, e_ind):
            edge_info_list = ir_lines[j].split()
            s_node_list = edge_info_list[0].split(':')
            e_node_list = edge_info_list[1].split(':')

            # 拆开下划线
            if args.word_split_type == 'split':
                if s_node_list[1] == 'control_label' or s_node_list[1] == 'label_true' or s_node_list[1] == 'label_false':
                    ir_words.append(s_node_list[1])
                else:
                    s_node = s_node_list[1].split('_')
                    for i in range(0, len(s_node)):
                        ir_words.append(s_node[i])
                if e_node_list[1] == 'control_label' or e_node_list[1] == 'label_true' or e_node_list[1] == 'label_false':
                    ir_words.append(e_node_list[1])
                else:
                    e_node = e_node_list[1].split('_')
                    for i in range(0, len(e_node)):
                        ir_words.append(e_node[i])
            # 不拆开下划线，对应type:'no_split'
            else:
                ir_words.append(s_node_list[1])
                ir_words.append(e_node_list[1])

    vocab_ir_info = Counter(ir_words)
    print("vocab_len:",len(vocab_ir_info))
    # print(vocab_ir_info)
    
    tmp = vocab_ir_info.most_common()
    #print(tmp[25000])
    for i in range(0, len(tmp)):
        t = tmp[i]
        if (t[1] == 4):
            print(i)
            break
    
    vocab_ir = [item[0] for item in vocab_ir_info.most_common()[:args.ir_word_num-2]]
    vocab_ir_index = {'<pad>':0, '<unk>':1}
    vocab_ir_index.update(zip(vocab_ir, [item+2 for item in range(len(vocab_ir))]))

    # 保存字典json文件
    vocab_ir_file_path = dir_path + args.vocab_ir_file
    ir_dic_str = json.dumps(vocab_ir_index)
    with open(vocab_ir_file_path, 'w') as vocab_ir_file:
        vocab_ir_file.write(ir_dic_str)


class multidict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


# 把txt格式的IR转成json输入，这样dataloader中按索引就可以遍历每张图
def txt2json(args, ir_txt_file_path):
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
            #根据edge_type改成数字
            if edge_type == "data":
                edge_type = 0
            elif edge_type == "control":
                edge_type = 1

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
                    
                    if s_node_list[1] in Opcode2:
                        graph_dict[i][s_node_index]['mask'] = 0
                    else:
                        graph_dict[i][s_node_index]['mask'] = 1
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
                    
                    if e_node_list[1] in Opcode2:
                        graph_dict[i][e_node_index]['mask'] = 0
                    else:
                        graph_dict[i][e_node_index]['mask'] = 1
                    
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

# 根据ir_mask_file中的内容判断对哪些节点做mask


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
    

def cnt_node_num(args):
    dir_path = args.data_path + args.dataset
    train_ir_file_path = dir_path + args.train_ir_file

    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()

    mark_list = []
    start_index, end_index = [0, 0]
    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)]) 
    
    node_num = []
    for i in range(0, len(mark_list)):
        s_ind = mark_list[i][0]+1
        line = ir_lines[s_ind]
        n_num = int(line.split()[0])
        node_num.append(n_num)

    cnt = 0
    for i in range(0, len(node_num)):
        if node_num[i] > 1024:
            cnt += 1
    print('cnt = ', cnt)
        

def parse_args():
    parser = argparse.ArgumentParser("Prepare IR data for IREmbeder")
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='c_python/')

    parser.add_argument('--origin_ir_file', type=str, default='origin.ir.txt')
    parser.add_argument('--all_ir_file', type=str, default='all.ir.txt')
    parser.add_argument('--shuffle_all_ir_file', type=str, default='shuffle.all.ir.txt')
    parser.add_argument('--all_mask_ir_file', type=str, default='all.mask.ir.txt')

    parser.add_argument('--train_ir_file', type=str, default='train.ir.txt')
    parser.add_argument('--test_ir_file', type=str, default='test.ir.txt') 
    parser.add_argument('--train_mask_ir_file', type=str, default='train.mask.ir.txt')
    parser.add_argument('--test_mask_ir_file', type=str, default='test.mask.ir.txt')
    parser.add_argument('--train_ir_json_file', type=str, default='train.ir.json')
    parser.add_argument('--test_ir_json_file', type=str, default='test.ir.json')
   
    parser.add_argument('--vocab_ir_file', type=str, default='vocab.ir.json')

    parser.add_argument('--n_node', type=int, default=150)
    parser.add_argument('--n_edge_types', type=int, default=2)
    parser.add_argument('--state_dim', type=int, default=512)
    parser.add_argument('--annotation_dim', type=int, default=300)
    parser.add_argument('--ir_word_num', type=int, default=15000)

    parser.add_argument('--trainset_num', type=int, default=39000)
    parser.add_argument('--testset_num', type=int, default=2000)
    parser.add_argument('--testset_start_ind', type=int, default=39000)

    parser.add_argument('--word_split_type', type=str, default='split') # no_split
    parser.add_argument('--with_opcode', type=int, default=0)

    parser.add_argument('--shuffle_index_file', type=str, default='data/shuffle_index.npy')

    return parser.parse_args()

def process_all():
    

    #observe_data(args)
    # print('1, preprocess_origin_ir')
    # preprocess_origin_ir(args)
    # print('2, construct_shuffle_data')
    # construct_shuffle_data(args)
    # print('3, split_data')
    # split_data(args)
    # print('4, create_dict_file')
    # create_dict_file(args)
    
    #split_mask_data(args)
    
    # dir_path = args.data_path + args.dataset
    # ir_txt_all_file_path = dir_path + args.all_ir_file
    # txt2json(args, ir_txt_all_file_path)
    

    dir_path = args.data_path + args.dataset
    ir_txt_train_file_path = dir_path + args.train_ir_file
    ir_txt_test_file_path = dir_path + args.test_ir_file
    # txt2json(args, ir_txt_train_file_path)
    txt2json(args, ir_txt_test_file_path)

def generate_test():
    args = parse_args()
    generate_test_data(args)
    dir_path = args.data_path + args.dataset

    dir_path = args.data_path + args.dataset
    ir_txt_test_file_path = dir_path + args.test_ir_file
    txt2json(args, ir_txt_test_file_path)

if __name__ == '__main__':
    args = parse_args()
    process_all()
    #generate_test()
    '''
    dir_path = args.data_path + args.dataset
    ir_txt_train_file_path = dir_path + args.train_ir_file
    ir_train_mask_file_path = dir_path + args.train_mask_ir_file
    ir_txt_test_file_path = dir_path + args.test_ir_file
    ir_test_mask_file_path = dir_path + args.test_mask_ir_file
    txt2json_mask(args, ir_txt_train_file_path, ir_train_mask_file_path)
    txt2json_mask(args, ir_txt_test_file_path, ir_test_mask_file_path)
    '''

    '''
    dir_path = args.data_path + args.dataset
    train_ir_file_path = dir_path + args.train_ir_file
    mark_list = []
    start_index, end_index = [0, 0]
    with open(train_ir_file_path, 'r') as train_ir_file:
        lines = train_ir_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])

    max_node_num = 0
    for i in range(0, len(mark_list)):
        line = lines[mark_list[i][0]+1]
        node_num = int(line.split()[0])
        if node_num > max_node_num:
            max_node_num = node_num
            print(max_node_num)
            print('i', i)
    '''
