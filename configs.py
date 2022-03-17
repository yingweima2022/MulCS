
def config_IREmbeder():   
    conf = {
            # added_params
            'transform_every_modal': 0,
            'use_attn': 1,
            'use_tanh': 1,
            'save_attn_weight': 0,
            'is_partial_attn': 1,
            'use_desc_attn': 1,

            # GGNN
            'state_dim': 512, # 512 300 # GGNN hidden state size
            'annotation_dim': 300,  # 300 100
            'n_edge_types': 2,
            'n_node': 250, # maximum nodenum
            'n_steps': 5, # propogation steps number of GGNN
            'output_type': 'no_reduce',
            'batch_size': 320,  # 64  320
            'n_layers': 1,  # 1
            'n_hidden': 512,  # 512  300
            'ir_attn_mode': 'sigmoid_scalar',
            'word_split': True,
            'pooling_type': 'avg_pooling', # avg_pooling
            'max_word_num': 8,  #not sure

            # data_params
            'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            # training data
            'train_ir':'train.ir.json',
            'train_desc':'train.desc.h5',
            # test data
            'test_ir':'test.ir.json',
            'test_desc':'test.desc.h5', 
            
            # user study data
            'all_ir': 'shuffle.all.ir.json',
            'query_desc': 'query.desc.h5',
                   
            # parameters
            'desc_len': 30,  # 30
            'n_desc_words': 10000,  # 10000    15000
            'n_ir_words': 15000,    # 15000    20000
            # vocabulary info
            'vocab_ir':'vocab.ir.json',
            'vocab_desc':'vocab.desc.json',
                    
            #training_params            
            'nb_epoch': 200,
            #'optimizer': 'adam',
            'learning_rate': 0.001, #0.0003, 0.001  5e-4
            'adam_epsilon':1e-8,
            'warmup_steps':1000,  # 5000
            'fp16': False,
            'fp16_opt_level': 'O1', #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].

            # model_params
            'emb_size': 300,   # 300 100
            # recurrent  
            'margin': 0.05,
            'sim_measure':'cos',
            'dropout': 0
    }
    return conf

