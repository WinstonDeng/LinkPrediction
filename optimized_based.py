
'''Optimized-based method for link prediction'''

__author__ = 'Wenjin Deng'

import pandas as pd
import numpy as np
import linkpred
import random

''' global args '''
A_path = './A_similarity.csv'
B_path = './B_similarity.csv'
A_B_path = './A_B_adjacent.csv'
hidden_edges_num = 4 # 4,8,16,32,64
algorithm='SimRank'

def get_heterograph_items(data_path, mode, threshold=0.5):
    """
        This function is used to get edge pairs.

        Parameters:
        data_path - Str. graph csv file path
        mode - Str. Valid values are 'self' and 'correlation'. 
               Use 'self' for normalizing and discretizing edge score, further picking edge where score is 1
               Use 'correlation' for directly picking edge where score is 1
        threshold - Float. edge score = 0 if input < threshold, else 1

        Returns:
        src - List. Left nodes of Edges
        dst - List. Right nodes of Edges
        rows_name_list - List. csv rows name
        cols_name_list - List. csv cols name

        Note:
        src contains duplicate elements, and must be subset of rows_name_list. 
        It's also true for pair (dst, cols_name_list).
    """
    assert mode in ['self', 'correlation']
    # 1. read csv
    matrix = pd.read_csv(data_path)
    # 2. get rows name and cols name
    rows_name_list = matrix[matrix.columns.tolist()[0]].tolist()
    cols_name_list = matrix.columns.tolist()[1:]
    
    src = []
    dst = []
    
    # 3. store node pair of edges to src and dst
    if mode == 'self':
        min_value = min([matrix[name].min() for name in cols_name_list])
        max_value = max([matrix[name].max() for name in cols_name_list])
        for idx, row in matrix.iterrows():
            for name in cols_name_list:
                # 1. normalization (set value to 0-1)
                # 2. discretization (round)
                item_val = 0 if float(row[name])/(max_value-min_value) < threshold else 1
                # 3. record
                if item_val == 1:
                    src.append(rows_name_list[idx])
                    dst.append(name)
    else:
        for idx, row in matrix.iterrows():
            for name in cols_name_list:
                # 3. record
                if int(row[name]) == 1:
                    src.append(rows_name_list[idx])
                    dst.append(name)
    
    return src, dst, rows_name_list, cols_name_list

def build_graph(hidden_edges_num=4):
    """
        This function is used to build graph. 
        The origin graph is built, and then randomly hide K edges to obtain training graph.
        The training graph is saved as '*.net', while hidden edge info is saved as '*.hidedge' 

        Parameters:
        hidden_edges_num - Int. the number of hidden edges

        Returns:
        hid_pair_dict - Dict. Item is {'src - dst': edge_type_id}
    """
    # 1. prepare items for graph
    print('preparing graph...')
    A_src, A_dst, A_rows_name_list, A_cols_name_list = get_heterograph_items(A_path, 'self', threshold=0.5)
    B_src, B_dst, B_rows_name_list, B_cols_name_list = get_heterograph_items(B_path, 'self', threshold=0.5)
    AB_src, AB_dst, AB_rows_name_list, AB_cols_name_list = get_heterograph_items(A_B_path, 'correlation')

    # 2. trans node name to dict for unique id
    A_full_dict={}
    B_full_dict={}
    for id,name in enumerate(AB_rows_name_list):
        A_full_dict.update({name: id})
    for id,name in enumerate(AB_cols_name_list):
        B_full_dict.update({name: len(AB_rows_name_list)+id})
    
    # 3. randomly hide edge 
    print('random hid...')
    hid_pair = random.sample(range(len(AB_src)), hidden_edges_num)
    hid_pair_dict = {}
    
    # 4. save training graph
    print('saving graph...')
    
    save_graph_path = f'./example_hid{hidden_edges_num}.net'
    save_graph_hid_path = f'./example_hid{hidden_edges_num}_test.net'
    head_info = f'*network example_hid{hidden_edges_num}'
    hid_pair_str = ''
    
    with open(save_graph_path, 'w') as f:
        f.write(head_info+'\n')
        # 1. save nodes
        nodes_info = f'*vertices {len(AB_rows_name_list)+len(AB_cols_name_list)}'+'\n'
        f.write(nodes_info)
        hid_pair_str+=nodes_info

        for key in A_full_dict:
            tmp_str = f"{A_full_dict[key]} \"{key}\""+"\n"
            f.write(tmp_str)
            hid_pair_str+=tmp_str
        for key in B_full_dict:
            tmp_str= f"{B_full_dict[key]} \"{key}\""+"\n"
            f.write(tmp_str)
            hid_pair_str+=tmp_str

        # 2. save edges
        edges_info = '*edges'+'\n'
        f.write(edges_info)
        hid_pair_str += edges_info

        # 2.1 save A-A
        for src, dst in zip(A_src,A_dst):
            src_id = A_full_dict['A_'+str(src)]
            dst_id = A_full_dict['A_'+str(dst)]
            tmp_str = f'{src_id} {dst_id} 1'+'\n'
            tmp_inv_str = f'{dst_id} {src_id} 1'+'\n'
            f.write(tmp_str)
            f.write(tmp_inv_str) # save inversion pair for undirected graph
   
        # 2.2 save B-B
        for src, dst in zip(B_src,B_dst):
            src_id = B_full_dict['B_'+str(src)]
            dst_id = B_full_dict['B_'+str(dst)]
            tmp_str = f'{src_id} {dst_id} 1'+'\n'
            tmp_inv_str = f'{dst_id} {src_id} 1'+'\n'
            f.write(tmp_str)
            f.write(tmp_inv_str) # save inversion pair for undirected graph
     
        # 2.3 save A-B
        for idx, zip_item in enumerate(zip(AB_src,AB_dst)):
            src, dst = zip_item
            if idx in hid_pair: # save hidden info 
                hid_pair_str += f'{src} {dst} 1\n'
                hid_pair_str += f'{dst} {src} 1\n'
                hid_pair_dict.update({f'{src} - {dst}': 1})
                hid_pair_dict.update({f'{dst} - {src}': 1}) # save inversion pair for undirected graph
            else: # save A-B which is not hid
                src_id = A_full_dict[src]
                dst_id = B_full_dict[dst]
                f.write(f'{src_id} {dst_id} 1'+'\n')
                f.write(f'{dst_id} {src_id} 1'+'\n') # save inversion pair for undirected graph
    
    # 5. save hidden info to file
    with open(save_graph_hid_path, 'w') as f:
        f.write(head_info+'_test\n')
        f.write(hid_pair_str)

    return hid_pair_dict

def run(hid_edges, hidden_edges_num=hidden_edges_num, algorithm='SimRank'):
    """
        This function is used to train and eval. 
        Eval metric is Pr@k.

        Parameters:
        hid_edges - Dict. Dict. Item is {'src - dst': edge_type_id}
        hidden_edges_num - Int. the number of hidden edges
        algorithm - Str. Used to choose an algorithm. Valid values are 'SimRank' and 'PageRank'.

    """
    assert algorithm in ['SimRank', 'PageRank']

    # load graph
    print('loading graph...')
    G_train = linkpred.read_network(f"./example_hid{hidden_edges_num}.net")
    G_test = linkpred.read_network(f"./example_hid{hidden_edges_num}_test.net")
    test_set = set(linkpred.evaluation.Pair(u, v) for u, v in G_test.edges())
    # print(type(G))
    # print(dir(G))
    
    if algorithm == 'SimRank':
        # using simrank for modeling
        print('SimRank predicting...')
        simrank = linkpred.predictors.SimRank(G_train, excluded=G_train.edges())
        simrank_results = simrank.predict(c=0.8,num_iterations=100)
        top = simrank_results.top(hidden_edges_num)
        final_rate = 0
        # print(hid_edges)
        for name, score in top.items():
            print(name, score)
            # print('start_'+str(name)+'_end')
            if name in test_set:
                final_rate += 1
            # if str(name) in hid_edges:
            #     final_rate += 1
        print(f'SimRank Pr@k (k={hidden_edges_num}): {final_rate/hidden_edges_num}')
    elif algorithm == 'PageRank':
        # using pagerank for modeling
        print('PageRank predicting...')
        pagerank = linkpred.predictors.RootedPageRank(G_train)
        pagerank_results = pagerank.predict()
        top = pagerank_results.top(hidden_edges_num)
        final_rate = 0
        for name, score in top.items():
            print(name, score)
            if str(name) in hid_edges:
                final_rate += 1
        print(f'PageRank Pr@k (k={hidden_edges_num}): {final_rate/hidden_edges_num}')

if __name__ == "__main__":
    # G_train = linkpred.read_network(f"./example_hid{hidden_edges_num}.net")
    # print(G_train)
    # exit(1)

    hid_edges = build_graph(hidden_edges_num=hidden_edges_num)
    run(hid_edges, hidden_edges_num=hidden_edges_num, algorithm='SimRank')

'''
    存在的问题：
    1. 目前两种方法的结果都是 0,
        首先要检查 top.items() 的 name 是否正确对应 [no error]
        其次是尝试在 edges 中加入反向边 [done]
        训练测试流程不符合预期？
    2. 解耦 node 和 edge 构建过程，node 相对固定，每次修改 edge 才需要额外花时间
    3. 缺少预测结果保存（方法参数保存）
'''