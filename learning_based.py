'''Learning-based method for link prediction'''

__author__ = 'Wenjin Deng'

import pandas as pd
import os
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random

''' global args '''
A_path = './A_similarity.csv'
B_path = './B_similarity.csv'
A_B_path = './A_B_adjacent.csv'
embed_dim = 256
mid_dim = 1024
out_dim = 512
hidden_edges = 4 # 4,8,16,32,64
max_epoch = 200
neg_node_num = 16 # neg node sample num
rebuild_graph = False
avg_cnt = 3
device=torch.device("cpu")

# if torch.cuda.is_available():
#     device=torch.device("cuda")

def get_heterograph_items(data_path, mode, threshold=0.75):
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
            # print(idx)
            # print(row)
            for name in cols_name_list:
                # 1. normalization (set value to 0-1)
                # 2. discretization (round)
                # item_val = round(float(row[name])/(max_value-min_value))
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

# Define a Heterograph Conv model
class RGCN(nn.Module):
    """
        copy from dgl offical docs
    """
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
        
# Define a Heterograph DotProduct Predictor
class HeteroDotProductPredictor(nn.Module):
    """
        copy from dgl offical docs
    """
    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

# Define a method to construct negative graph 
def construct_negative_graph(graph, k, etype):
    """
        copy from dgl offical docs
    """
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

# Define a method to normalize node embedding vector
def norm_node_embedding(node_h):
    """
        normalize node vector
    """
    node_h['A'] = F.normalize(node_h['A'], p=2, dim=1)
    node_h['B'] = F.normalize(node_h['B'], p=2, dim=1)
    return node_h

# Define a model
class Model(nn.Module):
    """
        refer from dgl offical docs
    """
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        h = norm_node_embedding(h) # if not use this, edge score will be out of range [-1,1]
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

# Define a loss
def compute_loss(pos_score, neg_score):
    """
        copy from dgl offical docs
    """
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

# train 
def train(hetero_graph, max_epoch, embed_dim, mid_dim, out_dim, hidden_edges):
    # 1. prepare nodes embedding
    A_embed = nn.Embedding(hetero_graph.number_of_nodes('A'), embed_dim)
    B_embed = nn.Embedding(hetero_graph.number_of_nodes('B'), embed_dim)
    A_edge_embed = nn.Embedding(hetero_graph.number_of_edges('A_self'), embed_dim)
    B_edge_embed = nn.Embedding(hetero_graph.number_of_edges('B_self'), embed_dim)
    AB_edge_embed = nn.Embedding(hetero_graph.number_of_edges('A_and_B'), embed_dim)
    hetero_graph.nodes['A'].data['feat'] = A_embed.weight
    hetero_graph.nodes['B'].data['feat'] = B_embed.weight
    hetero_graph.edges['A_self'].data['feat'] = A_edge_embed.weight
    hetero_graph.edges['B_self'].data['feat'] = B_edge_embed.weight
    hetero_graph.edges['A_and_B'].data['feat'] = AB_edge_embed.weight
    hetero_graph.to(device)

    node_features = {'A': hetero_graph.nodes['A'].data['feat'], 'B': hetero_graph.nodes['B'].data['feat']}

    # 2. instancing a model
    model = Model(embed_dim, mid_dim, out_dim, hetero_graph.etypes).to(device)
    model.train()
    
    # 3. prepare an optimizer
    opt = torch.optim.Adam(model.parameters())

    # 4 training schedule
    for epoch in range(max_epoch):
        negative_graph = construct_negative_graph(hetero_graph, neg_node_num, ('A', 'A_and_B', 'B'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('A', 'A_and_B', 'B'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f'training {epoch+1}//{max_epoch}\tloss:{loss.item()}')
        # print(pos_score.shape[0]) # edges num
    return model

# eval
def eval(model, hidden_graph):
    with torch.no_grad():
        model.eval()
        # 1. prepare nodes embedding
        src_embed = nn.Embedding(hidden_graph.number_of_nodes('A'), embed_dim)
        dst_embed = nn.Embedding(hidden_graph.number_of_nodes('B'), embed_dim)
        hidden_graph.nodes['A'].data['feat'] =  src_embed.weight
        hidden_graph.nodes['B'].data['feat'] =  dst_embed.weight
        hidden_graph.to(device)
        hidden_node = {'A':hidden_graph.nodes['A'].data['feat'], 'B':hidden_graph.nodes['B'].data['feat']}

        # 2. infer to obtain score list of hidden edges
        node_h = model.sage(hidden_graph, hidden_node)
        node_h = norm_node_embedding(node_h)
        score = model.pred(hidden_graph, node_h, ('A', 'A_and_B', 'B'))
        score = score.numpy()
        # print(score)
        print('src\tdst\tscore')
        valid_cnt = 0
        for idx, zip_item in enumerate(zip(AB_src_hid,AB_dst_hid)):
            src,dst = zip_item
            if score[idx][0]>0:
                valid_cnt += 1
            print(f'{src}\t{dst}\t{score[idx][0]}')
        print(f'Pr@k (k={len(AB_src_hid)}): {valid_cnt/len(AB_src_hid)}')
        return valid_cnt/len(AB_src_hid)

if __name__== "__main__":
    avg_score = 0

    for _ in range(avg_cnt):
        # 1. prepare items for graph
        A_src, A_dst, A_rows_name_list, A_cols_name_list = get_heterograph_items(A_path, 'self')
        B_src, B_dst, B_rows_name_list, B_cols_name_list = get_heterograph_items(B_path, 'self')
        AB_src, AB_dst, AB_rows_name_list, AB_cols_name_list = get_heterograph_items(A_B_path, 'correlation')

        # 2. trans node name to dict for unique id
        A_full_dict={}
        B_full_dict={}
        for id,name in enumerate(AB_rows_name_list):
            A_full_dict.update({name: id})
        for id,name in enumerate(AB_cols_name_list):
            B_full_dict.update({name: id})


        # 3. randomly hide edge
        if rebuild_graph:
            print('rebuild graph task')
            AB_src_hid = AB_src
            AB_dst_hid = AB_dst
        else:
            print('link prediction task')
            print('random hid...')
            hid_pair = random.sample(range(len(AB_src)), hidden_edges)
            AB_src_hid = []
            AB_dst_hid = []
            AB_src_rest = []
            AB_dst_rest = []
            for idx,zip_item in enumerate(zip(AB_src,AB_dst)):
                src, dst = zip_item
                if idx in hid_pair:
                    AB_src_hid.append(src)
                    AB_dst_hid.append(dst)
                else:
                    AB_src_rest.append(src)
                    AB_dst_rest.append(dst)

        # 4. build rest heterograph (rest = origin - hidden)
        if rebuild_graph:
            hetero_graph = dgl.heterograph(
                {
                    ('A', 'A_self', 'A') : ([A_full_dict['A_'+str(name)] for name in A_src], [A_full_dict['A_'+str(name)] for name in A_dst]),
                    ('B', 'B_self', 'B') : ([B_full_dict['B_'+str(name)] for name in B_src], [B_full_dict['B_'+str(name)] for name in B_dst]),
                    ('A', 'A_and_B', 'B') : ([A_full_dict[name] for name in AB_src],[B_full_dict[name] for name in AB_dst])
                }
            )
        else:
            hetero_graph = dgl.heterograph(
                {
                    ('A', 'A_self', 'A') : ([A_full_dict['A_'+str(name)] for name in A_src], [A_full_dict['A_'+str(name)] for name in A_dst]),
                    ('B', 'B_self', 'B') : ([B_full_dict['B_'+str(name)] for name in B_src], [B_full_dict['B_'+str(name)] for name in B_dst]),
                    ('A', 'A_and_B', 'B') : ([A_full_dict[name] for name in AB_src_rest],[B_full_dict[name] for name in AB_dst_rest])
                }
            )

        print('training grapha\n', hetero_graph)
        # print(hetero_graph.number_of_nodes())
        # print(hetero_graph.number_of_edges())

        # 5. train
        model = train(hetero_graph, max_epoch, embed_dim, mid_dim, out_dim, hidden_edges)
        
        # 6. build hidden graph with hidden edges
        if rebuild_graph:
            hidden_graph = hetero_graph
        else:
            hidden_graph = dgl.heterograph(
                    {
                        ('A', 'A_self', 'A') : ([A_full_dict['A_'+str(name)] for name in A_src], [A_full_dict['A_'+str(name)] for name in A_dst]),
                        ('B', 'B_self', 'B') : ([B_full_dict['B_'+str(name)] for name in B_src], [B_full_dict['B_'+str(name)] for name in B_dst]),
                        ('A', 'A_and_B', 'B') : ([A_full_dict[name] for name in AB_src_hid],[B_full_dict[name] for name in AB_dst_hid])
                    }
                )
        print('eval grapha\n', hidden_graph)

        # 7. eval
        final_score = eval(model, hidden_graph)

        avg_score += final_score
    
    print(f'avg_{avg_cnt}: {avg_score/avg_cnt}')
