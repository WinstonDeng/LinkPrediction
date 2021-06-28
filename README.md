## Link Prediction and Graph Reconstruction
This is a personal implementation for the task of Link Prediction.

### Dependencies
```
    python3
    pytorch 1.5+
    numpy
    pandas
    linkpred
    dgl
```

### Quickly start
1. Predict by optimized_based method

```python
    python optimized_based.py    
```

* For using different configs, you can modify global args in python file head.
    
```
    A_path = './A_similarity.csv' # self matrix A csv path
    B_path = './B_similarity.csv' # self matrix B csv path
    A_B_path = './A_B_adjacent.csv' # correlation matrix between A and B csv path
    hidden_edges_num = 4 # 4,8,16,32,64 
    algorithm='SimRank' # algorithm for modeling graph
```

2. Predict by learning_based method

```python
    python learning_based.py    
```

* For using different configs, you can modify global args in python file head.
    

```
    A_path = './A_similarity.csv' # self matrix A csv path
    B_path = './B_similarity.csv' # self matrix B csv path
    A_B_path = './A_B_adjacent.csv' # correlation matrix between A and B csv path
    hidden_edges_num = 4 # 4,8,16,32,64 
   
    embed_dim = 256 # node embedding size for input
    mid_dim = 1024 # node feature size for gcn hidden layer 
    out_dim = 512 # node feature size for output

    max_epoch = 200 # max training epooch
    neg_node_num = 16 # neg node sample num
    rebuild_graph = False # use for graph reconstruction task
    avg_cnt = 3 # repeat number for avgeraging result
    device=torch.device("cpu") # using cpu or gpu
```
### Todo
- [ ] fix bugs in optimized_based.py
- [x] fix bugs in learning_baesd.py
- [ ] clean and smiplify code for running faster

### License

This project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

### Acknowledgement
I would like to thank the [linkpred](https://github.com/rafguns/linkpred) and the [DGL](https://docs.dgl.ai/index.html).