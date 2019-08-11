import numpy as np
import torch
import pickle
from pygsp import graphs
from torch_geometric.data import DataLoader, Data

def geometric_graph(images):
    images_tensor = torch.tensor(images)
    images_change_dim = images_tensor.permute(0, 2, 3, 1)
    images_reshape = images_change_dim.contiguous().view(-1, 3)

    # here x would be the (number of nodes, channel)
    x = images_reshape
    # Now, create a edge indices that is [2, no.edges] i.e [source nodes, destination nodes]
    # create a pygsp graph for edge_index

    G1 = graphs.Grid2d(images_tensor.shape[1],images_tensor.shape[2])
    A = (G1.A)
    A = A.tocoo()
    row = A.row
    col = A.col
    #edge_index_row = np.concatenate([row,row])
    #edge_index_col = np.concatenate([col,col])


    '''with open("row.txt","wb") as fp:
        pickle.dump(edge_index_row,fp)
    with open("col.txt","wb") as fp1:
        pickle.dump(edge_index_col,fp1)


    with open("row.txt", "rb") as fp1:
        row = pickle.load(fp1)
    with open("col.txt", "rb") as fp:
        col = pickle.load(fp)'''

    # this is the edge_list for the data.edge_list for the databse graph
    edge_list = torch.tensor([row, col],dtype=torch.long)

    # edge_attr is the positional coordinates of the matrix having size, (edge_list,2) this attr taken from pygsp G1.coords
    edge_attr = G1.coords
    #pseudo = np.concatenate([edge_attr,edge_attr])

    x = torch.tensor(x, dtype=torch.float)
    pseudo = torch.tensor(edge_attr,dtype=torch.long)
    data = [Data(x= x, edge_index=edge_list,pos=pseudo)]

    return data