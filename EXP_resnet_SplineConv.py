#Here implementation of resnet using spline conv

import torch
import numbers as np
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import pdb
import pickle
from PIL import Image
import numpy as npy

from Bulid_pytorch_Graph import geometric_graph

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import SplineConv, global_mean_pool, global_max_pool, max_pool_x, voxel_grid, GMMConv

class baseblock(torch.nn.Module):
    expansion = 1               # explansion is used for number of filter . Here basebloack is have only one fileter.
    def __index__(self,input_planes, planes, stride=1,dim_change=None):
        super(baseblock, self).__index__()
        #declare convolutional layers with batch norms
        self.conv1 = torch.nn.Conv2d(input_planes,planes,stride=stride,kernel_size=3,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes,planes,stride=1,kernel_size=3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dim_change = dim_change
    def forward(self,x):
        #save the residue
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output

class bottleNeck(torch.nn.Module):
    expansion = 4
    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(bottleNeck, self).__init__()

        self.conv1 = SplineConv(input_planes, planes, dim=2,kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(planes)
        self.conv2 = SplineConv(planes, planes, dim=2, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(planes)
        self.conv3 = SplineConv(planes,planes*self.expansion, dim=2,kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(planes*self.expansion)
        self.dim_change = dim_change

    def forward(self,x):
        res = x.x
        data = x
        data.x = F.relu(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        data.x = F.relu(self.bn2(self.conv2(data.x, data.edge_index, data.edge_attr)))
        data.x = self.bn3(self.conv3(data.x, data.edge_index, data.edge_attr))

        if self.dim_change is not None:
            res = self.dim_change(res)

        data.x += res
        data.x = F.relu(data.x)
        return data.x

class ResNet(torch.nn.Module):
    def __init__(self,block, num_layers, classes=10):
        super(ResNet, self).__init__()
        #according to research paper
        self.input_planes = 64
        self.conv1 = SplineConv(3,64,dim=2, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.layer1 = self._layer(block,64,num_layers[0],stride=1)
        self.layer2 = self._layer(block,128,num_layers[1],stride=2)
        self.layer3 = self._layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 512, num_layers[3], stride=2)
        #self.averagePool = torch.nn.AvgPool2d(kernel_size=4,stride=1)
        self.fc = torch.nn.Linear(512*block.expansion,classes)

    def _layer(self,block,planes,num_layers,stride=1):
        dim_change = None
        if stride!=1 or planes != self.input_planes*block.expansion:
            dim_change = torch.nn.Sequential(SplineConv(self.input_planes,planes*block.expansion,dim=2,kernel_size=1),
                                             torch.nn.BatchNorm1d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes,planes,stride=stride,dim_change=dim_change))
        self.input_planes = planes*block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes,planes))

        return torch.nn.Sequential(*netLayers)

    def forward(self,x):
        data = x
        data.x = F.relu(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))

        data.x = self.layer1(data)
        data.x = self.layer2(data.x, data.edge_index, data.edge_attr)
        data.x = self.layer3(data.x, data.edge_index, data.edge_attr)
        data.x = self.layer4(data.x, data.edge_index, data.edge_attr)

        # clustering in Spline COnv
        cluster = voxel_grid(data.pos,data.batch, size=4)
        x = max_pool_x(cluster, data.x, data.batch, size=4)
        x = x.view(-1, self.fc.weight.size(1))
        x = self.fc(x)

        return x

def test():
        #To convert data from PIL to tensor
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        )

    #Load train and test set:
    '''train = torchvision.datasets.CIFAR10(root='./data_CIFAR',train=True,download=True,transform=transform)
    trainset = torch.utils.data.DataLoader(train,batch_size=128,shuffle=True)

    test = torchvision.datasets.CIFAR10(root='./data_CIFAR',train=True,download=True,transform=transform)
    testset = torch.utils.data.DataLoader(test,batch_size=128,shuffle=False)'''

    # with open("Data_after_pass through_graph.txt", 'rb') as pickleFile:
    #     graph_data = pickle.load(pickleFile)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load image and pass into Build pytorch graph
    image = Image.open("/home/sachin/Downloads/images.jpeg")
    image = image.resize((1024,1024))
    image = npy.array(image)
    image = npy.reshape(image,(-1,1024,1024,3))
    graph_data = geometric_graph(image)
    # e_att = graph_data.edge_attr
    e_att_t = torch.transpose(graph_data[0].edge_index, 0, 1)
    e_att_t = (e_att_t-torch.min(e_att_t))/((torch.max(e_att_t)-torch.min(e_att_t)))
    graph_data[0].edge_index = (graph_data[0].edge_index).type(torch.long)
    graph_data[0].edge_attr = e_att_t.float()
    train_loader = DataLoader(graph_data,batch_size=1)

    # resnet 18
    #net = ResNet(baseblock,[2,2,2,2],10)

    # REsNet 50
    net = ResNet(bottleNeck,[3,4,6,3])
    net.to(device)
    costFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.02,momentum=0.9)

    for graph_data in train_loader:
        graph_data = graph_data.to(device)
        prediction = net(graph_data)
        print("Fininsh")


'''for epoch in range(100):
        closs = 0
        for i,batch in enumerate(trainset,0):
            data,output = batch
            data,output = data.to(device),output.to(device)
            prediction = net(data)
            loss = costFunc(prediction,output)
            closs = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print every 100th time
            if i%100 ==0:
                print('[%d %d] loss: %.4f'%(epoch+1,i+1,closs/1000))
                closs=0

        correctHits=0
        total=0
        for batches in testset:
            data,output=batches
            data,output=data.to(device),output.to(device)
            prediction=net(data)
            _,prediction = torch.max(prediction.data,1)      #return max as well as its index
            total += output.size(0)
            correctHits += (prediction==output).sum().item()
        print('Accuracy on epoch', epoch+1, '= ',str((correctHits/total)*100))

    correctHits=0
    total=0
    for batches in testset:
        data,output = batches
        data,output= data.to(device),output.to(device)
        prediction=net(data)
        _,prediction= torch.max(prediction.data,1)  # return max as well as its index
        total += output.size(0)
        correctHits += (prediction==output).sum().item()
    print('Accuracy = '+str((correctHits/total)*100))'''

if __name__ == '__main__':
    test()





















