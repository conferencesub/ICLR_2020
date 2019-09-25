#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.get_data import load_benchmark, load_synthetic
from src.normalization import get_adj_feats
from src.args import get_args
from src.models import get_model
from src.utils import accuracy, LDA_loss
from src.plots import plot_feature
import torch.optim as optim
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


# In[2]:


# load dataset
# all tensor, dense
dataset_name = 'citeseer'
# dataset_name = input('input dataset name: cora/citeseer/pubmed/...')

adj, feats, labels, idx_train, idx_val, idx_test = load_benchmark(dataset_name)


# In[3]:


# get args
# model_name = input('choose model: GCN/SGC/GFNN/GFN/AGNN/GIN/...')
model_name = 'AGNN'
args = get_args(model_opt = model_name, dataset = dataset_name)

#add weight when train pre compute
weights = []
# In[4]:


# get input for model
adj, feats = get_adj_feats(adj = adj, feats = feats, model_opt = model_name, degree = args.degree, weights = weights)


# In[5]:


nb_class = (torch.max(labels) + 1).numpy()
Y_onehot =  torch.zeros(labels.shape[0], nb_class).scatter_(1, labels.unsqueeze(-1), 1)

nb_each_class_train = torch.sum(Y_onehot[idx_train], dim = 0)
nb_each_class_inv_train = torch.tensor(np.power(nb_each_class_train.numpy(), -1).flatten())
nb_each_class_inv_mat_train = torch.diag(nb_each_class_inv_train)

nb_each_class_val = torch.sum(Y_onehot[idx_val], dim = 0)
nb_each_class_inv_val = torch.tensor(np.power(nb_each_class_val.numpy(), -1).flatten())
nb_each_class_inv_mat_val = torch.diag(nb_each_class_inv_val)

nb_each_class_test = torch.sum(Y_onehot[idx_test], dim = 0)
nb_each_class_inv_test = torch.tensor(np.power(nb_each_class_test.numpy(), -1).flatten())
nb_each_class_inv_mat_test = torch.diag(nb_each_class_inv_test)


# In[6]:


# train, test


def train(epoch, model, optimizer, adj, feats, labels, idx_train, idx_val,           idx_test, Y_onehot, nb_each_class_inv_mat_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, fp1, fp2 = model(feats, adj)
    CE_loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    if model_name == 'AGNN':
        LDA_loss_train = LDA_loss(fp1[idx_train], Y_onehot[idx_train], nb_each_class_inv_mat_train, norm_or_not = False)
        loss_train = CE_loss_train - LDA_loss_train

    else:
        loss_train = CE_loss_train
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, fp1, fp2 = model(feats, adj)
    
    CE_loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = CE_loss_val
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    CE_loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = CE_loss_test
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return epoch+1, loss_train.item(), acc_train.item(), loss_val.item(),             acc_val.item(), loss_test.item(), acc_test.item(), time.time() - t,             



# In[7]:


def get_acc(adj, feats, labels, idx_train, idx_val, idx_test):

    # get model
    model = get_model(model_opt = model_name, nfeat = feats.size(1),                       nclass = labels.max().item()+1, nhid = args.hidden,                       dropout = args.dropout, cuda = args.cuda,                       dataset = dataset_name, degree = args.degree)
    # optimizer
    optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        if model_name!='AGNN' and model_name!='GIN':
            model.cuda()
            feats = feats.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()


    # Print model's state_dict    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor,"\t",model.state_dict()[param_tensor].size()) 
    print("optimizer's state_dict:")

    # Print optimizer's state_dict
    for var_name in optimizer.state_dict():
        print(var_name,"\t",optimizer.state_dict()[var_name])

    # # Print parameters
    # for name,param in model.named_parameters():
    #     print(name, param)


    training_log = []

    # Train model
    t_total = time.time()
    temp_val_loss = 999999
    temp_test_loss = 0
    temp_test_acc = 0
    PATH = "save/model_param/{}{}.pt".format(model_name, dataset_name)

    for epoch in range(args.epochs):

        epo, trainloss, trainacc, valloss, valacc, testloss, testacc, epotime = train(epoch, model,                                                                                       optimizer, adj, feats,                                                                                       labels, idx_train, idx_val,                                                                                      idx_test, Y_onehot,                                                                                       nb_each_class_inv_mat_train)
        training_log.append([epo, trainloss, trainacc, valloss, valacc, testloss, testacc, epotime])

        if valloss <= temp_val_loss:
            temp_val_loss = valloss
            temp_test_loss = testloss
            temp_test_acc = testacc
            torch.save(model.state_dict(), PATH)


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Best result:",
              "val_loss=",temp_val_loss,
                "test_loss=",temp_test_loss,
                 "test_acc=",temp_test_acc)
    bestmodel = torch.load(PATH)
    if model_name == 'AGNN':
        print("the weight is: ", torch.softmax(bestmodel['gc1.linear_weight'].data,dim=0))
        
    res_acc = temp_test_acc



    return temp_test_acc


# In[8]:


acc = []
for i in range(10):
    acc.append(get_acc(adj, feats, labels, idx_train, idx_val, idx_test))


# In[ ]:


acc = np.array(acc)
mean = acc.mean()
var = acc.var()
std = np.sqrt(var)
print("mean:",mean*100,"std:", std*100)


# In[ ]:





# In[ ]:




