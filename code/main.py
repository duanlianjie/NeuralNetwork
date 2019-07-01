#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import mynetwork
import data


# In[3]:


# 训练集：测试集 = 6：4
train_datas, test_datas = data.data_loader()
num_of_test = len(test_datas)
# 基础参数,下面的对比说明为什么选择这些作为基本参数
layers = [22,50,100,2]
epochs = 30
mini_batch = 10
eta = 0.5
lmbda = 0.1


# In[4]:


net_one = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
net_one.large_weight_initializer()


# In[5]:


accuracy = net_one.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,                       monitor_evaluation_accuracy = True)


# In[6]:


def make_plot(epochs, ls1, ls2, label1, label2, title):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,epochs+1,1), ls1, 'rx--', color = '#2A6EA6', label = label1)
    ax.plot(np.arange(1,epochs+1,1), ls2, 'bo:', color = '#FFA933', label = label2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    plt.legend(loc = 'best')
    plt.savefig(title)
    plt.show()


# In[7]:


# 对比1 分别使用交叉熵和二次代价函数
def compare_cost():
    net1 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net2 = mynetwork.Network(layers, cost=mynetwork.CrossEntropyCost)
    net1.large_weight_initializer()
    net2.large_weight_initializer()
    
    evaluation_accuracy1 = net1.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy2 = net2.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    make_plot(epochs, evaluation_accuracy1, evaluation_accuracy2,              "Evaluation_accuracy_Quadratic", "Evaluation_accuracy_CrossEntropy",              "CompareofCost")


# In[8]:


compare_cost()


# In[9]:


# 对比2 隐藏层的对比
def compare_layers():
    layers1 = [22,30,2]
    layers2 = [22,50,2]
    layers3 = [22,30,100,2]
    layers4 = [22,50,100,2]
    net1 = mynetwork.Network(layers1, cost=mynetwork.QuadraticCost)
    net2 = mynetwork.Network(layers2, cost=mynetwork.QuadraticCost)
    net3 = mynetwork.Network(layers3, cost=mynetwork.QuadraticCost)
    net4 = mynetwork.Network(layers4, cost=mynetwork.QuadraticCost)
    net1.large_weight_initializer()
    net2.large_weight_initializer()
    net3.large_weight_initializer()
    net4.large_weight_initializer()

    evaluation_accuracy1 = net1.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy2 = net2.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy3 = net3.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy4 = net4.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy1, 'x-', color='red', label = '[22,30,2]')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy2, 'o-', color='blue', label = '[22,50,2]')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy3, 'x-', color='green', label = '[22,30,100,2]')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy4, 'o-', color='yellow', label = '[22,50,100,2]')
    ax.set_xlim([1,31])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('CompareofLayers')
    plt.legend(loc='best')
    plt.savefig('CompareofLayers')
    plt.show()


# In[10]:


compare_layers()


# In[11]:


# 对比2 隐藏层的对比---局部图
def compare_layers():
    layers1 = [22,30,2]
    layers2 = [22,50,2]
    layers3 = [22,30,100,2]
    layers4 = [22,50,100,2]
    net1 = mynetwork.Network(layers1, cost=mynetwork.QuadraticCost)
    net2 = mynetwork.Network(layers2, cost=mynetwork.QuadraticCost)
    net3 = mynetwork.Network(layers3, cost=mynetwork.QuadraticCost)
    net4 = mynetwork.Network(layers4, cost=mynetwork.QuadraticCost)
    net1.large_weight_initializer()
    net2.large_weight_initializer()
    net3.large_weight_initializer()
    net4.large_weight_initializer()

    evaluation_accuracy1 = net1.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy2 = net2.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy3 = net3.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy4 = net4.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(18,epochs+1,1), evaluation_accuracy1[-13:], 'x-', color='red', label = '[22,30,2]')
    ax.plot(np.arange(18,epochs+1,1), evaluation_accuracy2[-13:], 'o-', color='blue', label = '[22,50,2]')
    ax.plot(np.arange(18,epochs+1,1), evaluation_accuracy3[-13:], 'x-', color='green', label = '[22,30,100,2]')
    ax.plot(np.arange(18,epochs+1,1), evaluation_accuracy4[-13:], 'o-', color='yellow', label = '[22,50,100,2]')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('CompareofLayers')
    plt.legend(loc='best')
    plt.savefig('CompareofLayers2')
    plt.show()


# In[12]:


compare_layers()


# In[13]:


# 对比3 小批次大小
def compare_mini_batch():
    batch1 = 10
    batch2 = 20
    batch3 = 100
    batch4 = 5
    net1 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net2 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net3 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net4 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net1.large_weight_initializer()
    net2.large_weight_initializer()
    net3.large_weight_initializer()
    net4.large_weight_initializer()

    evaluation_accuracy1 = net1.SGD(train_datas, epochs, batch1, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy2 = net2.SGD(train_datas, epochs, batch2, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy3 = net3.SGD(train_datas, epochs, batch3, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy4 = net4.SGD(train_datas, epochs, batch4, eta, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy1, 'x-', color='red', label = 'mini_batch=10')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy2, 'o-', color='blue', label = 'mini_batch=20')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy3, 'x-', color='green', label = 'mini_batch=100')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy4, 'o-', color='black', label = 'mini_batch=5')
    
    ax.set_xlim([1,31])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('CompareofMinibatch')
    plt.legend(loc='best')
    plt.savefig('CompareofMinibatch')
    plt.show()


# In[14]:


compare_mini_batch()


# In[15]:


# 对比四 学习率
def compare_eta():
    eta1 = 0.05
    eta2 = 0.5
    eta3 = 10
    eta4 = 50
    net1 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net2 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net3 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net4 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net1.large_weight_initializer()
    net2.large_weight_initializer()
    net3.large_weight_initializer()
    net4.large_weight_initializer()

    evaluation_accuracy1 = net1.SGD(train_datas, epochs, mini_batch, eta1, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy2 = net2.SGD(train_datas, epochs, mini_batch, eta2, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy3 = net3.SGD(train_datas, epochs, mini_batch, eta3, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy4 = net4.SGD(train_datas, epochs, mini_batch, eta4, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy1, 'x-', color='red', label = 'eta=0.05')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy2, 'o-', color='blue', label = 'eta=0.5')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy3, 'x-', color='green', label = 'eta=10')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy4, 'o-', color='yellow', label = 'eta=50')
    
    ax.set_xlim([1,31])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('CompareofEta')
    plt.legend(loc='best')
    plt.savefig('CompareofEta')
    plt.show()


# In[16]:


compare_eta()


# In[17]:


# 对比五 正则化参数
def compare_lmbda():
    lmbda1 = 0.1
    lmbda2 = 0.5
    lmbda3 = 5
    lmbda4 = 50
    net1 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net2 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net3 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net4 = mynetwork.Network(layers, cost=mynetwork.QuadraticCost)
    net1.large_weight_initializer()
    net2.large_weight_initializer()
    net3.large_weight_initializer()
    net4.large_weight_initializer()

    evaluation_accuracy1 = net1.SGD(train_datas, epochs, mini_batch, eta, lmbda = lmbda1, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy2 = net2.SGD(train_datas, epochs, mini_batch, eta, lmbda = lmbda2, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy3 = net3.SGD(train_datas, epochs, mini_batch, eta, lmbda = lmbda3, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    evaluation_accuracy4 = net4.SGD(train_datas, epochs, mini_batch, eta, lmbda = lmbda4, evaluation_data = test_datas,               monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy1, 'x-', color='red', label = 'lmbda=0.1')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy2, 'o-', color='red', label = 'lmbda=0.5')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy3, 'x-', color='green', label = 'lmbda=5')
    ax.plot(np.arange(1,epochs+1,1), evaluation_accuracy4, 'o-', color='yellow', label = 'lmbda=50')
    
    ax.set_xlim([1,31])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Compareoflmbda')
    plt.legend(loc='best')
    plt.savefig('Compareoflmbda')
    plt.show()


# In[18]:


compare_lmbda()

