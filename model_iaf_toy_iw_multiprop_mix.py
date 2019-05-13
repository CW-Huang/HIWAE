#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:47:09 2018

"""



import numpy as np
import torch
cuda = torch.cuda.is_available()
if cuda:
    print 'using gpu'
else:
    print 'using cpu'

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchkit.utils import log_mean_exp, log_sum_exp, log_normal
from torchkit import nn as nn_
from torch.nn import functional as F

from torchkit.toy_energy import U2 as ef


inv = np.log(np.exp(1)-1)
class Transition(nn.Module):
    
    def __init__(self, dim, h, oper=nn_.WNlinear, 
                 gate=True, actv=nn.ELU(), cdim=0, doubler=0):
        super(Transition, self).__init__()
        
        if cdim:
            self.hidn = nn_.CWNlinear(dim, h, cdim)
        else:
            self.hidn = oper(dim,h)
        self.actv = actv
        self.mean = oper(h,dim)
        self.lstd = oper(h,dim)
        if gate:
            self.gate = oper(h,dim)
        self.ifgate = gate
        self.doubler = doubler
        self.reset_params()
        
    def reset_params(self):
        self.lstd.bias.data.zero_().add_(inv)
        if isinstance(self.lstd, nn_.WNlinear):
            self.lstd.scale.data.uniform_(-0.001,0.001)
        elif isinstance(self.lstd, nn.Linear):
            self.lstd.weight.data.uniform_(-0.001,0.001)
        if self.ifgate:
            self.gate.bias.data.zero_().add_(-1.0)
        
    def forward(self, z, z_targ=None, context=None):
        
        ep = Variable(torch.zeros(z.size()).normal_()) 
        if cuda:
            ep = ep.cuda()
        if context is not None:
            h = self.actv(self.hidn(z, context))
        else:
            h = self.actv(self.hidn(z))
        if self.ifgate:
            gate = torch.sigmoid(self.gate(h)) 
            mean = gate*(self.mean(h)) + (1-gate)*z
        else:
            mean = self.mean(h)
        lstd = self.lstd(h)
        std = F.softplus(lstd)
        z_ = mean + ep * std
        if z_targ is None:
            if not self.doubler:
                return z_, log_normal(z_, mean, torch.log(std) * 2).sum(1)
            else:
                return z_, log_normal(
                    z_, mean.detach(), torch.log(std).detach() * 2).sum(1)
        else:
            return z_, log_normal(z_targ, mean, torch.log(std) * 2).sum(1)
    



from hiwvi import Qnet, Rnet 
class model(object):
    
    def __init__(self, target_energy, iw=1, n=64, 
                 dimh=64, lr=0.001, doubler=0, dep=1, verbose=1, wtype='bh'):
        self.n = n
        self.iw = iw
        self.doubler = doubler
        self.dep = dep
        self.verbose = verbose
        
        self.qnet = Qnet(2, 2, dimh, 1, iw,
                      doubler=doubler,wtype=wtype)
        self.rnet = Rnet(2, 2, dimh, 1, iw,wtype=wtype)
        
        
        if cuda:
            self.qnet = self.qnet.cuda()
            self.rnet = self.rnet.cuda()
            
        self.optim_f = optim.Adam(self.qnet.parameters(), lr=lr, 
                betas=(0.9, 0.999))
        self.optim_b = optim.Adam(self.rnet.parameters(), lr=lr, 
                betas=(0.9, 0.999))
            
        
        self.target_energy = target_energy
        
    def train(self, iterations=5000):
        
        for it in range(iterations):
            
                self.optim_f.zero_grad()
                self.optim_b.zero_grad()
                
                data, spl, losses, norm, _ = self.sample(self.n)
                
                if not self.doubler:
                    losses = -log_mean_exp(-losses,1) 
                    loss = losses.mean()
                    loss.backward()
                    self.optim_f.step()
                    self.optim_b.step()
                else:
                    norm = log_sum_exp(-losses,1)
                    prob = torch.exp(- losses - norm).detach()
                    torch.autograd.backward(losses, prob**1, retain_graph=True)
                    self.optim_b.step()
                    
                    torch.autograd.backward(losses, prob**2-prob**1)
                    self.optim_f.step()
                
                
                if ((it + 1) % 100) == 0 and self.verbose:
                    loss = -log_mean_exp(-losses,1).mean()
                    print 'Iteration: [%4d/%4d] loss: %.8f' % \
                        (it+1, iterations, loss.data.cpu().item())
        
        
    def sample(self, n):
        if self.dep == 1:
            z0, z, logq = self.qnet.sample(n=n)
            logr = self.rnet.evaluate(z, z0.unsqueeze(1))
            z_ = z.view(n*self.iw,2)
            
        elif self.dep == 0:
            z0 = list()
            z = list()
            logq = list()
            for j in range(self.iw):
                z0_, z_, logq_ = self.qnet.sample(n=n)
                z0.append(z0_.unsqueeze(1))
                z.append(z_[:,j:j+1])
                logq.append(logq_[:,j:j+1])
            z0 = torch.cat(z0, 1)
            z = torch.cat(z, 1)
            z_ = z.view(n*self.iw,2)
            logq = torch.cat(logq, 1)
            logr = self.rnet.evaluate(z, z0)
        
        losses = logq - self.target_energy(z_).view(n, self.iw) - logr
        
        spl = z
            
        norm = log_sum_exp(-losses,1)
        ind = torch.multinomial(torch.exp(- losses - norm),1)
        data = torch.gather(
            spl.view(n,self.iw,2),
            1,ind.unsqueeze(2).repeat((1,1,2))).squeeze(1).data.cpu().numpy()
        return data, spl, losses, norm, z0
    
        
args = dict(iw = 3,
    n = 64,
    lr = 0.0005,
    doubler = 1,
    dimh = 64,
    dep = 1)
          
#def plot(args=args_):
if 1:
    iw = args['iw']
    n = args['n']
    lr = args['lr']
    doubler = args['doubler']
    dimh = args['dimh']
    dep = args['dep']
    
    # build and train
    mdl = model(ef, iw, n, dimh, lr, doubler, dep)
    mdl.train()
    
    
    n = 400
    # plot figure
    fig = plt.figure(figsize=(5*(2+iw),5))
     
    
    # Target      
    ax = fig.add_subplot(1,2+iw,1)
    x = np.linspace(-10,10,n)
    y = np.linspace(-10,10,n)
    xx,yy = np.meshgrid(x,y)
    X = np.concatenate((xx.reshape(n**2,1),yy.reshape(n**2,1)),1)
    X = X.astype('float32')
    X = Variable(torch.from_numpy(X))
    Z = ef(X).data.cpu().numpy().reshape(n,n)
    ax.pcolormesh(xx,yy,np.exp(Z), cmap='plasma')
    ax.axis('off')
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.title('target', fontsize=25)
    
    
    # SIR
    ax = fig.add_subplot(1,2+iw,2)
    data, spl, losses, _, z0 = mdl.sample(n**2)
    
    XX = data[:,0]
    YY = data[:,1]
    ax.hist2d(XX,YY,200,range=np.array([(-10, 10), (-10, 10)]), 
              cmap='plasma')
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.axis('off')
    plt.title('SIR', fontsize=25)
    
    
    # Proposals
#    data0 = z_.data.cpu().numpy()
    XX0 = z0[:,0].data.cpu().numpy()
    YY0 = z0[:,1].data.cpu().numpy()
    color = (XX0**2+YY0**2)**0.5
    color = color - color.mean()
    sc_n = 10000
    
    for i in range(1,iw+1):
        ax = fig.add_subplot(1,2+iw,2+i)
        data = spl.view(n**2,iw,2)[:,i-1].data.cpu().numpy()
        XX = data[:,0]
        YY = data[:,1]
        ax.scatter(XX[:sc_n],YY[:sc_n], s=3, 
                   c=color[:sc_n], alpha=0.2, cmap='hsv')
        plt.xlim((-10,10))
        plt.ylim((-10,10))
        plt.axis('off')
        plt.title('Proposal {}'.format(i), fontsize=25)
    
    plt.tight_layout()
    

def analyze(args=args):
    iw = args['iw']
    n = args['n']
    lr = args['lr']
    doubler = args['doubler']
    dimh = args['dimh']
    dep = args['dep']
    seed = args['seed']
    
    
    np.random.seed(seed)
    torch.manual_seed(seed+10000)
    
    
    # build and train
    mdl = model(ef, iw, n, dimh, lr, doubler, dep, verbose=1)
    mdl.train()
    
    
    # Analyze
    losses_ = list()
    z = torch.zeros(n**2, 2)
    if cuda:
        z = z.cuda()
    tf = mdl.Tf[-1]
    z, logq = tf(z)
    
    logq_ = logq
    zs = list()
    z_ = z
    for i in range(0,mdl.iw):
        if dep == 0:
            z = torch.zeros(n, 2)
            if cuda:
                z = z.cuda()
            tf = mdl.Tf[-1]
            z, logq = tf(z)
            logq_ = logq
            z_ = z
        tf = mdl.Tf[i]
        tb = mdl.Tb[i]
        z, logq = tf(z_)
        _, logr = tb(z,z_)
        losses_.append(
            (logq + logq_ - mdl.target_energy(z) - logr).unsqueeze(1))
        zs.append(z)
        
    losses = torch.cat(losses_, 1)
    
    
    logw = log_mean_exp(-losses,1)
    w = torch.exp(logw)
    logw, w = logw.data.cpu().numpy(), w.data.cpu().numpy()
    
    return logw.mean(), w.std(), w.var(), logw.std(), logw.var()

