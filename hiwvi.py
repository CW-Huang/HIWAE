#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:30:39 2019

"""

# hierarchical importance weighted variational inference


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



class Gaussian(nn.Module):
    
    def __init__(self, dim, dimc, 
                 oper=nn_.ResLinear, realify=nn_.softplus):
        super(Gaussian, self).__init__()
        self.realify = realify
        self.dim = dim
        self.dimc = dimc
        self.mean = oper(dimc, dim)
        self.lstd = oper(dimc, dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        if isinstance(self.mean, nn_.ResLinear):
            self.mean.dot_01.scale.data.uniform_(-0.001, 0.001)
            self.mean.dot_h1.scale.data.uniform_(-0.001, 0.001)
            self.mean.dot_01.bias.data.uniform_(-0.001, 0.001)
            self.mean.dot_h1.bias.data.uniform_(-0.001, 0.001)
            self.lstd.dot_01.scale.data.uniform_(-0.001, 0.001)
            self.lstd.dot_h1.scale.data.uniform_(-0.001, 0.001)
            if self.realify == nn_.softplus:
                inv = np.log(np.exp(1-nn_.delta)-1) * 0.5
                self.lstd.dot_01.bias.data.uniform_(inv-0.001, inv+0.001)
                self.lstd.dot_h1.bias.data.uniform_(inv-0.001, inv+0.001)
            else:
                self.lstd.dot_01.bias.data.uniform_(-0.001, 0.001)
                self.lstd.dot_h1.bias.data.uniform_(-0.001, 0.001)
        elif isinstance(self.mean, nn.Linear):
            self.mean.weight.data.uniform_(-0.001, 0.001)
            self.mean.bias.data.uniform_(-0.001, 0.001)
            self.lstd.weight.data.uniform_(-0.001, 0.001)
            if self.realify == nn_.softplus:
                inv = np.log(np.exp(1-nn_.delta)-1) * 0.5
                self.lstd.bias.data.uniform_(inv-0.001, inv+0.001)
            else:
                self.lstd.bias.data.uniform_(-0.001, 0.001)
        
    def sample(self, context=None, n=None):
        ep0 = Variable(torch.zeros(n, self.dim).normal_()) 
        zero = Variable(torch.zeros(1))
        if cuda:
            ep0 = ep0.cuda()
            zero = zero.cuda()
        
        
        mean = self.mean(context)
        lstd = self.lstd(context)
        std = self.realify(lstd)
    
        z = mean + std * ep0
        logq0 = log_normal(z, mean, torch.log(std)*2).sum(1)
        
        return z, logq0


inv = np.log(np.exp(1)-1)
class Qnet(nn.Module):
    
    def __init__(self, dim1, dim2, dimh, dimc, k, oper=nn_.CWNlinear, 
                 actv=nn.ELU(), doubler=0, varred=0, wtype='ar'):
        """
            wtype: ar[arithmetic] or bh[balanced heuristic]
        """
        super(Qnet, self).__init__()
        # k: number of importance samples
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.dimh = dimh
        self.dimc = dimc
        self.k = k
        self.varred = varred
        
        self.z0 = Gaussian(dim1, dimc)
        
        self.hidn = oper(dim1, dimh, dimc)
        self.mean = oper(dimh, dim2*k, dimc)
        self.lstd = oper(dimh, dim2*k, dimc)
        self.skip = oper(dim1, dim2*k, dimc)
        self.actv = actv
        self.doubler = doubler
        self.wtype = wtype
        
    def sample(self, context=None, n=None):
        if context is None:
            assert n is not None, 'context and n cannot both be None'
            context = torch.ones(n, self.dimc)
        
        n = context.size(0)
        
        ep = Variable(torch.zeros(n, self.dim2*self.k).normal_()) 
        if cuda:
            ep = ep.cuda()
            context = context.cuda()
        z0, logq0 = self.z0.sample(context, n)
        logq0 = logq0.unsqueeze(1)
        
        
        skip = self.skip((z0, context))[0]  
        h = self.actv(self.hidn((z0, context))[0])
        mean = self.mean((h, context))[0] + skip
        lstd = self.lstd((h, context))[0]  
        std = F.softplus(lstd)
        
        ep = ep.view(n, self.k, self.dim2)
        mean = mean.view(n, self.k, self.dim2)
        lstd = lstd.view(n, self.k, self.dim2)
        std = std.view(n, self.k, self.dim2)
        z_ = mean + ep * std
        
        if self.wtype == 'ar' or self.wtype=='pi':
            if not self.doubler:
                logq = logq0 + log_normal(z_, mean, torch.log(std) * 2).sum(2)
            else:
                logq = logq0 + log_normal(
                    z_, mean.detach(), torch.log(std).detach() * 2).sum(2)
        elif self.wtype == 'bh':
            if not self.doubler:
                logq = logq0 + log_mean_exp(
                    log_normal(
                        z_.unsqueeze(2), 
                        mean.unsqueeze(1), 
                        torch.log(std).unsqueeze(1) * 2).sum(3), 2)[:,:,0]
            else:
                logq = logq0 + log_mean_exp(
                    log_normal(
                        z_.unsqueeze(2),
                        mean.detach().unsqueeze(1),
                        torch.log(std).unsqueeze(1).detach() * 2).sum(3), 2)[:,:,0]
        elif self.wtype[0] == 'l':
            p = float(self.wtype[1:]) # power
            if not self.doubler:
                logq = logq0 + log_normal(z_, mean, torch.log(std) * 2).sum(2)
                den = log_sum_exp(
                    log_normal(
                        z_.unsqueeze(2), 
                        mean.unsqueeze(1), 
                        torch.log(std).unsqueeze(1) * 2).sum(3) * p, 2)[:,:,0]
                nom = log_normal(z_, mean, torch.log(std) * 2).sum(2) * p
                logq = logq - (nom - den)
                
            else:
                logq = logq0 + log_normal(z_, mean.detach(), 
                                          torch.log(std).detach() * 2).sum(2)
                den = log_sum_exp(
                    log_normal(
                        z_.unsqueeze(2), 
                        mean.detach().unsqueeze(1), 
                        torch.log(std).detach().unsqueeze(1) * 2).sum(3) * p, 2)[:,:,0]
                nom = log_normal(z_, mean.detach(), 
                                 torch.log(std).detach() * 2).sum(2) * p
                logq = logq - (nom - den)            
                
            logq = logq - np.log(self.k)
        return z0, z_, logq
    
    
class Rnet(nn.Module):
    
    def __init__(self, dim1, dim2, dimh, dimc, k, oper=nn_.CWNlinear, 
                 actv=nn.ELU(), wtype='ar'):
        super(Rnet, self).__init__()
        """
            wtype: ar[arithmetic] or bh[balanced heuristic]
        """
        # k: number of importance samples
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.dimh = dimh
        self.dimc = dimc
        self.k = k
        
        if wtype == 'ar':
            self.function_emb = nn_.WNlinear(dimc, dimc*k)
        else:#elif wtype == 'bh' or wtype == 'pi':
            self.function_emb = nn_.WNlinear(dimc, dimc)
        
        self.hidn = oper(dim2, dimh, dimc)
        self.mean = oper(dimh, dim1, dimc)
        self.lvar = oper(dimh, dim1, dimc)
        self.skip = oper(dim2, dim1, dimc)
        self.actv = actv
        self.wtype = wtype
          
    def evaluate(self, z_, z0, context=None):
        n = z_.size(0)
        if context is None:
            context = torch.ones(n, self.dimc)        
        if cuda:
            context = context.cuda()
        
        if self.wtype == 'ar':
            context = self.function_emb(context).view(n, self.k, self.dimc)
        else:#elif self.wtype == 'bh' or self.wtype == 'pi':
            context = self.function_emb(context).view(n, 1, self.dimc)
            context = context.repeat((1,self.k,1))
            
        context = context.view(n*self.k, self.dimc)
        z_ = z_.view(n*self.k,self.dim2)
        
        skip = self.skip((z_, context))[0]
        h = self.actv(self.hidn((z_, context))[0])
        mean = self.mean((h, context))[0] + skip
        lvar = self.lvar((h, context))[0]
        
        mean = mean.view(n, self.k, self.dim1)
        lvar = lvar.view(n, self.k, self.dim1)     
        logr = log_normal(z0, mean, lvar).sum(2)
            
        return logr



if __name__ == '__main__':
    dim1, dim2, dimh, dimc, k = 2, 2, 100, 1, 3
    bs = 64
    lr = 0.001
    betas = (0.0,0.000)
    mode = 2
    if mode==1:
        doubler = 0
        varred = 1
    if mode==2:
        doubler = 1
        varred = 1
    if mode==3:
        doubler = 0
        varred = 1
    
    qnet = Qnet(dim1, dim2, dimh, dimc, k, oper=nn_.CWNlinear, 
        actv=nn.ELU(), doubler=doubler, varred=varred)
    rnet = Rnet(dim1, dim2, dimh, dimc, k, oper=nn_.CWNlinear, 
        actv=nn.ELU())
    optim_q = optim.Adam(qnet.parameters(), lr=lr, 
        betas=betas)
    optim_r = optim.Adam(rnet.parameters(), lr=lr, 
        betas=betas)
    

    L = []
    V = []
    for i in range(1,2000+1):
        optim_q.zero_grad()
        optim_r.zero_grad()
        
        if varred:
            z0, z_, logq, baseline = qnet.sample(n=bs)
        else:
            z0, z_, logq = qnet.sample(n=bs)
        logr = rnet.evaluate(z_, z0.unsqueeze(1))
        
        losses = logq - ef(z_.view(bs*k,2)).view(bs,k) - logr
        loss = (-log_mean_exp(-losses,1)).mean()
        L.append(loss.data.numpy())
        
        if mode==1:
            losses = -log_mean_exp(-losses,1) 
            loss = losses.mean()
            loss.backward()
            optim_q.step()
            optim_r.step()
            
            var = (torch.abs(losses-loss)**2).mean()
            V.append(var.data.numpy())
        
        if mode==3:
            losses = -log_mean_exp(-losses,1) 
            loss = losses.mean()
            loss.backward(retain_graph=True)
            
            var = (torch.abs(losses-baseline)**2).mean()
            V.append(var.data.numpy())
            (var*0.1).backward()
            
            optim_q.step()
            optim_r.step()
            
            
        if mode==2:
            norm = log_sum_exp(-losses,1)
            prob = torch.exp(- losses - norm).detach()
            torch.autograd.backward(losses, prob**1, retain_graph=True)
            optim_r.step()
            
            torch.autograd.backward(losses, prob**2-prob**1)
            optim_q.step()
    
            var = (torch.abs(-log_mean_exp(-losses,1)-loss)**2).mean()
            V.append(var.data.numpy())
            
        if i % 100 == 0:
            print i, loss
    
    plt.figure()
    plt.plot(L)
    plt.figure()
    plt.plot(V)
    
    n = 400
    # plot figure
    fig = plt.figure(figsize=(5*(2+k),5))
     
    
    # Target      
    ax = fig.add_subplot(1,2+k,1)
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
    ax = fig.add_subplot(1,2+k,2)
    if varred:
        z0, z_, logq, baseline = qnet.sample(n=n**2)
    else:
        z0, z_, logq = qnet.sample(n=n**2)
    logr = rnet.evaluate(z_, z0.unsqueeze(1))
    losses = logq - ef(z_.view(n**2*k,2)).view(n**2,k) - logr
    norm = log_sum_exp(-losses,1)
    ind = torch.multinomial(torch.exp(- losses - norm),1)
    
    data = torch.gather(
        z_,1,ind.unsqueeze(2).repeat((1,1,2))).squeeze(1).data.cpu().numpy()
    
    
    XX = data[:,0]
    YY = data[:,1]
    ax.hist2d(XX,YY,200,range=np.array([(-10, 10), (-10, 10)]), 
              cmap='plasma')
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.axis('off')
    plt.title('SIR', fontsize=25)
    
    
    # Proposals
    data0 = z0.data.cpu().numpy()
    XX0 = data0[:,0]
    YY0 = data0[:,1]
    color = (XX0**2+YY0**2)**0.5
    sc_n = 10000
    
    for i in range(1,k+1):
        ax = fig.add_subplot(1,2+k,2+i)
        data = z_[:,i-1].data.cpu().numpy()
        XX = data[:,0]
        YY = data[:,1]
        ax.scatter(XX[:sc_n],YY[:sc_n], s=3, 
                   c=color[:sc_n], alpha=0.2, cmap='hsv')
        plt.xlim((-10,10))
        plt.ylim((-10,10))
        plt.axis('off')
        plt.title('Proposal {}'.format(i), fontsize=25)
    
    plt.tight_layout()
    plt.savefig('test_hiwvi.png')

