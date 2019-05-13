#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:21:17 2018

"""

import numpy as np
import torch
cuda = torch.cuda.is_available()
if cuda:
    print 'using gpu'
else:
    print 'using cpu'


import torch.nn as nn
from torch.autograd import Variable
from torchkit import nn as nn_, helpers, utils, optim

import torch
import torch.utils.data as data
from torchkit.transforms import from_numpy, binarize
from torchvision.transforms import transforms
from torchkit import datasets
from torchkit.datasets import DatasetWrapper
from itertools import chain
import json
import argparse, os


from hiwvi import Qnet, Rnet, Gaussian


class VAE(helpers.Model):
    
    def __init__(self, args):

        self.args = args        
        self.__dict__.update(args.__dict__)
        
        dimz = args.dimz
        dimc = args.dimc
        dimh = args.dimh
        niw = args.niw


                 
        act = nn.ELU()
        
        self.enc = nn.Sequential(
            nn_.ResConv2d(1,16,3,2,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,32,3,2,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,2,padding=1,activation=act),
            act,
            nn_.Reshape((-1,32*4*4)),
            nn_.ResLinear(32*4*4,dimc),
            act
        )
        
        if self.mode == 'iwae':
            self.qnet = Gaussian(dimz, dimc)
            self.rnet = nn.ModuleList([])
        elif self.mode == 'hiwae':
            if self.dep in [0,1]:
                self.qnet = Qnet(dimz, dimz, dimh, dimc, niw,
                    doubler=args.doubler, varred=args.varred, 
                          wtype=args.wtype)
                self.rnet = Rnet(dimz, dimz, dimh, dimc, niw,
                    wtype=args.wtype)
            elif self.dep == 2:
                self.qnet = Qnet(dimz, dimz, dimh, dimc, 1,
                    doubler=args.doubler, varred=args.varred, 
                          wtype=args.wtype)
                self.rnet = Rnet(dimz, dimz, dimh, dimc, 1,
                    wtype=args.wtype)
        self.dec = nn.Sequential(
            nn_.ResLinear(dimz,dimc),
            act,
            nn_.ResLinear(dimc,32*4*4),
            act,
            nn_.Reshape((-1,32,4,4)),
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.slicer[:,:,:-1,:-1],                
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn_.ResConv2d(32,16,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
            act,
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn_.ResConv2d(16,1,3,1,padding=1,activation=act),
        )

        self.dec[-1].conv_01.bias.data.normal_(-3, 0.0001)

        if cuda:
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()            
            self.qnet = self.qnet.cuda()
            self.rnet = self.rnet.cuda()
        
        amsgrad = bool(args.amsgrad)
        polyak = args.polyak
        
        self.optim_enc = optim.Adam(
            chain(self.enc.parameters(),
                  self.qnet.parameters()),
            lr=args.lr, 
            betas=(args.beta1, args.beta2),
            amsgrad=amsgrad,
            polyak=polyak)
        self.optim_dec = optim.Adam(
            chain(self.dec.parameters(),
                  self.rnet.parameters()),
            lr=args.lr, 
            betas=(args.beta1, args.beta2),
            amsgrad=amsgrad,
            polyak=polyak)
            
        self.optim = helpers.MultiOptim(self.optim_enc, self.optim_dec)
        

        
    
    def loss(self, x):
        n = x.size(0)
        zero = utils.varify(np.zeros(1).astype('float32'))
        if cuda:
            zero = zero.cuda()
        context = self.enc(x)
        if self.mode == 'iwae':
            context = context.repeat(1,self.niw).view(n*self.niw,self.dimc)
            z, logq = self.qnet.sample(context, n*self.niw)
            logq = logq.view(n, self.niw)
            logr = 0
        elif self.mode == 'hiwae':
            if self.dep == 0:
                z0 = list()
                z = list()
                logq = list()
                for j in range(self.niw):
                    z0_, z_, logq_ = self.qnet.sample(context)
                    # z0_: batch_size x dimz
                    # z_: batch_size x niw x dimz
                    # logq_: batch_size x niw
                    z0.append(z0_.unsqueeze(1))
                    z.append(z_[:,j:j+1])
                    logq.append(logq_[:,j:j+1])
                z0 = torch.cat(z0, 1)
                z = torch.cat(z, 1)
                logq = torch.cat(logq, 1)
                logr = self.rnet.evaluate(z, z0, context)
            elif self.dep == 1:
                z0, z, logq = self.qnet.sample(context)
                logr = self.rnet.evaluate(z, z0.unsqueeze(1), context)
            elif self.dep == 2:
                """
                    iwae with hierarchical q; baseline
                """
                context = context.repeat(1,self.niw).view(n*self.niw,self.dimc)
                z0, z, logq = self.qnet.sample(context)
                logr = self.rnet.evaluate(z, z0.unsqueeze(1), context)
                logq = logq.view(n, self.niw)
                logr = logr.view(n, self.niw)
                
            z = z.view(n*self.niw,self.dimz)
        
        pi = nn_.sigmoid(self.dec(z))
        pi = pi.view(n, self.niw, *x.size()[1:])
        logpx = - utils.bceloss(pi, x.unsqueeze(1)).sum(2).sum(2).sum(2)
        logpz = utils.log_normal(z, zero, zero).sum(1).view(n, self.niw)
        
        return logpx, logpz, logq, logr
        
        
    def iwlb(self, x, niw=1):
        LOSSES = list()
        for i in range(niw):
            logpx, logpz, logq, logr = self.loss(x)
            loss = logq - logr - logpx - logpz 
            LOSSES.append(loss.data.cpu().numpy())
        return -utils.log_mean_exp_np(-np.concatenate(LOSSES, 1))
        
    def state_dict(self):
        return self.enc.state_dict(), \
               self.dec.state_dict(), \
               self.qnet.state_dict(), \
               self.rnet.state_dict()

    def load_state_dict(self, states):
        self.enc.load_state_dict(states[0]) 
        self.dec.load_state_dict(states[1])
        self.qnet.load_state_dict(states[2])
        self.rnet.load_state_dict(states[3])

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(chain(self.enc.parameters(),
                                      self.dec.parameters(),
                                      self.qnet.parameters(),
                                      self.rnet.parameters()),
                                self.clip)



class Trainer(helpers.Trainer):
    
    def __init__(self, args, filename, tr, va, te):
        
        self.__dict__.update(args.__dict__)

        self.filename = filename
        self.args = args        

        self.train_loader, self.valid_loader, self.test_loader = tr, va, te
        
        self.model = VAE(args)
        self.train_params = None
        
    def train(self):
        if self.train_params is None:
            self.train_params = {}
            self.train_params['t'] = 0
            self.train_params['e'] = 0
            self.train_params['best_val_epoch'] = 0
            self.train_params['best_val'] = float('inf')
            self.train_params['best_val_trn'] = float('inf')
            self.train_params['best_val_tst'] = float('inf')
        else:
            helpers.logging("Resuming training", self.filename)
            
        # Epoch loop
        for e in range(self.epoch):
            # Resume
            if e < self.train_params['e']:
                continue
            # Minibatch loop
            for x in self.train_loader:
                
                for qmany in range(self.qmany):
                    self.model.optim_enc.zero_grad()
                    self.model.optim_dec.zero_grad()
                    weight = min(1.0,
                        max(self.anneal0,
                            self.train_params['t']/float(self.anneal+1e-5)))
                    x = Variable(x).view(-1,1,28,28)
                    if cuda:
                        x = x.cuda()
                        
                    logpx, logpz, logq, logr = self.model.loss(x)
                    
                    kl = logq - logpz - logr
                    losses =  - (logpx - torch.max(
                        kl*weight, torch.ones_like(kl)*args.bits))
                    #losses = logq - logr - logpx - logpz 
                    losses_ = - utils.log_mean_exp(-losses)
                    
                    if self.doubler==0:
                        losses_.mean().backward()
                        self.model.clip_grad_norm()
                        self.model.optim_enc.step()
                        if qmany == self.qmany-1:
                            self.model.optim_dec.step()
                    elif self.doubler==1:
                        norm = utils.log_sum_exp(-losses,1)
                        prob = torch.exp(- losses - norm).detach()
                        torch.autograd.backward(losses, prob**1, retain_graph=True)
                        if qmany == self.qmany-1:
                            self.model.clip_grad_norm()
                            self.model.optim_dec.step()
                        
                        self.model.optim_enc.zero_grad()
                        torch.autograd.backward(losses, prob**2)
                        self.model.optim_enc.step()
                        
                    
                self.train_params['t'] += 1
            
            # Evaluate 
            self.model.optim_enc.swap()
            self.model.optim_dec.swap()
            loss_trn = self.evaluate(self.train_loader, 1)
            loss_val = self.evaluate(self.valid_loader, 1)
            loss_tst = self.evaluate(self.test_loader, 1)
            self.model.optim_enc.swap()
            self.model.optim_dec.swap()
            
            helpers.logging('Epoch: [%4d/%4d] train: %.2f ' \
                  'valid: %.3f test: %.3f; w: %.4f' % \
                (e + 1, self.epoch,
                 loss_trn,
                 loss_val,
                 loss_tst, weight), self.filename)
            
            # Save best val model
            if loss_val < self.train_params['best_val']:
                helpers.logging(' [^] Best validation loss [^] ... [saving]',
                                self.filename)
                self.save(self.filename + '/best')
                self.train_params['best_val'] = loss_val
                self.train_params['best_val_trn'] = loss_trn
                self.train_params['best_val_tst'] = loss_tst
                self.train_params['best_val_epoch'] = e+1
                
            # Save latest model
            self.train_params['e'] += 1
            self.save(self.filename + '/last')
            
            # Early stopping
            if self.impatient():
                helpers.logging('Terminating due to impatience ... \n', 
                                self.filename)
                break
            
        if self.final_mode==0:
            # loading best val model (early stopping)
            self.load(self.filename+'/best')

    def evaluate(self, dataloader, niw=1):
        LOSSES = 0 
        c = 0
        for x in dataloader:
            x = Variable(x).view(-1,1,28,28)
            if cuda:
                x = x.cuda()
                
            losses = self.model.iwlb(x, niw)
            LOSSES += losses.sum()
            c += losses.shape[0]
        return LOSSES / float(c)

#
# =============================================================================
# main
# =============================================================================


"""parsing and configuration"""
def parse_args():
    desc = "VAE"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='caltech101', 
                        choices=['sb_mnist', 
                                 'db_mnist',
                                 'db_omniglot',
                                 'caltech101'],
                        help='static/dynamic binarized mnist')
    parser.add_argument('--mode', type=str, default='hiwae',
                        choices=['iwae','hiwae'])
    parser.add_argument('--wtype', type=str, default='ar',
                        choices=['ar','bh','l1','l2','l3'])
    parser.add_argument('--epoch', type=int, default=400, 
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='The size of batch')
    parser.add_argument('--seed', type=int, default=1993,
                        help='Random seed')
    parser.add_argument('--to_train', type=int, default=1,
                        help='1 if to train 0 if not')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--anneal', type=int, default=0) # 50000
    parser.add_argument('--anneal0', type=float, default=0.0001)
    parser.add_argument('--bits', type=float, default=0.00) # 0.10
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--niw', type=int, default=2)
    parser.add_argument('--qmany', type=int, default=2)
    parser.add_argument('--varred', type=int, default=0)
    parser.add_argument('--doubler', type=int, default=0,
                        choices=[0,1],
                        help='double reparam')
    parser.add_argument('--dep', type=int, default=2,
                        choices=[0,1,2],
                        help='dependency across samples')
    parser.add_argument('--polyak', type=float, default=0.00) # 0.995
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--final_mode', type=int, default=0,
                        choices=[0,1])
    
    parser.add_argument('--dimz', type=int, default=32)
    parser.add_argument('--dimc', type=int, default=100)
    parser.add_argument('--dimh', type=int, default=100)
    
    parser.add_argument('--report', type=str, default='report_model_hiwae')
    
    
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""getting dataset"""
def get_dataset(args,root='dataset'):
    if args.dataset == 'sb_mnist':
        tr, va, te = datasets.load_bmnist_image(root)
    elif args.dataset == 'db_mnist':
        tr, va, te = datasets.load_mnist_image(root)
    elif args.dataset == 'db_omniglot':
        tr, va, te = datasets.load_omniglot_image(root)
    elif args.dataset == 'caltech101':
        tr, va, te = datasets.load_caltech101_image(root)    
    else:
        raise Exception('dataset {} not supported'.format(args.dataset))
    
    if args.final_mode==1:
        tr = np.concatenate([tr, va], axis=0)
        va = te[:]
    
    if args.dataset[:2] == 'db':
        compose = transforms.Compose([from_numpy(), binarize()])
        tr = DatasetWrapper(tr,compose)
        va = DatasetWrapper(va,compose)
        te = DatasetWrapper(te,compose)
        
    train_loader = data.DataLoader(tr, 
                                   batch_size=args.batch_size,
                                   shuffle=True)
    valid_loader = data.DataLoader(va, 
                                   batch_size=args.batch_size,
                                   shuffle=False)
    test_loader = data.DataLoader(te, 
                                  batch_size=args.batch_size,
                                  shuffle=False)
    return train_loader, valid_loader, test_loader
    
"""run"""
#def main():
if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed+10000)
    

    droot, sroot, spath = helpers.getpaths(save_folder='model_hiwae')
    
    print(' [*] Loading dataset {}'.format(args.dataset))
    tr, va, te = get_dataset(args, droot)
    
    model_name = 'hiwae:{}:e{}:{}:niw{}:dp{}:db{}:b{}:s{}:' \
        'lr{}:cl{}:a{}:bt{}:am{}:pl{}:dzch{}-{}-{}:f{}'.format(
        args.dataset, args.epoch, 
        args.mode, args.niw, 
        args.dep, args.doubler,
        args.batch_size, args.seed,
        args.lr, args.clip, args.anneal, args.bits, args.amsgrad, 
        args.polyak, args.dimz, args.dimc, args.dimh,
        args.final_mode)
    if args.wtype != 'ar':
        model_name += ':{}'.format(args.wtype)
    if args.qmany>1:
        model_name += ':q{}'.format(str(args.qmany))
        
    helpers.create(spath, model_name)
    fn = spath+'/'+model_name
    helpers.logging(str(args), fn)
    helpers.logging(' [*] spath: {}'.format(spath), fn)
    helpers.logging(' [*] model name: {}'.format(model_name), fn)
    if cuda:
        helpers.logging('using gpu', fn)
    else:
        helpers.logging('using cpu', fn)
    

    
    helpers.logging(" [*] Building model!", fn)
    old_fn = fn+'/last'
    old_arg_fn = old_fn+'_args.txt'
    if os.path.isfile(old_arg_fn):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        d = without_keys(json.loads(open(old_arg_fn,'r').read()),
                         ['to_train','epoch','anneal'])
        args.__dict__.update(d)
        helpers.logging(" New args:" , fn)
        helpers.logging(str(args), fn)
        mdl = Trainer(args, fn, tr, va, te)
        helpers.logging(" [*] Loading model!", fn)
        mdl.load(old_fn)
    else:
        mdl = Trainer(args, fn, tr, va, te)
    
    # launch the graph in a session
    if args.to_train:
        helpers.logging(" [*] Training started!", fn)
        mdl.train()
        helpers.logging(" [*] Training finished!", fn)

    
    mdl.model.optim_enc.swap()
    mdl.model.optim_dec.swap()
    nll_val = mdl.evaluate(mdl.valid_loader, 2000)
    nll_tst = mdl.evaluate(mdl.test_loader, 2000)
    mdl.model.optim_enc.swap()
    mdl.model.optim_dec.swap()
    helpers.logging(" [**] Valid: %.4f" % nll_val, fn)
    helpers.logging(" [**] Test: %.4f" % nll_tst, fn)
    
    helpers.logging(" [*] Testing finished!", fn)
    
    
    with open('{}.txt'.format(args.report), 'a') as out:
        out.write(str(args.__dict__.items())+'\n'+\
                  fn+'\n'+\
                  str([nll_val, nll_tst])+'\n'+\
                  str(mdl.train_params)+'\n')





