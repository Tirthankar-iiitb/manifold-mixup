#!/usr/bin/env python
# coding: utf-8

# using frames for HIN Stage1

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

import pickle, os, random, io
import torch.nn.functional as F
import numpy as np
import pandas as pd

import datetime

from torch.utils.data import Dataset, DataLoader

def gethms(secs):
    mm, ss = divmod(secs, 60)
    hh, mm= divmod(mm, 60)
    return f'{int(hh):02d}:{int(mm):02d}:{ss:.2f}'
        
def gethms_timedelta(td):
    secs=td.total_seconds()
    return gethms(secs)

from dataloader_chars_indic_timit import Dataloader_chars_indic_timit
from dataloader_chars_indic_timit import stackup_inputs, collate_wrapper
from dataloader_chars_indic_timit import device


seed=25
torch.manual_seed(seed)
np.random.seed(seed)

FEATSIZE=1024

TRAIN_LANG='HIN'

class cross_entropy_loss():
    def __init__(self):
        super(cross_entropy_loss, self).__init__()
        self.eps=torch.tensor(np.finfo(float).eps)
    
    def loss(self,ypred,ytruth):
        cross_entropy = -torch.mean(ytruth * torch.log(ypred + self.eps))
        return cross_entropy

class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        self.fc=nn.Linear(input_size, output_size)
        #self.dropout=nn.Dropout(p=0.3, inplace=False)
        self.bnorm=nn.BatchNorm1d(output_size)
        self.relu=nn.ReLU(inplace=True)
        #self.residual = input_size == output_size    
        self.residual = False
                          
    def ops(self,x):
        x=self.fc(x)
        #x=self.dropout(x)   # <-- new
        x=self.bnorm(x.permute(0,2,1))
        x=self.relu(x.permute(0,2,1))
        return x
    
    def forward(self, x):
        if self.residual:
            return (self.ops(x) + x) / np.sqrt(2)
        return self.ops(x)

def mixup(x, shuffle, lam, i, j):
    if shuffle is not None and lam is not None and i == j:
        x = lam * x + (1 - lam) * x[:,shuffle,:]
    return x
    
class Mixup_Model(nn.Module):
    def __init__(self, num_classes, inputsize):
        super(Mixup_Model, self).__init__()
        self.sizes=[inputsize,512,128,32,2,32,128,512]
        self.numlayers=len(self.sizes)-1
        layers=[]
        for i in range(len(self.sizes)-1):
            layers.append(FCLayer(self.sizes[i],self.sizes[i+1]))
        self.layers=nn.ModuleList(layers)
        self.projection = nn.Linear(self.sizes[-1], num_classes)
        
    def forward(self, x):
        if isinstance(x, list):
            x, shuffle, lam = x
        else:
            shuffle = None
            lam = None

        taps=[]
        # Decide which layer to mixup
        j = np.random.randint(self.numlayers)
        for k in range(self.numlayers):
            x = mixup(x, shuffle, lam, k, j)
            taps.append(x[0].cpu())
            x = self.layers[k](x)
        taps.append(x[0])
        encout=x
        x = self.projection(x)
        taps.append(x[0])
        
        return x, encout, taps


def train(epoch, train_history, forcenomix=20, lastmixtaps=1):  
    #<-- forcenomix in % of nomix while training for mix training, lastmixtaps to tap last x%
    #model.train()
    tloss=[]
    vloss=[]
    vacc=[]
    bloss=[]
    itr_ctr=0
    tot_bloss=0.
    for batch_idx, (XS,yS,_) in enumerate(train_dataloader):
        rbeg=datetime.datetime.now()
        model.train()
        bs=XS.shape[0]
        XS=XS.contiguous().view(bs,-1,featsize).to(device)
        shuffle = torch.randperm(XS.shape[1])
        #mixup
        if resume_training:
            condition=True
        else:
            condition=(batch_idx > int(len(train_dataloader)*forcenomix/100))
        if TRAIN_MIX=='mix':
            if condition or epoch>0:
                alpha = 2
                lam = np.random.beta(alpha, alpha)
            else:
                shuffle=None
                lam=None
        else:
            shuffle=None
            lam=None

        output,_,_ = model([XS, shuffle, lam])
        output=nn.Softmax(dim=2)(output)

        if lam is not None:
            target=lam * yS + (1 - lam) * yS[:,shuffle,:]
        else:
            target=yS
        target=target.to(device)
        
        loss = celoss.loss(output, target)
        tot_bloss+=loss
        bloss.append(loss.detach().cpu().data.numpy())

        if (batch_idx+1) % report_interval == 0:
 
            itr_ctr+=1
            print(f'\t--> {itr_ctr}')
            
            optimizer.zero_grad()
            tot_bloss.backward()
            optimizer.step()
            tot_bloss=0.
            ls=np.mean(bloss)
            tloss.append(ls)
            bloss=[]
            
            eval_loss, eval_acc = evaluate()
            vloss.append(eval_loss)
            vacc.append(eval_acc)
            iter_cnt=f'{epoch+(itr_ctr*report_interval/tot_train_batch):.3f}'

            new_row={'epoch':[iter_cnt],'train loss':[f"{ls:.4f}"],'val loss':[f"{eval_loss:.4f}"], 'val acc':[f"{eval_acc*100:.2f}"]}
            if train_history is not None:
                df1=pd.DataFrame(new_row)
                train_history=pd.concat([train_history.loc[:],df1]).reset_index(drop=True)
            rend=datetime.datetime.now()
            etime=gethms_timedelta(rend-rbeg)
            print(f"\tepoch:{new_row['epoch'][0]}/{o_epochs+n_epochs}-[{etime}], train loss:{new_row['train loss'][0]}, val loss:{new_row['val loss'][0]}, val acc:{new_row['val acc'][0]}")
    
    trn_loss=np.mean(tloss)
    val_loss=np.mean(vloss)
    val_acc=np.mean(vacc)

    print(f'Train Epoch: {epoch+1}/{o_epochs+n_epochs}\tTrain Loss: {trn_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc*100:.2f}%,')

    return trn_loss, val_loss, val_acc, train_history

def evaluate():  #epoch, history=None
    loss = []
    acc = []
    
    all_labels=[]
    all_preds=[]
    
    with torch.no_grad():
        for batch_idx, (XS,yS,Xlbls) in enumerate(val_dataloader):
            bs=XS.shape[0]
            XS=XS.contiguous().view(bs,-1,featsize).to(device)
            output,_,_ = model(XS)
            target=yS.to(device)
            output=nn.Softmax(dim=2)(output)
            loss.append(celoss.loss(output, target).cpu().data)
            pred = output.data.max(2, keepdim=True)[1]

            acc.append(pred.eq(target.max(2, keepdim=True)[1].data.view_as(pred)).cpu().float().mean().numpy())
            for pr in pred.squeeze(2):
                predstr=[chardict[k.tolist()] for k in pr.cpu().data]
                all_preds.append(predstr)
            all_labels.extend(Xlbls)
    
    eval_loss=np.mean(loss)
    accuracy = np.mean(acc)
    
    return eval_loss, accuracy

old_epochs=0  #10
resume_training=False #True

pkldir='/root/manifold/experiments/new6b/pkldata_new6b'
execdir='/root/manifold/experiments/new6b'
print()

spe_mix=100000
spe_nomix=10000

for TRAIN_MIX in ['mix','nomix']:
    if TRAIN_MIX=='mix': continue
    msg=f'Stage1 {TRAIN_LANG}_{TRAIN_MIX} training'
    print(f'{msg}...\n')
    k=25; batch_size=8
    samplingperepoc=spe_mix if TRAIN_MIX=='mix' else spe_nomix
    train_pkl=f'{pkldir}/timit_support_set_{TRAIN_LANG}_960h_train_1.pkl'
    vocabjson=f'{pkldir}/nvocabs_960h.json'
    train_dataset=Dataloader_chars_indic_timit(kshot=k, support_pkl=train_pkl,                                   
            vocabjson=vocabjson,  
            samplingperepoch=samplingperepoc, 
            transform=transforms.Compose([stackup_inputs(),]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False,
            collate_fn=collate_wrapper)

    tot_train_batch=len(train_dataloader)
    report_interval=100 #1 #check <<------
    print(f'---> tot_train_batch: {tot_train_batch}, report_interval: {report_interval}')
    
    chardict=train_dataloader.dataset.chardict

    k=25; batch_size=4
    val_samplesperepoch=100

    val_pkl=f'{pkldir}/timit_support_set_{TRAIN_LANG}_960h_val_1.pkl'

    val_dataset=Dataloader_chars_indic_timit(kshot=k, support_pkl=val_pkl,
            vocabjson=vocabjson,  
            samplingperepoch=val_samplesperepoch,
            transform=transforms.Compose([stackup_inputs(),]))

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False, 
            collate_fn=collate_wrapper)

    classes=27; # 27 for Indic TIMIT English
    kshot=25
    featsize=FEATSIZE  
    n_epochs = 20  #160

    # define model
    
    mpath=f'{execdir}/models'
    model_desc=f'model_stage1_{TRAIN_LANG}_{TRAIN_MIX}'
    modelpath=f'{mpath}/{model_desc}.pth'
    best_modelpath=f'{modelpath[:-4]}_best.pth'
    chkp_modelpath=f'{modelpath[:-4]}_chkp.pth'
    
    model=Mixup_Model(classes,featsize).to(device)
    if resume_training:
        stage1modelpath=best_modelpath
        model.load_state_dict(torch.load(stage1modelpath))
        o_epochs = old_epochs
    else:
        o_epochs = 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    celoss=cross_entropy_loss()    

    train_history = pd.DataFrame()
    #history=None
    val_acc=0.
    train_start=datetime.datetime.now()
    for epoch in range(n_epochs):
        epoch_start=datetime.datetime.now()
        print(f'\t--> epoch: {o_epochs+epoch+1}/{o_epochs+n_epochs}')
        
        tloss,vloss,vacc,train_history=train(o_epochs+epoch,train_history)   #,forcenomix=5)
        
        epoch_end=datetime.datetime.now()
        print(f'\tEpoch {o_epochs+epoch+1} completed in time: {gethms_timedelta(epoch_end-epoch_start)}')
        if vacc > val_acc:
            val_acc=vacc
            bmpath=f'{best_modelpath}'
            torch.save(model.state_dict(), bmpath)
            bestepoch=o_epochs+epoch+1
        if epoch != 0 and (o_epochs+epoch+1) % 10 == 0:
            ckmpath=f'{chkp_modelpath[:-4]}_ep{o_epochs+epoch+1}.pth'
            torch.save(model.state_dict(), ckmpath)
            print(f'checkpoint model saved {ckmpath}')
        print()
    train_end=datetime.datetime.now()
    print(f'{model_desc} Training completed in time: {gethms_timedelta(train_end-train_start)}')

    #save model
    torch.save(model.state_dict(), modelpath)
    print(f'{model_desc} saved in:{modelpath}')
    print(f'Best {model_desc} saved in:{best_modelpath} at epoch - {bestepoch}')

    #save df for plots
    if train_history is None:
        pass
    else:
        loss_acc_fn=f'{execdir}/plots/loss_acc_{model_desc}.csv'
        if resume_training:
            train_history.to_csv(loss_acc_fn, encoding='utf-8', mode='a', index=False, header=False)
        else:
            train_history.to_csv(loss_acc_fn, encoding='utf-8', index=False)
        print(f'loss_acc saved in:{loss_acc_fn}')



    print('Done...')
    del model, optimizer
    print('--------------------------')
        
