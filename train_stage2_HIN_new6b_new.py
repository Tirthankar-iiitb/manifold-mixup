#!/usr/bin/env python
# coding: utf-8

# # using 960h HIN Stage1_best model
# Mar 2024
# 

import torch,re
from torch import nn
from torch.optim import Adam
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms
from accelerate import Accelerator

import pickle, os, random, io
import torch.nn.functional as F
import numpy as np
import pandas as pd

import datetime

from torch.utils.data import Dataset, DataLoader

from datasets import DatasetDict
import evaluate
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Model

from dataloader_utts_indic_timit_1 import Dataloader_utts_indic_timit
from dataloader_utts_indic_timit_1 import stackup_inputs, collate_wrapper
from dataloader_utts_indic_timit_1 import device

# In[ ]:

cer_metric=evaluate.load('cer')


def get_audio(audfile):
    waveform, sample_rate = torchaudio.load(audfile)
    duration=waveform.shape[1]/sample_rate
    return waveform,sample_rate,duration

def gethms(secs):
    mm, ss = divmod(secs, 60)
    hh, mm= divmod(mm, 60)
    return f'{int(hh):02d}:{int(mm):02d}:{ss:.2f}'
        
playaudio = lambda wvf,wsr: display(pAudio(wvf, rate=wsr))
SAMPLERATE=16000

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined





# ## define model

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
    mixflag=False
    if shuffle is not None and lam is not None and i == j:
        x = lam * x + (1 - lam) * x[:,shuffle,:]
        mixflag=True
    return x,mixflag
    
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
        mixonce=False
        for k in range(self.numlayers):
            x,mflag = mixup(x, shuffle, lam, k, j)
            if mflag:
                mixonce=True
            if k==0:
                taps.append(x[0].cpu())
            
            x = self.layers[k](x)
            taps.append(x[0])
        encout=x
        x = self.projection(x)
        taps.append(x[0])
        
        return x, encout, taps, mixonce

class Mixup_CTC_Model(nn.Module):
    def __init__(self, num_classes, inputsize):
        super(Mixup_CTC_Model, self).__init__()
        
        self.mixup_model=Mixup_Model(num_classes, inputsize)
        encoutsize=self.mixup_model.sizes[-1]
        self.ctc_block=FCLayer(encoutsize,num_classes+1)
        
    def forward(self, x):
        x,encout,_,_=self.mixup_model(x)
        x=self.ctc_block(encout)
        return x


blank_label=0
ctc_loss=nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)

def train(epoch, history=None):
    #print(f'--> Train epoch: {epoch+1}')
    ctc_model.train()
    loss=[]
    results={'pred_str':[],'text':[]}
    batch100_start=datetime.datetime.now()
    for batch_idx, (X,y,txtlst) in enumerate(train_dataloader):
        #print(f'--->{X.shape}')
        ctcouts=ctc_model(X.to(device))
        log_probs=ctcouts.permute(1,0,2).log_softmax(2)
        targets=y.to(device)
        input_lengths=torch.tensor([X[i].shape[0] for i in range(X.shape[0])])
        target_sizes=torch.tensor([len(txt) for txt in txtlst])
        input_lengths=input_lengths.to(device)
        target_sizes=target_sizes.to(device)
        
        bloss_ctc=ctc_loss(log_probs, targets, input_lengths, target_sizes)

        optimizer.zero_grad()
        #accelerator.backward(bloss_ctc)
        bloss_ctc.backward()
        optimizer.step()
        
        bloss=bloss_ctc.detach().data.cpu().numpy()
        loss.append(bloss)
        
        logprobs=log_probs.permute(1,0,2)
        predtarget=greedy_decoder(logprobs[0])
        predtarget=re.sub('\|',' ',predtarget)
        results['pred_str'].extend([predtarget])

        txt=re.sub('\|',' ',txtlst[0])
        results['text'].extend([txt])
        
        if (batch_idx + 1) % 500 == 0:
            batch100_end=datetime.datetime.now()
            te=(batch100_end-batch100_start).total_seconds()
            print(f'\t--> [{gethms(te)}] {batch_idx + 1} loss: {np.mean(loss)}')
            batch100_start=datetime.datetime.now()

    tloss=np.mean(loss)
    cer=cer_metric.compute(predictions=results['pred_str'], references=results['text'])
    
    if history is not None:
        history.loc[epoch, 'train loss'] = tloss
        history.loc[epoch, 'train cer'] = cer*100
        
    return cer, tloss

def validate(epoch, history=None):
    print(f'--> Validation epoch: {epoch+1}')
    ctc_model.eval()
    loss=[]
    results={'pred_str':[],'text':[]}
    batch100_start=datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (X,y,txtlst) in enumerate(val_dataloader):
            ctcouts=ctc_model(X.to(device))
            log_probs=ctcouts.permute(1,0,2).log_softmax(2)
            targets=y.to(device)
            input_lengths=torch.tensor([X[i].shape[0] for i in range(X.shape[0])])
            target_sizes=torch.tensor([len(txt) for txt in txtlst])
            input_lengths=input_lengths.to(device)
            target_sizes=target_sizes.to(device)
            
            bloss_ctc=ctc_loss(log_probs, targets, input_lengths, target_sizes)     
            loss.append(bloss_ctc.detach().data.cpu().numpy())
            
            logprobs=log_probs.permute(1,0,2)
            predtarget=greedy_decoder(logprobs[0])
            predtarget=re.sub('\|',' ',predtarget)
            results['pred_str'].extend([predtarget])
    
            txt=re.sub('\|',' ',txtlst[0])
            results['text'].extend([txt])
    
            
            if (batch_idx + 1) % 500 == 0:
                batch100_end=datetime.datetime.now()
                te=(batch100_end-batch100_start).total_seconds()
                print(f'\t--> [{gethms(te)}] {batch_idx + 1} loss: {np.mean(loss)}')
                batch100_start=datetime.datetime.now()

    tloss=np.mean(loss)
    cer=cer_metric.compute(predictions=results['pred_str'], references=results['text'])
   
    if history is not None:
        history.loc[epoch, 'val_loss'] = tloss
        history.loc[epoch, 'val cer'] = cer*100
    return cer, tloss


classes=27
featsize=1024

TRAINED_LANG='HIN'

# dataloader
datadir=f'/root/Datasets/Indic_TIMIT_Data'
dset_name=f'{datadir}/dataset_indic_timit_{TRAINED_LANG}'

vocabjson=f'{datadir}/IndicTimit_vocab.json'

execdir=f'/root/manifold/experiments/new6b'
modeldir=f'{execdir}/models'
refmodel="facebook/wav2vec2-large-960h"

stage1Model_dir=f'{execdir}/models'

print()
spe_mix='100K'
spe_nomix='10K'

setup=f'new6b_new_{spe_mix}_{spe_nomix}'

for MODELMIXUP in ['mix','nomix']:
    if MODELMIXUP=='mix': continue
    print(f'--->> Traininng {setup}_{MODELMIXUP}')
    ctc_model=Mixup_CTC_Model(classes,featsize).to(device)
    
    if MODELMIXUP=='mix':
        #model_stage1_HIN_nomix_new6b_new_100K_10K_best.pth
        stage1modelpath=f'{stage1Model_dir}/model_stage1_HIN_{MODELMIXUP}_{setup}_best.pth'
    else:
        stage1modelpath=f'{stage1Model_dir}/model_stage1_HIN_{MODELMIXUP}_{setup}_best.pth'
    
    ctc_model.mixup_model.load_state_dict(torch.load(stage1modelpath,map_location=device))
    
    lr=1e-3
    optimizer = Adam(ctc_model.parameters(), lr=lr)
    
    if MODELMIXUP=='mix':
        ctc_model_desc=f'model_stage2_HIN_{MODELMIXUP}_new6b_new_{spe_mix}_{spe_nomix}'
    else:
        ctc_model_desc=f'model_stage2_HIN_{MODELMIXUP}_new6b_new_{spe_mix}_{spe_nomix}'
    
    ctc_modelpath=f'{modeldir}/{ctc_model_desc}.pth'
    csvfn=f'{execdir}/plots/loss_cer_{ctc_model_desc}.csv'

    batch_size=1
    
    dataset1=Dataloader_utts_indic_timit(dataset_file=dset_name, 
            category='train', vocabjson=vocabjson, refmodeldir=refmodel, 
            transform=transforms.Compose([stackup_inputs(),]))
    train_dataloader = DataLoader(dataset1, batch_size=batch_size, pin_memory=False, collate_fn=collate_wrapper)

    dataset2=Dataloader_utts_indic_timit(dataset_file=dset_name, 
            category='val', vocabjson=vocabjson, refmodeldir=refmodel, 
            transform=transforms.Compose([stackup_inputs(),]))
    val_dataloader = DataLoader(dataset2, batch_size=batch_size, pin_memory=False, collate_fn=collate_wrapper)

    charlst=dataset1.onlychars
    charmap=dataset1.ctc_charmap

    greedy_decoder = GreedyCTCDecoder(charlst)

    # train

    n_epochs=30

    history = pd.DataFrame()
    train_start=datetime.datetime.now()
    print(f'Train & validate for {n_epochs} epochs...')

    for epoch in range(n_epochs):
        epoch_start=datetime.datetime.now()
        tcer, tloss = train(epoch,history)
        vcer, vloss = validate(epoch,history)
        epoch_end=datetime.datetime.now()
        epoch_time=gethms((epoch_end-epoch_start).total_seconds())
        print(f'epoch: {epoch+1}/{n_epochs} [{epoch_time}] [cer,loss] [train: {tcer*100:.2f},{tloss:.4f}], [val: {vcer*100:.2f},{vloss:.4f}]')

    train_end=datetime.datetime.now()

    #save model
    torch.save(ctc_model.state_dict(), ctc_modelpath)
    print(f'Stage2: {ctc_model_desc} saved in:{ctc_modelpath}')

    print(f'Training time:{gethms((train_end-train_start).total_seconds())}')

    del ctc_model, optimizer, train_dataloader, val_dataloader

    #save df for plots
    if history is None:
        pass
    else:
        history.to_csv(csvfn, encoding='utf-8', index=False, header=True)
        print(f'loss_cer saved in:{csvfn}')
        
    print('---------------')
        

# @new8
# workon tirthoenv
# nohup python -u train_stage2_HIN_new6b_new.py > logs/train_stage2_HIN_new6b_new_100K_10K_nomix.log &
