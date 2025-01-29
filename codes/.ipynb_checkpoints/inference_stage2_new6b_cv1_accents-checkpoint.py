#!/usr/bin/env python
# coding: utf-8

# ctc inference: Inference on HIN trained stage2 model on grouped commonvoice accent and cv1-test
# ctc inference: 29 Feb

# cvgroups={'uk':['england','ireland'], 'oriental':['indian','malaysia'], 'northam':['us','canada'], \
#         'african':['african'], 'australian':['australia','newzealand']}

import torch
from torch import nn
from torch.optim import Adam
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms

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

from dataloader_utts_cv1_accent import Dataloader_utts_cv
from dataloader_utts_cv1_accent import stackup_inputs, collate_wrapper
from dataloader_utts_cv1_accent import device

from jiwer import wer, cer
from pyctcdecode import build_ctcdecoder


# In[2]:


#wer_metric=evaluate.load('wer')
cer_metric=evaluate.load('cer')

# In[4]:


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

import string
def get_clean_texts(textlist):
    ntextlist=[]
    for text in textlist:
        ntext=text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        ntext=clean_text(ntext)
        ntextlist.append(ntext)
    return ntextlist

def remove_chars(subj):
    repl_dict={'’': '', 'ó':'o', '–':'', 'á':'a', 'ú':'u', 'â':'a', 'é':'e', 'ñ':'n','ë':'e','“':'','”':'', '—':''}
    return subj.translate(str.maketrans(repl_dict))

chars_to_ignore_regex = '’–“”—'
def clean_text(txt):
    txt = remove_chars(txt)
    txt = re.sub(chars_to_ignore_regex, '', txt)  #.lower()
    return txt
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

class Mixup_CTC_Model(nn.Module):
    def __init__(self, num_classes, inputsize):
        super(Mixup_CTC_Model, self).__init__()
        
        self.mixup_model=Mixup_Model(num_classes, inputsize)
        encoutsize=self.mixup_model.sizes[-1]
        self.ctc_block=FCLayer(encoutsize,num_classes+1)
        
    def forward(self, x):
        x,encout,_=self.mixup_model(x)
        x=self.ctc_block(encout)
        return x

# In[6]:


blank_label=0

def inference_grcv(loader,model,accnt):
    #model.eval()
    #results={'pred_str':[],'text':[]}
    results={}

    batch100_start=datetime.datetime.now()
    tot=len(loader)
    with torch.no_grad():
        for batch_idx, (X,y,txtlst,_) in enumerate(loader):
            #print(f'--> {batch_idx+1}/{len(loader)}',end='\r')
            x=X[0].unsqueeze(0)
            ctcouts=model(x.to(device))

            ptxt=decoder.decode(ctcouts[0].cpu().numpy())
            if ptxt=='':
                continue
            
            try:
                results[accnt]['pred_str'].append(ptxt)
            except:
                results[accnt]={'pred_str':[],'text':[]}
                results[accnt]['pred_str'].append(ptxt)

            #results['pred_str'].extend([''.join([charmap[ch] for ch in pred_ids[0].tolist()])])
            gtxt=''.join([ch if ch != '|' else ' ' for ch in txtlst[0]])
            results[accnt]['text'].append(gtxt)
            #print(f'--> {batch_idx+1}/{tot}', end='\r')
            
    print()
    cers={}
    for ky in results.keys():
        cers[ky]=cer_metric.compute(predictions=results[ky]['pred_str'], references=results[ky]['text'])
   
    return cers

    
        


# ## load trained ctc model

TRAINED_LANG='HIN'
classes=27
featsize=1024

datadir=f'/root/Datasets/Indic_TIMIT_Data'
vocabjson=f'{datadir}/IndicTimit_vocab.json'

execdir=f'/root/manifold/experiments/new6b'
modeldir=f'{execdir}/models'

cvcdir='/root/manifold/datasets/CV1group'
dsets=[]
TEST_LANGS=['uk', 'oriental', 'northam', 'african', 'australian', 'cv1-test']
for TEST_LANG in TEST_LANGS:
    dset_name=f'{cvcdir}/CV1_gr_{TEST_LANG}_dataset'
    dsets.append(dset_name)
cv1_test_dset_name='/root/manifold/datasets/CV1_dataset'
dsets.append(cv1_test_dset_name)

#refmodel for dataloader
refmodel="facebook/wav2vec2-large-960h"
feat_desc='w2v2large960h_feat'
batch_size=1

dataset1=Dataloader_utts_cv(dataset_file=dsets[0],
        category='train', vocabjson=vocabjson, refmodeldir=refmodel,
        transform=transforms.Compose([stackup_inputs(),]))
charlst=dataset1.onlychars
charmap=dataset1.ctc_charmap
del dataset1

# ## ctc decoder
decoder_chars=charlst[:-1]+[' ']
decoder = build_ctcdecoder(
            decoder_chars,
            #kenlm_model_path=None,
            #kenlm_model_path="/root/langmodels/kannada/5gram_correct1.arpa",
         )

records = pd.DataFrame()

print()
testenv='stage2_HIN_1000K_10K_new6b_cv1'

for i,dset_name in enumerate(dsets):
    print('---------------------------')
    for mixstate in ['mix','nomix']:
        #if mixstate=='mix': continue

        test_setup=f'{testenv}_{mixstate}' 
        print(f'Processing --> {dset_name} -- Inference for {test_setup}...')
        
        TEST_LANG=TEST_LANGS[i]
        cat='test' if i==(len(dsets)-1) else 'train'
        
        dataset1=Dataloader_utts_cv(dataset_file=dset_name,
                category=cat, vocabjson=vocabjson, refmodeldir=refmodel,  # check on 'train' data for CV1 dataset else 'test'
                transform=transforms.Compose([stackup_inputs(),]))
        loader = DataLoader(dataset1, batch_size=batch_size, pin_memory=False, collate_fn=collate_wrapper)
        
        if mixstate=='mix':
            ctc_model_desc='model_stage2_HIN_mix_1000K_new6b'
        else:
            ctc_model_desc='model_stage2_HIN_nomix_new6b_10K_best'
            #ctc_model_desc='model_stage2_HIN_nomix_1000K_new6b'
            
        print(f'Processing --> {dset_name[-3:]}-{mixstate} -- Inference using model {ctc_model_desc}.pth')

        ctc_modelpath=f'{modeldir}/{ctc_model_desc}.pth'
    
        ctc_model=Mixup_CTC_Model(classes,featsize).to(device)
        ctc_model.load_state_dict(torch.load(ctc_modelpath,map_location=device))
    
        test_start=datetime.datetime.now()
        allcers = inference_grcv(loader,ctc_model,TEST_LANG)
        test_end=datetime.datetime.now()
        et=(test_end-test_start).total_seconds()
        print(f'[Time:{gethms(et)}] Test Lang:{TEST_LANGS[i]}')
        for ky in allcers.keys():
            print(f'\t{mixstate} {ky} --> CER:{allcers[ky]*100:.2f}%') 

        for j,ky in enumerate(allcers.keys()):
            records.loc[i,'cvgrp']= ky
            records.loc[i,mixstate] = f'{allcers[ky]*100:.2f}'

        del loader, ctc_model
print()
print('---------------------------')
results_fn=f'{execdir}/plots/inference_{testenv}.csv'
records.to_csv(results_fn, encoding='utf-8', index=False, header=True)
print(f'{test_setup} grouped cv1 inference results saved in:{results_fn}')
print('Done...')
            
            
# workon tirthoenv
# nohup python -u inference_stage2_new6b_grouped_cv1_accents.py > logs/inference_stage2_new6b_cv1_accents_1000K_10K.log &
