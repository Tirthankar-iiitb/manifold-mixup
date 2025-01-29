#!/usr/bin/env python
# coding: utf-8

# Using stage1 model
# 

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

from manifold_dataloader_utts_indic_timit_accelerate import Dataloader_utts_indic_timit_xlsr_manifold
from manifold_dataloader_utts_indic_timit_accelerate import stackup_inputs, collate_wrapper
from manifold_dataloader_utts_indic_timit_accelerate import device

from jiwer import wer, cer
from pyctcdecode import build_ctcdecoder


# In[2]:


wer_metric=evaluate.load('wer')
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
        ntextlist.append(ntext)
    return ntextlist

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

# In[6]:


blank_label=0

def inference(loader,model):
    #model.eval()
    results={'pred_str':[],'text':[]}
    batch100_start=datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (X,y,txtlst) in enumerate(loader):
            #print(f'--> {batch_idx+1}/{len(loader)}',end='\r')
            ctcouts=model(X.to(device))

            ptxt=decoder.decode(ctcouts[0].cpu().numpy())
            if ptxt=='':
                continue
            
            results['pred_str'].append(ptxt)

            #results['pred_str'].extend([''.join([charmap[ch] for ch in pred_ids[0].tolist()])])
            gtxt=''.join([ch if ch != '|' else ' ' for ch in txtlst[0]])
            results['text'].append(gtxt)
            
    #print()

    cer=cer_metric.compute(predictions=results['pred_str'], references=results['text'])
    #wer=wer_metric.compute(predictions=results['pred_str'], references=results['text'])
    
    return cer #, wer

    
        


# ## load trained ctc model

# In[7]:
TRAINED_LANG='HIN'
classes=27
featsize=1024 #960h based

execdir=f'/root/manifold/experiments/new6b'
csrdir=f'{execdir}/models'

datadir=f'/root/Datasets/Indic_TIMIT_Data'

dsets=[]
batch_size=1
TEST_LANGS=['HIN','TAM','BEN','MLY','MAR','KAN']

for TEST_LANG in TEST_LANGS:
    dset_name=f'{datadir}/dataset_indic_timit_{TEST_LANG}'
    dsets.append(dset_name)
    
vocabjson=f'{datadir}/IndicTimit_vocab.json'    


#refmodel for dataloader
refmodeldir="facebook/wav2vec2-large-960h"

dataset1=Dataloader_utts_indic_timit_xlsr_manifold(dataset_file=dsets[0],
        category='test', vocabjson=vocabjson, refmodeldir=refmodeldir,
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


print()

mix_spe='100K'
nomix_spe='10K'
test_setup=f'Stage2_new6b_new_{mix_spe}_{nomix_spe}_best'
print(f'Inference on {test_setup} model...')

stage2models=[]

for i,dset_name in enumerate(dsets):
    for mixstate in ['mix','nomix']:
        if mixstate=='mix': continue
        print(f'Processing --> {dset_name}-{mixstate}')
        TEST_LANG=TEST_LANGS[i]
        
        dataset1=Dataloader_utts_indic_timit_xlsr_manifold(dataset_file=dset_name,
                category='test', vocabjson=vocabjson, refmodeldir=refmodeldir,
                transform=transforms.Compose([stackup_inputs(),]))
        loader = DataLoader(dataset1, batch_size=batch_size, pin_memory=False, collate_fn=collate_wrapper)
        
        if mixstate=='mix':
            
            ctc_model_desc=f'model_stage2_HIN_{mixstate}_new6b_new_{mix_spe}_{nomix_spe}'
        else:
            #model_stage2_HIN_nomix_new6b_new_100K_10K.pth
            ctc_model_desc=f'model_stage2_HIN_{mixstate}_new6b_new_{mix_spe}_{nomix_spe}'
        
        stage2models.append(f'{ctc_model_desc}.pth')
        ctc_modelpath=f'{csrdir}/{ctc_model_desc}.pth'
    
        ctc_model=Mixup_CTC_Model(classes,featsize).to(device)
        ctc_model.load_state_dict(torch.load(ctc_modelpath,map_location=device))
    
        test_start=datetime.datetime.now()
        #cer,wer = inference(loader,ctc_model)
        cer = inference(loader,ctc_model)
        test_end=datetime.datetime.now()
        et=(test_end-test_start).total_seconds()
        print(f'[Time:{gethms(et)}] Test Lang:{TEST_LANGS[i]}-{mixstate} CER:{cer*100:.2f}%') #, WER:{wer*100:.2f}%')         
        records.loc[i, 'testlang'] = TEST_LANG
        records.loc[i, f'{mixstate}_cer'] = f'{cer*100:.2f}'
    
        del loader, ctc_model
    print('-------------------------------')
    
results_fn=f'{execdir}/plots/inference_{test_setup}_nomix.csv'
records.to_csv(results_fn, encoding='utf-8', index=False, header=True)
print(f'stage2 HIN_nomix new6b inference results saved in:{results_fn}')
print(f'using stage2 models: {stage2models}')

    
            
# workon tirthoenv
# nohup python -u inference_stage2_new_new6b.py > logs/inference_stage2_new_new6b_100K_10K_nomix.log &
