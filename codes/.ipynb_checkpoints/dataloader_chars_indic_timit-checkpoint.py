#!/usr/bin/env python
# coding: utf-8
# 

from datasets import DatasetDict
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
import torch, pickle, random,json,re, io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
import numpy as np

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAMPLERATE=16000
repl_sep = lambda s:re.sub('\|+', ' ', s)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class stackup_inputs(object):
    def __init__(self, kshot=25): #,featsz=1024):
        self.kshot=kshot
        #self.featsz=featsz
        
    def __call__(self, sample):
        support_set, kys = sample['sup'], sample['kys']
        XS=[];Xlbls=[]
        
        for i,chrname in enumerate(kys):
            supp=support_set[chrname]
            XS.append(supp)
            lbl=[chrname]*self.kshot
            Xlbls.extend(lbl)
        XS=torch.stack(XS)

        classes=XS.shape[0]
        yS=torch.nn.functional.one_hot(torch.arange(0, classes, 1 / self.kshot).long()).to(device)

        return XS,yS,Xlbls


class Dataloader_chars_indic_timit(Dataset):
    def __init__(self,  support_pkl='',  vocabjson='', kshot=25,
                 transform=None, samplingperepoch=500): 
        
        self.kshot=kshot
        self.transform=transform
        self.spe=samplingperepoch
        
        with open(support_pkl,'rb') as f:
            if device=='cpu':
                data=CPU_Unpickler(f).load()
            else:
                data=pickle.load(f)
        support_data=data[0]
        support_data_cnt=data[1]

        self.support_data=support_data
        self.support_data_cnt=support_data_cnt
                    
        self.support_keys=list(self.support_data.keys())
        self.support_keys.sort()
        
        chards={}
        for i,ch in enumerate(self.support_keys):
            chards[i]=ch
            chards[ch]=i
        self.chardict=chards
        
        charsamples=[len(self.support_data[ky]) for ky in self.support_keys]
        self.minlen=min(charsamples)
        
        with open(vocabjson,'r') as f:
            self.charmap=json.load(f)
        # "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>":
        onlychars=[ch for ch in self.charmap.keys() if ch not in ["<pad>", "<unk>","<s>","</s>","'"]]
        self.onlychars=['_']+[ch if ch !='|' else 'sil' for ch in onlychars]
        self.onlychars.sort()
        self.ctc_charmap={}
        for i,ch in enumerate(self.onlychars):
            self.ctc_charmap[i]=ch
            self.ctc_charmap[ch]=i
            
        self.w2v_charmap={}
        for ky in self.charmap.keys():
            self.w2v_charmap[ky]=self.charmap[ky]
            self.w2v_charmap[self.charmap[ky]]=ky
            
    def __len__(self):
        return self.spe
    
    def __getitem__(self, ix):
        sup_data={}
        for ch in self.support_keys:
            supsamples=self.support_data[ch]
            
            if len(supsamples) > self.kshot:
                idxs=torch.randperm(len(supsamples))[:self.kshot]
            else:
                idxs=torch.randint(len(supsamples), (self.kshot,))
            sup_data[ch]=supsamples[idxs]

        sample={'sup':sup_data,'kys':self.support_keys}
        if self.transform:
            sample = self.transform(sample)
        return sample

def collate_wrapper(batch):
    allXS=[];allyS=[];allXlbls=[]
    for i,item in enumerate(batch):
        XS,yS,Xlbls=item
        allXS.append(XS)
        allyS.append(yS)
        allXlbls.append(Xlbls)
    XS=torch.stack(allXS)
    yS=torch.stack(allyS)
    
    return XS,yS,allXlbls

