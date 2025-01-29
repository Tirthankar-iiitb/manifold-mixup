#!/usr/bin/env python
# coding: utf-8
# 23 Jan 2024
# 

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
import soundfile as sf
from torch import nn

seed=25
torch.manual_seed(seed)
np.random.seed(seed)

device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

decoder_chars=['_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', \
               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', \
               't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
decoder = build_ctcdecoder(
            decoder_chars,
         )

class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        self.fc=nn.Linear(input_size, output_size)
        #self.dropout=nn.Dropout(p=0.3, inplace=False)
        self.bnorm=nn.BatchNorm1d(output_size)
        self.relu=nn.ReLU(inplace=True)
    
    def forward(self, x):
        x=self.fc(x)
        x=self.bnorm(x.permute(0,2,1))
        x=self.relu(x.permute(0,2,1))
        return x

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
        shuffle = None
        lam = None

        taps=[]
        taps.append(x[0])
        for k in range(self.numlayers):
            x = self.layers[k](x)
            taps.append(x[0])
        return x, taps

class Mixup_CTC_Model(nn.Module):
    def __init__(self, num_classes, inputsize):
        super(Mixup_CTC_Model, self).__init__()
        self.mixup_model=Mixup_Model(num_classes, inputsize)
        encoutsize=self.mixup_model.sizes[-1]
        self.ctc_block=FCLayer(encoutsize,num_classes+1)
        
    def forward(self, x):
        encout,_=self.mixup_model(x)
        x=self.ctc_block(encout)
        return x

class MixupModel_Inference(nn.Module):
    def __init__(self, ctc_modelpath,refmodeldir="facebook/wav2vec2-large-960h",device='cpu'):
        super(MixupModel_Inference, self).__init__()
        classes=27
        featsize=1024
        ctc_model=Mixup_CTC_Model(classes,featsize).to(device)
        ctc_model.load_state_dict(torch.load(ctc_modelpath,map_location=device))
        self.ctc_model=ctc_model.eval()
        
        feat_model = Wav2Vec2ForCTC.from_pretrained(refmodeldir).to(device)
        self.feat_model=feat_model.float()

    def getw2v2feat(self,wavfn):
        waveform, samplerate = sf.read(wavfn) #sr 16K
        waveform=torch.from_numpy(waveform)
        assert samplerate==16000
    
        #assert 1==2
        waveform=waveform.unsqueeze(0).float()
        #features
        fout=self.feat_model.wav2vec2.feature_extractor(waveform.to(device))
        #projection
        fpout=self.feat_model.wav2vec2.feature_projection(fout.permute(0,2,1))[0]
        #encoder out
        eout=self.feat_model.wav2vec2.encoder(fpout).last_hidden_state
        return eout[0]

    def transcribe(self,wavfn,**kwargs):
        feat=self.getw2v2feat(wavfn).unsqueeze(0)
        with torch.no_grad():
            ctcouts=self.ctc_model(feat.to(device))
    
            ptxt=decoder.decode(ctcouts[0].cpu().numpy())
            if ptxt=='':
                return({'text':'Error: Nothing decoded'})
            
            return({'text':ptxt})
