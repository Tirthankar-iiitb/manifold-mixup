# Manifold-Mixup
This is a repo for cross-accented ASR by adapting the [Manifold Mixup](
https://doi.org/10.48550/arXiv.1806.05236) theory.

The network architecture consists of Stage-1 and Stage-2 as shown below-

<img src="https://github.com/Tirthankar-iiitb/manifold-mixup/blob/main/arch_schematic.png" width="30%" height="30%">

<!---
 ![schematic](https://github.com/Tirthankar-iiitb/manifold-mixup/blob/main/arch_schematic.png) 
 -->
## Data Preparation
Two datasets are used for the experiments - [Indic-TIMIT](https://doi.org/10.1109/O-COCOSDA46868.2019.9041230) and [Common Voice](
https://doi.org/10.48550/arXiv.1912.06670) for Indian English (speaking English TIMIT utterances by Indian L1 speakers) and English accents spanning various countries from New Zealand, Australia, Malaysia, India, Africa, US, Canada, UK, Ireland etc.

For the experiments, there are 6 Accents chosen from Indic-TIMIT and 5 groups of accents from Common Voice. The Huggingface datasets for these are in the datasets folder in the Google Drive link - [Manifold_Datasets](https://drive.google.com/drive/folders/1ghwxLAFrOnnPtr2YOwV5F8bAJHYqsTVR?usp=sharing).

### Bag of Frames (BoF) for Stage-1
A bag of frames for each character is prepared using [CTC-forced alignment](https://doi.org/10.1121/10.0008579) (Also refer to the [tutorial](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html) using the train and validation utterances of Hindi-Indic-TIMIT corpus. For each character and the separater (English Alphabets), there are 10,000 unique samples saved as a pickle file. These are also available in the Google Drive link as above.

### Stage-1 training
Stage-1 training using Cross-entropy loss using the BoFs for Hindi results into the Stage-1 models for 'mixup' and 'no-mixup' scenarios.

### Stage-2 training
The Stage-1 model weights are transferred to the Stage-2 model with a CTC-head. This is trained on the Hindi-utterances using CTC-loss function. 

### Inference
CER (Character Error Rates) are then calculated on the test corpus of Indic TIMIT (all six accents) and CV1 dataset. Separately, CV1 dataset is also grouped into 5 groups using the corresponding 'train' corpus (not seen by the model) as well as the CV1-test corpus. The CV1 groups are shown below-

{'uk':('england','ireland'), 'oriental':('indian','malaysia'), 'northam':('us','canada'), 'african':('african'), 'australian':('australia','newzealand')}

## The CER results
The results on CER for the above test sets show more than 2% absolute performance gain as shown the plots.



