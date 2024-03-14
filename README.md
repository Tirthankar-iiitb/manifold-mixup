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

For the experiments, there are 6 Accents chosen from Indic-TIMIT and 5 groups of accents from Common Voice. The Huggingface datasets for these are in the datasets folder.

