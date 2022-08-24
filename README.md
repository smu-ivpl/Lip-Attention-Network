# Lip-Attention Network (LAN)

---------
This project tried to improve the lip recognition performance by designing the attention module and applying it to improve the lipreading network.

It was developed based on the [LipNet](https://github.com/rizkiarm/LipNet).



## Model

<center> <img src="./assets/Lip-Attention%20Network.png" alt="{{ include.description }}">
  <figcaption style="text-align:center; font-size:16px;> Fig1. Architecture of Lip-Attention Network.
  </figcaption>
  </center> 
<br>

<center> <img src="./assets/RCAB.png" alt="{{ include.description }}">
  <figcaption style="text-align:center; font-size:16px;> Fig2. RCAB
  </figcaption>
  </center> 
<br>

You can check the architecture of 'Lip-Attention Network' in 'RG1_RCAB10.txt'.


## Results

----

|          Scenario           |   samples    | Epoch |  CER   |  WER   |  BLEU  |
|:---------------------------:|:------------:|:-----:|:------:|:------:|:------:|
|       Original LipNet       | 3964 (whole) |  149  | 12.21% | 19.10% | 81.56% |
| Lip-Attention Network (LAN) | 3964 (whole) |  149  | 8.02%  | 14.23% | 84.02% |


## Installation
To use the model, first you need to clone the repository:
```
git clone https://github.com/smu-ivpl/LAN
```
Then you can install the package:
```
cd LAN/

pip install -r requirements.txt

```
**Note:** if you don't want to use CUDA, you need to edit the ``requirements.txt`` and change ``tensorflow-gpu`` to ``tensorflow``

## Dataset

---
This model uses GRID corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)

## Pre-trained weights

----
You can download and use the weights provided here: https://github.com/Han-lim/Lip-Attention-Network/tree/master/evaluation/models. 

## Get started

----
1. Download all datasets from the [GRID Corpus website](http://spandh.dcs.shef.ac.uk/gridcorpus/).
2. Extracts all the videos and aligns.
3. Create ``datasets`` folder on ``training/unseen_speakers`` folder.
4. All current ``train_XXX.py`` expect the videos to be in the form of 100x50px mouthcrop image frames.
5. The other way would be to extract the mouthcrop image using ``scripts/extract_mouth_batch.py`` (usage can be found inside the script).
6. Create symlink from each ``training/*/datasets/align`` to your align folder.

## Train

----
First, save the validation datasets (``s1``, ``s2``, ``s20``, and ``s22``) in the val folder, and the rest of datasets in the train folder.

Then, create symlink from ``training/unseen_speakers/datasets/[train|val]`` to your selection of ``[train|val]`` inside of the video dataset folder.

Train the model using the following command:

```
./train unseen_speakers [GPUs (optional)]
```

## Evaluate

To evaluate and visualize the trained model on a single video / image frames, you can execute the following command:
```
./predict [path to weight] [path to video]
```
**Example:**
```
./predict evaluation/models/unseen-weights126.h5 evaluation/samples/id2_vcd_swwp2s.mpg
```

## Acknowledgement

---
Many thanks to the excellent open source projects:
- [LipNet](https://github.com/rizkiarm/LipNet)
