# metaphor-in-context
Code for [Neural Metaphor Detection in Context](https://arxiv.org/pdf/1808.09653.pdf).

## Table of Contents
- [Repo Basics](#Basics)
- [Get Embeddings](#Embeddings)
- [Installation](#Installation)
- [Reproduce the Results](#Reproduction)
- [Reference](#Reference)

## Basics
Brief intro for each folder:

- corpora: contains raw datasets published online by researchers. can ignore.

- data: contains formatted version of each corpus. Check notes in data folder for details.

- baseline: lexical baseline for MOH-X, TroFi, and VUA dataset.

- classification: BiLSTM for verb classification task.

- sequence: BiLSTM for sequence labeling task.

## Embeddings
1. GloVe

Visit https://nlp.stanford.edu/projects/glove/, download glove.840B.300d.zip, and unzip it into a folder named glove.

2. ELMo

The ELMo data will release upon request by other means, since they are too large to be uploaded on github.

We have ELMo vectors for MOH-X, TroFi, and VUA dataset with train/dev/test division. 

## Installation
1. This project is developed in Python 3.6. Using Conda to set up a virtual environment is recommended.

2. Install the required dependencies. 
    ```
    pip install -r requirements.txt
    ```
    
3. Install PyTorch from http://pytorch.org/.


## Reproduction

1. classificaiton task (classification model): Check the main_XXX.py in the folder "classification".

2. sequence labeling task (sequence labeling model): Check the main_XXX.py in the folder "sequence".

Overall guideline:

- main_XXX.py is the training and testing script for classification or labeling model on dataset XXX. 

- All main_XXX.py shares the variable naming convection and similar code structure.

- Directly run main_XXX.py would train a model on dataset XXX, and report the performance during training and the final test performance without early stop.

- Default GPU usage is True. Change using_GPU to False if not using GPU.

- To try different sets of hyperparameters, please check comments for details.


## Reference
```
@InProceedings{gao18nmd,
  author    = {Ge Gao, Eunsol Choi, Yejin Choi, and Luke Zettlemoyer},
  title     = {Neural Metapor Detection in Context},
  booktitle = {EMNLP},
  year      = {2018}
}
```
