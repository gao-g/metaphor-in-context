# metaphor-in-context
Code for [_Neural Metaphor Detection in Context_] TODO:Add URL

## Table of Contents
- [Repo Basics](#Basics)
- [Get Embeddings](#Embeddings)
- [Installation](#Installation)
- [Reproduce the Results](#Reproduction)
- [Reference](#Reference)

## Basics
Brief intro for each folder:

- corpora:contains raw datasets published online by researchers. can ignore.

- data: contains formatted version of each corpus. Details see notes in data repo.

- baseline: lexical baseline for MOH-X (10-fold cross-validation), TroFi and VUA.

- context: BiLSTM for classification task

- sequence: BiLSTM for sequence labeling task

## Embeddings
1. GloVe

Visit https://nlp.stanford.edu/projects/glove/, download glove.840B.300d.zip, and unzip it into a folder named glove.

2. ELMo

The ELMo data will release upon request by other means, since they are too large to be uploaded on github.

We have ELMo vectors for MOH-X, TroFi and VUA dataset with train/dev/test division. 

## Installation (TODO: details to be checked)
1. This project is developed in Python 3.6. Using Conda to set up a virtual enviroment is recommended.

2. Visit http://pytorch.org/, and install the PyTorch package. (TODO: add version here)

3. Install the required dependencies. (TODO: prepare requirements.txt)
    ```
    pip install -r requirements.txt
    ```

## Reproduction (TODO: details to be added)
1. classificaiton task

2. sequence labeling task


## Reference
(TODO: details to be added)
