# metaphor-in-context
Code for [_Neural Metaphor Detection in Context_](https://arxiv.org/pdf/1808.09653.pdf).

## Table of Contents
- [Repo Basics](#basics)
- [Get Embeddings](#embeddings)
- [Installation](#installation)
- [Reproduce the Results](#reproduction)
- [Citation](#citation)

## Basics
Brief intro for each folder:

- corpora: contains raw datasets published online by researchers. can ignore.

- data: contains formatted version of each corpus. Check notes in "data" folder for details.

- baseline: lexical baseline for the MOH-X, TroFi, and VUA datasets.

- classification: BiLSTM for the verb classification task.

- sequence: BiLSTM for the sequence labeling task.

## Embeddings
1. GloVe

Visit https://nlp.stanford.edu/projects/glove/, download glove.840B.300d.zip, and unzip it into a folder named "glove". Change the file name from "glove.840B.300d.txt" to "glove840B300d.txt".

2. ELMo

The ELMo vector will be released upon request by other means (too large to be uploaded on GitHub).

We have ELMo vectors for the MOH-X, TroFi, and VUA datasets with train/dev/test division. 

## Installation
1. This project is developed in Python 3.6. Using Conda to set up a virtual environment is recommended.

2. Install the required dependencies. 
    ```
    pip install -r requirements.txt
    ```
    
3. Install PyTorch from http://pytorch.org/.


## Reproduction

1. classification task (classification model): Check the main_XXX.py in the folder "classification".

2. sequence labeling task (sequence labeling model): Check the main_XXX.py in the folder "sequence".

### _Overall guideline_:

- Each main_XXX.py is a training and testing script for a classification model or sequence labeling model on dataset XXX. 

- Directly running main_XXX.py will train a model on dataset XXX, report the performance on validation set during training (codes for getting performance on training set are commented out), and report the final test performance *without* early stop. 

- Every main_XXX.py script contains some codes for plotting the model performance, which are commented out in order to directly run the script in terminal.

- All main_XXX.py scripts share the same variable naming convention and similar code structure.

- Default GPU usage is True. Change using_GPU to False if not using GPU.

- To try different sets of hyperparameters, please read the code comments for details.

### _Some details_:

- Note that it takes time to finish 10-fold cross validation on the MOH-X and TroFi datasets.

- For classification models, directly running the script is expected to get some numbers that are slightly lower than the reported numbers. Performances reported in the paper are steadily achieved **with** early stop and additional trainings with smaller learning rates, both of which are **not** included in the script for the consideration on runtime.

- For the classification model trained on the VUA verb classification dataset, the script does **not** report the macro-averaged F1. (The script does not save the genre of each example, so we wrote out predictions to compute this measure separately with a lookup table.)

- For sequence labeling models, directly running the script is expected to get results matched with the reported performance (likely to get slightly higher performance; possible to observe some small fluctuations).

- **Please run "mkdir predictions" at the root directory *before* running "python sequence/main_vua.py".** A "predictions" folder is where sequence/main_vua.py writes predictions, required to complete further evaluations on the VUA verb classification dataset.

- For the sequence labeling model trained on the VUA sequence labeling dataset, the script will report the model performance under five different evaluation setups in the following order:
    - performance on the VUA sequence labeling test set by POS tags regardless of genres
    - performance on the VUA verb classification test set by genres
    - performance on the VUA verb classification test set regardless of genres
    - performance on the VUA sequence labeling test set by genres
    - performance on the VUA sequence labeling test set regardless of genres



## Citation
```
@InProceedings{gao18nmd,
  author    = {Ge Gao, Eunsol Choi, Yejin Choi, and Luke Zettlemoyer},
  title     = {Neural Metaphor Detection in Context},
  booktitle = {EMNLP},
  year      = {2018}
}
```
