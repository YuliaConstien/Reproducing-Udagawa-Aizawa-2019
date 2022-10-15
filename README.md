# Reproducing Udagawa & Aizawa (2019):
This repository shows the code for the reproduction of Udagawa and Aizawa 2019:
>Takuma Udagawa and Akiko Aizawa. "A Natural Language Corpus of Common Grounding under Continuous and Partially-Observable Context". In: Proceedings of the AAAI Conference on Artificial Intelligence 33.01 (2019), pp. 7120-7127. 

## Structure of the Repository
### Branches
**main:** 
- Find the original code of the authors in folder authors_repo_2019 with small changes for compatability.
- Find the code used for reproduction in aaai2019_changed_0802, including the requirements.txt which specifies the package versions used for reproduction.

**Hyperparameter_changes**
- Includes the results of training/validation/test accuracy for the experiments conducted with the baseline models (MLP and RN) on the full corpus. The experiments included changes in three hyperparameters: batch size, learning rate and number of epochs.

**embeds_2_types:**
- Find the code used for reproduction plus additional pretrained embedding layer.

**arch-RNN/BiRNN/GRU/LSTM/BiLSTM**
- Find the code for each model variation in the respective branch.

## Reproduction
Install the package versions used for reproducing the results. File can be found in folder ` aaai2019_changed_0802`
```
$ pip install requirements.txt
```
Train the model using Context (MLP-encoded) and Dialogue on the full test set using the command below. See parser arguments for running with RN encoded Context or test set subsets.
```
$ python train.py --test_corpus_full
```
Command for reproducing the heatmap shown in section 4.2 of the original paper:
```
$ python simple_analysis.py \
--basic_statistics --count_dict \
--plot_selection_bias
```

## Code Additions
### 1. Hyperparameter Changes
Switch to branch `Hyperparameter_changes`

The commands for testing the hyperparameters can be found and run in the file `HyperparamChanges_Common_Grounding_.ipynb`

### 2. Pretrained Embeddings
Switch to branch `embeds_2_types`

Download word vectors:
```
$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.3.0/en_vectors_web_lg-2.3.0.tar.gz
```
Install word vectors:
```
$ pip install en_vectors_web_lg-2.3.0.tar.gz
```
Run the training with the pretrained embeddings on Context (MLP-encoded) + Dialogue on the full test set and with frozen embeddings using the command below. Set parser argument `--trainable_embeds True` for trainable embeddings, default is False. See other parser arguments for RN or test set subset experiments.

```
$ python train.py --test_corpus full --nembed_word 300
```

### 3. Model Type Variations
To run the model with RNN, GRU, LSTM, BiRNN or BiLSTM select the according branch and run the same commands as for the reproduction, e.g.

```
$ python train.py --test_corpus full
```


