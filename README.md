# LegalBert_LegalPegasus

## Overview

Welcome to the LegalBert_AustralianLegalCitation repository! This repository serves as a learning platform for exploring the capabilities of LEGAL-BERT and legal-pegasus transformers in processing legal text data. LEGAL-BERT is transformer model fine-tuned specifically for legal domain tasks, and legal-pegasus is designed for legal document summarization, they offer powerful tools for natural language processing in the legal context. Please note that this repository is intended only for my educational and practice purposes only, it is not focus on clening and processing text steps.

## Setup and Requirements

I run this notebooks mainly with GPU acceleration using a RTX3070 with 8GB vRAM on my laptop with Windows OS.

Datasets:
* [Legal Citation Text Classification](https://www.kaggle.com/datasets/shivamb/legal-citation-text-classification/code?resource=download)
* [EUR-Lex-Sum](https://huggingface.co/datasets/dennlinger/eur-lex-sum)

Transformers model:
* [LEGAL-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
* [legal-pegasus](https://huggingface.co/nsi319/legal-pegasus)

library used:
* Hugging Face
* Pythorch
* Sklearn
* pandas
* Optuna

## LEGAL-BERT

I used LEGAL-BERT almost in all notebooks:
* In the feature_extraction_legalbert notebook i provide an exploration of feature extraction using LEGAL-BERT. Through this notebook, i used legal-bert-small-uncased transformer for exctract the last hidden state output and use as feature for a Logistic model. In this case i had to use cpu as device instead of gpu because of low gpu memory.
* In the training_and_finetuning_legalbert notebook i offer a solution for training and fine-tuning LEGAL-BERT.Through this notebook, i used bert-base-uncased-eurlex transformer and trained over Legal Citation Text Classification and fine tuned using Optuna hyperparameter search.
* In the LogisticHead_LegalBertFeatureExtraction_PegasusSummarizer notebook is similar to training_and_finetuning_legalbert notebook with the addition of a pegasus model that summarize longer samples before processing them with BERT.
  
## legal-pegasus Notebook
I used legal-pegasu in couple notebooks:
* In pegasus_summarizer_splitpredict notebook i used legal-pegasus for predict summarizies from EUR-Lex-Sum samples, single summary is made by splitting single sample in chuncks and then concatenating all chunks summary on a single summary.
* In the LogisticHead_LegalBertFeatureExtraction_PegasusSummarizer notebook is similar to pegasus_summarizer_splitpredict, but this time the summirize text in output goes as input of BERT tokenizer.

## Conclusion

The final bench mark are not quite well for use it as a real world model and i think that this is because i should preprocess the text before using BERT or pegasus, but i really enjoy working with trasformers for the first time in my life.
