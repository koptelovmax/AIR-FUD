# AIR-FUD+ (Automatic Identification of Rules in French Urban Documents extended)
The code to support our submission to the SN Computer Science journal

## Structure of the code:

### Data preparation

- **data_loader.py** -- load binary segments from multi-labeled corpus

### Baseline methods

- **triggerwords.py** -- method based on the list of trigger words contained in the *'trigger\_words'*  file 

- **tf_vectors.py** -- construction of term frequency vectors by the vector similarity model and the ML method

- **vectorsim.py** -- vector similarity model

- **ML-model.py** -- machine learning method using term frequency vectors

### State-of-the art implementation

- **camembert.py** -- fine-tuning the CamemBERT model for binary classification of text segments

- **camembert_weighted.py** -- fine-tuning the CamemBERT model for binary classification using weighted accuracy as an optimization criterion

### Text augmentation
Jupyter notebooks adapted for Google Colab:

- **augmentation_pos-driven.ipynb** -- POS-driven method implementing CamemBERT masked word prediction 

- **augmentation_semantic.ipynb** -- Semantic-driven method exploiting nomenclature concepts contained in the *'nomenclature'*  file 

- **augmentation_combined.ipynb** -- Combined approach applying POS-driven method to segments which contain expert nomenclature concepts stored in the *'nomenclature\_expert'* file 

### Rule formulation

- **segment_selection.py** -- selection of segments for evaluation with rule formulation module

- **rule_formulation_AMR.py** -- summary generation using AMR graphs

- **rule_formulation_ChatGPT.ipynb** -- summary generation using ChatGPT

- **bert_similarity.ipynb** -- automatic evaluation by computing Bert similarity

- **evaluation_auto.py** -- calculation of quality measures for automatic evaluation

- **evaluation_manual.py** -- calculation of quality measures for manual evaluation

### Processing of new documents

- **pdf2text.py** -- text extraction from the documents in the PDF format

- **segment_construction.py** -- segment construction from the annotated documents in the txt format

- **segment_construction_auto.py** -- segment construction from the non-annotated documents in the txt format (fully automatic approach)

## To reconstruct the experiments:

### 1st classification

1) Prepare segments for 1st classification:

`python3 data_loader.py 1`

2) Run trigger words with window of size 10:

`python3 triggerwords.py 1 10`

3) Construct TF and TF-IDF vectors:

`python3 tf_vectors.py 1`

4) Run the vector similarity model with TF-IDF vectors:

`python3 vectorsim.py 1 tfidf`

5) Run the ML method with TF vectors:

`python3 ML-model.py 1 tf`

6) Run the CamemBERT model on the 'cuda:0' GPU:

`python3 camembert.py 1 orig cuda:0`

### 2nd classification

1) Prepare segments for 2nd classification:

`python3 data_loader.py 2`

2) Run trigger words with window of size 1:

`python3 triggerwords.py 2 1`

3) Construct TF and TF-IDF vectors:

`python3 tf_vectors.py 2`

4) Run the vector similarity model with TF-IDF vectors:

`python3 vectorsim.py 2 tfidf`

5) Run the ML method with TF-IDF vectors:

`python3 ML-model.py 2 tfidf`

### 3rd classification

1) Prepare segments for 3rd classification:

`python3 data_loader.py 3`

2) Run trigger words with window of size 1:

`python3 triggerwords.py 3 1`

3) Construct TF and TF-IDF vectors:

`python3 tf_vectors.py 3`

4) Run the vector similarity model with TF-IDF vectors:

`python3 vectorsim.py 3 tfidf`

5) Run the ML method with TF vectors:

`python3 ML-model.py 3 tf`

6) Run the CamemBERT model on the 'cuda:0' GPU:

`python3 camembert_weighted.py 3 orig cuda:0`

7) Perform data augmentation in Google Colab:

`augmentation_pos-driven.ipynb > Runtime > Run all`

`augmentation_semantic.ipynb > Runtime > Run all`

`augmentation_combined.ipynb > Runtime > Run all`

8) Run the CamemBERT model on the augmented data:

`./augmentation_experiments.sh`

### Rule formulation (sequence of steps and tips)

1) Sample segments for evaluation by using `python3 segment_selection.py`

2) Extract keywords with [https://rdius-herelles-ner-app-smoynm.streamlit.app](https://rdius-herelles-ner-app-smoynm.streamlit.app) and download them in the JSON format

3) Generate rules using AMR graphs:

`python3 rule_formulation_AMR.py Verifiable`

`python3 rule_formulation_AMR.py Soft`

`python3 rule_formulation_AMR.py Non-verifiable`

4) Generate rules using ChatGPT (in Google Colab):

`rule_formulation_ChatGPT.ipynb > Runtime > Run all`

5) Compute Bert similarity for automatic evaluation (in Google Colab):

`bert_similarity.ipynb > Runtime > Run all`

6) Compute quality measures for automatic evaluation by using `python3 evaluation_auto.py`

7) Perform evaluation by a group of volunteers

8) Compute quality measures for manual evaluation by usung  `python3 evaluation_manual.py`

Environment requirements: Python 3.8.8, nltk, easynmt, simalign, penman, amrlib, numpy, sklearn, torch 1.10.1 + cuda 10.2, transformers (Hugging Face)
