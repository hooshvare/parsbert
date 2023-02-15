<h1 align="center">ParsBERT: Transformer-based Model for Persian Language Understanding 🦁</h1>

<br/><br/>


ParsBERT is a monolingual language model based on Google’s BERT architecture. This model is pre-trained on large Persian corpora with various writing styles from numerous subjects (e.g., scientific, novels, news) with more than `3.9M` documents, `73M` sentences, and `1.3B` words. 

<!-- Paper presenting ParsBERT: [arXiv:2005.12515](https://arxiv.org/abs/2005.12515) -->
Paper presenting ParsBERT: [DOI: 10.1007/s11063-021-10528-4](https://doi.org/10.1007/s11063-021-10528-4)

**CURRENT VERSION: V3**



## Introduction

ParsBERT trained on a massive amount of public corpora ([Persian Wikidumps](https://dumps.wikimedia.org/fawiki/), [MirasText](https://github.com/miras-tech/MirasText)) and six other manually crawled text data from a various type of websites ([BigBang Page](https://bigbangpage.com/) `scientific`, [Chetor](https://www.chetor.com/) `lifestyle`, [Eligasht](https://www.eligasht.com/Blog/) `itinerary`,  [Digikala](https://www.digikala.com/mag/) `digital magazine`, [Ted Talks](https://www.ted.com/talks) `general conversational`, Books `novels, storybooks, short stories from old to the contemporary era`).

As a part of ParsBERT methodology, an extensive pre-processing combining POS tagging and WordPiece segmentation was carried out to bring the corpora into a proper format. 


[![ParsBERT Demo](/assets/parsbert-playground.png)](https://www.youtube.com/watch?v=Fyirkq668PE)

<strike><a href="http://lab.hooshvare.com/">ParsBERT Playground</a></strike>


## Evaluation

ParsBERT is evaluated on three NLP downstream tasks: Sentiment Analysis (SA), Text Classification, and Named Entity Recognition (NER). For this matter and due to insufficient resources, two large datasets for SA and two for text classification were manually composed, which are available for public use and benchmarking. ParsBERT outperformed all other language models, including multilingual BERT and other hybrid deep learning models for all tasks, improving the state-of-the-art performance in Persian language modeling.

## Results

The following table summarizes the F1 score obtained by ParsBERT as compared to other models and architectures.


### Sentiment Analysis (SA) task

|          Dataset         | ParsBERT v3 | ParsBERT v2 | ParsBERT v1 | mBERT | DeepSentiPers |
|:------------------------:|:-----------:|:-----------:|:-----------:|:-----:|:-------------:|
|  Digikala User Comments  |      -      |    81.72    |    81.74*   | 80.74 |       -       |
|  SnappFood User Comments |      -      |    87.98    |    88.12*   | 87.87 |       -       |
|  SentiPers (Multi Class) |      -      |    71.31*   |    71.11    |   -   |     69.33     |
| SentiPers (Binary Class) |      -      |    92.42*   |    92.13    |   -   |     91.98     |



### Text Classification (TC) task

|      Dataset      | ParsBERT v3 | ParsBERT v2 | ParsBERT v1 | mBERT |
|:-----------------:|:-----------:|:-----------:|:-----------:|:-----:|
| Digikala Magazine |      -      |    93.65*   |    93.59    | 90.72 |
|    Persian News   |      -      |    97.44*   |    97.19    | 95.79 |


### Named Entity Recognition (NER) Task

| Dataset | ParsBERT v3 | ParsBERT v2 | ParsBERT v1 | mBERT | MorphoBERT | Beheshti-NER | LSTM-CRF | Rule-Based CRF | BiLSTM-CRF |
|:-------:|:-----------:|:-----------:|:-----------:|:-----:|:----------:|:------------:|:--------:|:--------------:|:----------:|
|  PEYMA  |             |    93.40*   |    93.10    | 86.64 |      -     |     90.59    |     -    |      84.00     |      -     |
|  ARMAN  |             |    99.84*   |    98.79    | 95.89 |    89.9    |     84.03    |   86.55  |        -       |    77.45   |

**If you tested ParsBERT on a public dataset, and you want to add your results to the table above, open a pull request or contact us. Also make sure to have your code available online so we can add it as a reference**

## How to use

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel

# v3.0
model_name_or_path = "HooshvareLab/bert-fa-zwnj-base"
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# model = TFAutoModel.from_pretrained(model_name_or_path)  For TF
model = AutoModel.from_pretrained(model_name_or_path)

text = "ما در هوش‌واره معتقدیم با انتقال صحیح دانش و آگاهی، همه افراد میتوانند از ابزارهای هوشمند استفاده کنند. شعار ما هوش مصنوعی برای همه است."
tokenizer.tokenize(text)

['ما', 'در', 'هوش', '[ZWNJ]', 'واره', 'معتقدیم', 'با', 'انتقال', 'صحیح', 'دانش', 'و', 'آ', '##گاهی', '،', 'همه', 'افراد', 'میتوانند', 'از', 'ابزارهای', 'هوشمند', 'استفاده', 'کنند', '.', 'شعار', 'ما', 'هوش', 'مصنوعی', 'برای', 'همه', 'است', '.']
```

## Derivative models

### V3.0

#### BERT v3.0 Model
- [HooshvareLab/bert-fa-zwnj-base](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base) 

#### DistilBERT v3.0 Model
- [HooshvareLab/distilbert-fa-zwnj-base](https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base) 

#### ALBERT v3.0 Model
- [HooshvareLab/albert-fa-zwnj-base-v2](https://huggingface.co/HooshvareLab/albert-fa-zwnj-base-v2) 

#### ROBERTA v3.0 Model
- [HooshvareLab/roberta-fa-zwnj-base](https://huggingface.co/HooshvareLab/roberta-fa-zwnj-base) 


### V2.0

#### ParsBERT v2.0 Model
- [HooshvareLab/bert-fa-base-uncased](https://huggingface.co/HooshvareLab/bert-fa-base-uncased) 

#### ParsBERT v2.0 Sentiment Analysis
- [HooshvareLab/bert-fa-base-uncased-sentiment-digikala](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-digikala) 
- [HooshvareLab/bert-fa-base-uncased-sentiment-snappfood](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-snappfood) 
- [HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary) 
- [HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-multi](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-multi) 

#### ParsBERT v2.0 Text Classification
- [HooshvareLab/bert-fa-base-uncased-clf-digimag](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-clf-digimag) 
- [HooshvareLab/bert-fa-base-uncased-clf-persiannews](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-clf-persiannews) 

#### ParsBERT v2.0 NER 
- [HooshvareLab/bert-fa-base-uncased-ner-peyma](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-ner-peyma) 
- [HooshvareLab/bert-fa-base-uncased-ner-arman](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-ner-arman) 

### V1.0

#### ParsBERT v1.0 Model
- [HooshvareLab/bert-base-parsbert-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased) 

#### ParsBERT v1.0 NER 
- [HooshvareLab/bert-base-parsbert-peymaner-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-peymaner-uncased) 
- [HooshvareLab/bert-base-parsbert-armanner-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-armanner-uncased) 
- [HooshvareLab/bert-base-parsbert-ner-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-ner-uncased) 


## NLP Tasks Tutorial  :hugs:

| Notebook                 |                                                                                 |
|--------------------------|---------------------------------------------------------------------------------|
| Text Classification      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hooshvare/parsbert/blob/master/notebooks/Taaghche_Sentiment_Analysis.ipynb) |
| Sentiment Analysis       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hooshvare/parsbert/blob/master/notebooks/Taaghche_Sentiment_Analysis.ipynb) |
| Named Entity Recognition |  |
| Text Generation          |  |


## Cite 

Please cite the following paper in your publication if you are using [ParsBERT](https://doi.org/10.1007/s11063-021-10528-4) in your research:

```bibtex
 @article{ParsBERT, 
    title={Parsbert: Transformer-based model for Persian language understanding}, 
    DOI={10.1007/s11063-021-10528-4}, 
    journal={Neural Processing Letters}, 
    author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri}, 
    year={2021}
} 
```

## Acknowledgments

We hereby, express our gratitude to the [Tensorflow Research Cloud (TFRC) program](https://tensorflow.org/tfrc) for providing us with the necessary computation resources. We also thank [Hooshvare](https://hooshvare.com) Research Group for facilitating dataset gathering and scraping online text resources.


## Contributors

- Mehrdad Farahani: [Linkedin](https://www.linkedin.com/in/m3hrdadfi/), [Twitter](https://twitter.com/m3hrdadfi), [Github](https://github.com/m3hrdadfi)
- Mohammad Gharachorloo:  [Linkedin](https://www.linkedin.com/in/mohammad-gharachorloo/), [Twitter](https://twitter.com/MGharachorloo), [Github](https://github.com/baarsaam)
- Marzieh Farahani:  [Linkedin](https://www.linkedin.com/in/marziehphi/), [Twitter](https://twitter.com/marziehphi), [Github](https://github.com/marziehphi)
- Mohammad Manthouri:  [Linkedin](https://www.linkedin.com/in/mohammad-manthouri-aka-mansouri-07030766/), [Twitter](https://twitter.com/mmanthouri), [Github](https://github.com/mmanthouri)
- Hooshvare Team:  [Official Website](https://hooshvare.com/), [Linkedin](https://www.linkedin.com/company/hooshvare), [Twitter](https://twitter.com/hooshvare), [Github](https://github.com/hooshvare), [Instagram](https://www.instagram.com/hooshvare/)

## Releases

### v3.0 (2021-02-28)
The new version of BERT v3.0 for Persian is available today and can tackle the zero-width non-joiner character for Persian writing. 
Also, the model was trained on new multi-types corpora with a new set of vocabulary.

Available by: [HooshvareLab/bert-fa-zwnj-base](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base)



### v2.0 (2020-09-05)
ParsBERT v2.0: We reconstructed the vocabulary and fine-tuned the ParsBERT v1.1 on the new Persian corpora in order to provide some functionalities for using ParsBERT in other scopes!
Objective goals during training are as below (after 300K steps).

```bash
***** Eval results *****
global_step = 300000
loss = 1.4392426
masked_lm_accuracy = 0.6865794
masked_lm_loss = 1.4469004
next_sentence_accuracy = 1.0
next_sentence_loss = 6.534152e-05
```

Available by: [HooshvareLab/bert-fa-base-uncased](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)

### v1.1 (2020-06-24)
ParsBERT v1.1: We continued the training for more than 2.5M steps based on the same Persian corpora and BERT-Base config.
Objective goals during training are as below (after 2.5M steps).

```bash
***** Eval results *****
global_step = 2575000
loss = 1.3973521
masked_lm_accuracy = 0.70044917
masked_lm_loss = 1.3974043
next_sentence_accuracy = 0.9976562
next_sentence_loss = 0.0088804625
```

Available by: [HooshvareLab/bert-base-parsbert-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)

### v1.0 (2020-05-27)
ParsBERT v1: This is the first version of our ParsBERT based on BERT-Base. The model was trained on vast Persian corpora for 1920000 steps.
Objective goals during training are as below (after 1.9M steps).

```bash
***** Eval results *****
global_step = 1920000
loss = 2.6646128
masked_lm_accuracy = 0.583321
masked_lm_loss = 2.2517521
next_sentence_accuracy = 0.885625
next_sentence_loss = 0.3884369
```

Available by: [HooshvareLab/bert-base-parsbert-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)

## License

[Apache License 2.0](LICENSE)
