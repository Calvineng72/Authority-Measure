# Unsupervised Extraction of Workplace Rights and Duties from Collective Bargaining Agreements

This is the (partial) code base for our paper [Unsupervised Extraction of Workplace Rights and Duties from Collective Bargaining Agreements](https://www.research-collection.ethz.ch/handle/20.500.11850/473199.1). For questions regarding the code base, please contact dominik.stammbach@gess.ethz.ch.

The code base covers:

* Parsing articles
* Computing Authority Scores per Article level

The code base does **_not_** cover:

* Splitting Contracts into articles (we developed a customized solution for converting PDFs to text and splitting labor union contracts which is domain specific and does not translate well to other contracts)
* Parallelized parsing (the computational bottleneck for large collections is spaCy dependency parsing. We computed this in parallel using linux command line tools and 96 machines)
* Aggregating authority scores (we aggregated on contract level, but this depends obviously on the specific use case)
* Our analysis which is also heavily customized for Canadian Union Contracts and does not necessarily translate well to other domain


## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n labor-contracts python=3.6
conda activate labor-contracts
pip install -r requirements.txt
```

### Installing spaCy and linking with neuralcoref

Installing spaCy and linking with neuralcoref does not work out of the box on linux, the following steps eventually worked

```shell
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
pip install spacy==2.3.2
python -m spacy download en
```

## Run the pipeline

Input is ...

Output will be ...

```shell
python pipeline.py
```


## What probably needs to be customized for other contract collections

* data loading
* different roles




