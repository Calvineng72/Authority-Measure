# Unsupervised Extraction of Workplace Rights and Duties from Collective Bargaining Agreements

This is the (partial) code base for our paper [Unsupervised Extraction of Workplace Rights and Duties from Collective Bargaining Agreements](https://www.research-collection.ethz.ch/handle/20.500.11850/473199.1). For questions regarding the code base, please contact dominik.stammbach@gess.ethz.ch.

The code base covers:

* Parsing articles
* Computing Authority Scores on statement level
* Aggregating authority scores on contract level

The main output files of the pipeline are:

* $output_directory/04_auth.pkl (contains all entitlements/obligations/constraints/permissions per clause-level)
* $output_directory/05_aggregated.pkl (contains aggregated scores by different subjects on contract-level)


The code base does **_not_** cover:

* Splitting Contracts into articles (we developed a customized solution for converting PDFs to text and splitting labor union contracts which is domain specific and does not translate well to other contracts)
* Parallelized parsing (the computational bottleneck for large collections is spaCy dependency parsing. We computed this in parallel using linux command line tools and 96 machines)
* Our analysis which is also heavily customized for Canadian Union Contracts and does not necessarily translate well to other domain (e.g. clustering on article headers and training LDA)


## Update December 2023

* Added aggregating scores per subject on contract level
* Re-factored how entitlemnets are computed in src/main04_compute_auth.py

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

The pipeline covers two types of input. In any case, input is a directory containing each contract, either as a .txt file or as a .json file.

* if input is json, each contract should already be split into "articles", and contain a contract_id.
* else, rights and duties are computed on contract level, and the filename becomes the article_id

Output will be stored in output_directory, the main output there is the file 04_auth.pkl. For each subject-verb tuple, it contains a boolean value whether it is an entitlement, obligation etc. and saves the "role" of the subject (worker, firm etc.). These results can then be aggregated at any desired level. Intermediate pipeline steps will get saved as well in the output directory.

```shell
input_directory="sample_data"
output_directory="output_sample_data"
python src/pipeline.py --input_directory $input_directory --output_directory $output_directory
```

Our results in the paper are computed with coreferences resolved. However, setting up neuralcoref spacy in 2022 is somewhat cumbersome and the whole process is computationally expensive. If coreference resolution is desired, install neural_coref and run the pipeline with the flug --use_neural_coref 

```shell
input_directory="path/to/directory/with/contracts"
output_directory="authority_measures"
python src/pipeline.py --use_neural_coref --input_directory $input_directory --output_directory $output_directory
```


## What probably needs to be customized for other contract collections
* We were interested in very specific roles, e.g. worker, firm etc. This is simply a dictionary lookup of the subject of a clause, e.g. following words are considered to be *worker*: worker="employee,worker,staff,teacher,nurse,mechanic,operator,steward,personnel" Overwrite these for customized applications in the file main04_compute_auth.py





