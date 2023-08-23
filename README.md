# An Authority Measure for Collective Bargaining Agreements.

This repository is built upon the research paper titled ["Unsupervised Extraction of Workplace Rights and Duties from Collective Bargaining Agreements"](https://www.research-collection.ethz.ch/handle/20.500.11850/473199.1) by Elliot Ash, Jeff Jacobs, Bentley MacLeod, Suresh Naidu, and Dominik Stammbach. The original code base can be found [here](https://github.com/dominiksinsaarland/labor-contracts). The pipeline is adapted for Brazilian collective bargaining agreements with modifications to the parsing algorithm and dictionaries to accommodate the Portuguese language.  

The repository covers the following:
* Cleaning Documents (Sample Code)
* Parsing Documents in Portuguese
* Computing Authority Scores on Statement Level
* Aggregating Authority Scores on Contract Level

The main output of the pipeline is the file $output_directory/05_aggregated.csv, which contains information for each document on the number of obligations, permissions, entitlements, and constraints for each agent type. The pipeline may be run with each document as an observation, or with each clause as an observation, whereby the '--clause' flag is used to specify the latter. 

Within this repository are sample Jupyter notebooks used to clean collective bargaining agreements by document and by clause. Please note that minor adjustments may be necessary for the pipeline to function properly when used with different document formats.

## Getting Started

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and Linux, the environment can be installed with the following commands:
```shell
conda create -n authority_measure python=3.9.13
conda activate authority_measure
pip install -r requirements.txt
pip install spacy==3.5.4
python -m spacy download pt_core_news_sm
```

## Running the Pipeline

The pipeline accepts cleaned .txt files, where the file name is the contract ID for a given CBA. The input_directory should be the name of the folder containing the cleaned documents. In order to run the pipeline, run the following command:

```shell
input_directory="cleaned_cbas"
output_directory="output"
python src/pipeline.py --input_directory $input_directory --output_directory $output_directory
```

There is also an option to run the pipeline by clause rather than by document. With this option, the cleaned.txt files must contain a double array, where each interior array is composed of two elements: the clause name and clause text. 

```shell
input_directory="cleaned_cbas_clause"
output_directory="output"
python src/pipeline.py --input_directory $input_directory --output_directory $output_directory --clause
```

## References
E. Ash, J. Jacobs, B. MacLeod, S. Naidu and D. Stammbach, "Unsupervised Extraction of Workplace Rights and Duties from Collective Bargaining Agreements," *2020 International Conference on Data Mining Workshops (ICDMW)*, Sorrento, Italy, 2020, pp. 766-774, doi: 10.1109/ICDMW51313.2020.00112.
