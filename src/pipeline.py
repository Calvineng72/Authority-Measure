import argparse
import json
import os
from tqdm import tqdm
# 3rd party imports
import spacy
import neuralcoref
import sys
from collections import defaultdict
from main02_parse_articles import parse_article
from main03_get_parse_data import extract_pdata
from main04_compute_auth import combine_auth, compute_statement_auth
import pandas as pd


class Pipeline():
	def __init__(self, args):
		self.args = args
		os.makedirs(self.args.output_directory, exist_ok=True)
		self.nlp = spacy.load('en_core_web_sm', disable=["ner"])
		neuralcoref.add_to_pipe(self.nlp)

	def parse_articles(self):
		for filename in tqdm(os.listdir(args.input_directory)):
			parse_article(filename, self.nlp, self.args)

	def extract_parsed_data(self):
		extract_pdata(self.args)

	def compute_authority_measures(self):
		chunks = os.listdir(os.path.join(args.output_directory, "03_pdata"))
		chunks = sorted(chunks, key=lambda x: int(x.split("_")[-1][:-4]))
		for filename in tqdm(chunks):
			filepath = os.path.join(args.output_directory, "03_pdata", filename)
			cur_df = pd.read_pickle(filepath)
			compute_statement_auth(args, cur_df, filename)
		combine_auth()

	def run_main(self):
		# dependency parsing
		#os.makedirs(os.path.join(self.args.output_directory, "02_parsed_articles"), exist_ok=True)
		#self.parse_articles()

		# extract necessary parsed information
		#os.makedirs(os.path.join(self.args.output_directory, "03_pdata"), exist_ok=True)
		#self.extract_parsed_data()

		os.makedirs(os.path.join(self.args.output_directory, "04_auth"), exist_ok=True)

		# compute authority measures for chunks
		self.compute_authority_measures()


		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_directory", type=str, default="")
	parser.add_argument("--output_directory", type=str, default="")
	args = parser.parse_args()
	pipeline = Pipeline(args)
	pipeline.run_main()


