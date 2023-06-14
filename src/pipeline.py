import argparse
import os
import re
from tqdm import tqdm
import spacy
from main02_parse_articles import parse_article
from main03_get_parse_data import extract_pdata
from main04_compute_auth import combine_auth, compute_statement_auth
import pandas as pd
import numpy as np

# python src/pipeline.py --input_directory cleaned_cba_samples --output_directory output

pd.options.mode.chained_assignment = None

class Pipeline():
	def __init__(self, args):
		self.args = args
		os.makedirs(self.args.output_directory, exist_ok=True)
		self.nlp = spacy.load('pt_core_news_sm', disable=["ner"])

	def parse_articles(self):
		for filename in tqdm(os.listdir(args.input_directory)):
			parse_article(filename, self.nlp, self.args)

	def extract_parsed_data(self):
		extract_pdata(self.args)

	def compute_authority_measures(self):
		chunks = os.listdir(os.path.join(self.args.output_directory, "03_pdata"))
		chunks = sorted(chunks, key=lambda x: int(x.split("_")[-1][:-4]))
		for filename in tqdm(chunks):
			filepath = os.path.join(self.args.output_directory, "03_pdata", filename)
			cur_df = pd.read_pickle(filepath)
			compute_statement_auth(self.args, cur_df, filename)
		combine_auth(self.args)

	def determine_subject_verb_prefixes(self):
		df = pd.read_pickle(os.path.join(self.args.output_directory, "04_auth.pkl"))
		df_prefixes = df[['obligation', 'constraint', 'permission', 'entitlement', 'other_provision']]
		
		# replaces boolean values with text
		df['neg'] = df['neg'].apply(lambda x: 'não' if x else '')

		# forms subject verb prefixes
		prefix_components = np.where(
			df['subject'] == 'se',
			df['neg'] + ' ' + df['subject'] + ' ' + df['modal'] + ' ' + df['helping_verb'] + ' ' + df['verb'],
			df['subject'] + ' ' + df['neg'] + ' ' + df['modal'] + ' ' + df['helping_verb'] + ' ' + df['verb']
		)
		df_prefixes['subject_verb_prefix'] = [re.sub(' +', ' ', x.lower()) for x in prefix_components]

		# creates dummy variables for agents
		subject_df = pd.get_dummies(df['subnorm']).astype('bool')
		df_prefixes = pd.concat([df_prefixes, subject_df], axis=1)

		# counts and sorts subject verb prefixes
		df_grouped = df_prefixes.groupby('subject_verb_prefix').size().reset_index()
		df_grouped.columns = ['subject_verb_prefix', 'count']
		df_prefixes = pd.merge(df_prefixes, df_grouped, on='subject_verb_prefix')
		df_prefixes = df_prefixes.drop_duplicates(subset='subject_verb_prefix')
		df_prefixes = df_prefixes.sort_values(by='count', ascending=False).reset_index(drop=True)

		df_prefixes.to_csv(os.path.join(self.args.output_directory, "05_subject_verb_prefixes.csv"))

	def aggregate_measures(self):
		df = pd.read_pickle(os.path.join(self.args.output_directory, "04_auth.pkl"))		
		print(df)
		print(df.columns)
		subjects = ["worker", 'firm', 'union', 'manager']
		statements = ['obligation', 'constraint', 'permission', 'entitlement']

		# adds subject-mesaure counts
		for cur_measure in statements:
			df[cur_measure] = df[cur_measure].astype(float)
			for cur_subnorm in subjects:
				new_col_name = cur_measure + "_" + cur_subnorm
				df[new_col_name] = [(1 * i) if j == cur_subnorm else (0 * i) for i, j in zip(df[cur_measure], df["subnorm"])]
		
		# adds subnorm counts
		for cur_subnorm in subjects:
			df[cur_subnorm + "_count"] = [1 if i == cur_subnorm else 0 for i in df["subnorm"]]
		df["num_statements"] = [1] * len(df)
		df = df.groupby("contract_id", as_index=False).sum()
		df["contract_id"] = df["contract_id"].str.replace("_cleaned.txt", "", regex=True)
		print(df)
		print(df.columns)
		df.to_csv(os.path.join(self.args.output_directory, "05_aggregated.csv"))

	def run_main(self):
		# dependency parsing
		os.makedirs(os.path.join(self.args.output_directory, "02_parsed_articles"), exist_ok=True)
		self.parse_articles()

		# extract necessary parsed information
		os.makedirs(os.path.join(self.args.output_directory, "03_pdata"), exist_ok=True)
		self.extract_parsed_data()

		os.makedirs(os.path.join(self.args.output_directory, "04_auth"), exist_ok=True)

		# compute authority measures for chunks
		self.compute_authority_measures()
		self.determine_subject_verb_prefixes()
		self.aggregate_measures()
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_directory", type=str, default="sample_data")
	parser.add_argument("--output_directory", type=str, default="output_sample_data")
	parser.add_argument("--use_neural_coref", action='store_true')
	args = parser.parse_args()
	pipeline = Pipeline(args)
	pipeline.run_main()
