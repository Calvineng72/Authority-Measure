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

# command to run the file in the terminal
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
		df_prefixes = df[['obligation', 'constraint', 'permission', 'entitlement', 'other_provision', 'vlem', 
		    'obligation_1', 'obligation_2', 'constraint_1', 'constraint_2', 'constraint_3', 'permission_1', 
			'permission_2', 'permission_3', 'entitlement_1', 'entitlement_2', 'entitlement_3']]
		provisions = ['obligation', 'constraint', 'permission', 'entitlement', 'other_provision', 'obligation_1',
			'obligation_2', 'constraint_1', 'constraint_2', 'constraint_3', 'permission_1', 
			'permission_2', 'permission_3', 'entitlement_1', 'entitlement_2', 'entitlement_3']
		df_prefixes[provisions] = df_prefixes[provisions].astype(int)
		
		# replaces boolean values with text
		df['neg'] = np.where(df['neg'], 'não', '')

		# forms subject verb prefixes
		prefix_components = np.where(
			df['subject'] == 'se',
			df['neg'] + ' ' + df['subject'] + ' ' + df['modal'] + ' ' + df['helping_verb'] + ' ' + df['verb'],
			df['subject'] + ' ' + df['neg'] + ' ' + df['modal'] + ' ' + df['helping_verb'] + ' ' + df['verb']
		)
		df_prefixes['subject_verb_prefix'] = [re.sub(' +', ' ', x.lower().strip()) for x in prefix_components]

		# creates dummy variables for agents
		subject_df = pd.get_dummies(df['subnorm'])
		df_prefixes = pd.concat([df_prefixes, subject_df], axis=1)

		# counts and sorts subject verb prefixes
		df_prefixes['count'] = df_prefixes.groupby('subject_verb_prefix')['subject_verb_prefix'].transform('size')
		df_prefixes.drop_duplicates(subset='subject_verb_prefix', inplace=True)
		df_prefixes.sort_values(by='count', ascending=False, inplace=True)

		# saves combined DataFrame and DataFrames for each agent type
		df_prefixes.head(10000).to_csv(os.path.join(self.args.output_directory, "05_subject_verb_prefixes.csv"), index=False)
		df_worker = df_prefixes[df_prefixes['worker'] == 1].head(5000)
		df_worker.to_csv(os.path.join(self.args.output_directory, "05_worker_subject_verb_prefixes.csv"), index=False)
		df_firm = df_prefixes[df_prefixes['firm'] == 1].head(5000)
		df_firm.to_csv(os.path.join(self.args.output_directory, "05_firm_subject_verb_prefixes.csv"), index=False)
		df_union = df_prefixes[df_prefixes['union'] == 1].head(5000)
		df_union.to_csv(os.path.join(self.args.output_directory, "05_union_subject_verb_prefixes.csv"), index=False)
		df_manager = df_prefixes[df_prefixes['manager'] == 1].head(5000)
		df_manager.to_csv(os.path.join(self.args.output_directory, "05_manager_subject_verb_prefixes.csv"), index=False)

	def aggregate_measures(self):
		df = pd.read_pickle(os.path.join(self.args.output_directory, "04_auth.pkl"))		
		to_keep = ['contract_id', 'md', 'passive', 'neg', 'strict_modal', 'permissive_modal', 'obligation_verb', 
	     	'constraint_verb', 'permission_verb', 'entitlement_verb', 'promise_verb', 'special_verb', 'active_verb', 
			'obligation', 'constraint', 'permission', 'entitlement', 'other_provision', 'subnorm']
		df = df[to_keep]

		subjects = ['worker', 'firm', 'union', 'manager']
		statements = ['obligation', 'constraint', 'permission', 'entitlement']

		# adds subject-mesaure counts
		for cur_measure in statements:
			df[cur_measure] = df[cur_measure].astype(float)
			for cur_subnorm in subjects:
				new_col_name = cur_measure + "_" + cur_subnorm
				df[new_col_name] = [(1 * i) if j == cur_subnorm else (0 * i) for i, j in zip(df[cur_measure], df["subnorm"])]
		
		# adds subnorm counts
		all_subjects = ['worker', 'firm', 'union', 'manager', 'other_agent']
		for cur_subnorm in all_subjects:
			df[cur_subnorm + "_count"] = [1 if i == cur_subnorm else 0 for i in df["subnorm"]]
		df["num_statements"] = [1] * len(df)
		df = df.groupby("contract_id", as_index=False).sum()
		df["contract_id"] = df["contract_id"].str.replace("_cleaned.txt", "", regex=True)
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
