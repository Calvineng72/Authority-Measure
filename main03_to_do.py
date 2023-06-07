# -*- coding: utf-8 -*-

from collections import Counter
import os
import argparse
import pandas as pd
import joblib
from tqdm import tqdm

                        
other = ['agreement',
         'day',
         'assignment','leave','payment',
         'he/she','they','party']

conditionals = ['if','when','unless']
	#'where', 'whereas','whenever', 'provided that', 'in case']
    
    
def extract_pdata():
    subcount = Counter()
    subnouncount = Counter()
    modalcount = Counter()

    iteration_num = 0
    chunk_num = 0
    pdata_rows = []

    files = os.listdir('output_parsed_documents')
    filenames = [os.path.join('output_parsed_documents', fn) for fn in files]
    for filename in tqdm(filenames, total=len(filenames)):
        for statement_data in joblib.load(filename):
            contract_id = statement_data["contract_id"]

            # Loop over each statement, getting the subject/subject_branch/subject_tag
            subject = statement_data['subject']
            statement_dict = {'contract_id':contract_id,
                              'sentence_num':statement_data['sentence_num'],
                              'statement_num':statement_data['statement_num'],
                              'full_sentence':statement_data['full_sentence'],
                              'full_statement':statement_data['full_statement'],
                              'subject':statement_data['subject'], 'passive':statement_data['passive'],
                              'subject_tags':statement_data['subject_tags'],
                              'subject_branch':statement_data['subject_branch'],    
                              'object_tags':statement_data['object_tags'],
                              'verb':statement_data['verb'], 'modal':statement_data['modal'],
                              'md':statement_data['md'], 'neg':statement_data['neg'],
                              'object_branches':statement_data['object_branches']}

            pdata_rows.append(statement_dict)
            subjectnouns = sorted([x for x, t in zip(statement_data['subject_branch'], statement_data['subject_tags']) if t.startswith('N')])
            subcount[subject] += 1
            if statement_data['md'] == 1:
                modalcount[subject] += 1
            for x in subjectnouns:
                if x != subject:
                    subnouncount[x] += 1
            iteration_num = iteration_num + 1
            # Print a message and save the statements every 100k
            if iteration_num % 100000 == 0:
                print("Iteration ", iteration_num)
                cur_df = pd.DataFrame(pdata_rows)
                cur_df.to_pickle("output_pdata_" + str(chunk_num) + ".pkl")
                chunk_num += 1
                pdata_rows.clear()

    # Make a Pandas df out of whatever's left in pdata_rows and save it
    cur_df = pd.DataFrame(pdata_rows)
    cur_df.to_pickle("output_pdata_" + str(chunk_num) + ".pkl")
    sub_counts_filename = "output_subject_counts.pkl"
    joblib.dump(subcount, sub_counts_filename)
    modal_counts_filename = "output_modal_counts.pkl"
    joblib.dump(subcount, sub_counts_filename)    
    print("most common subjects", subcount.most_common()[:100])
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_directory", type=str, default="")
    # parser.add_argument("--output_directory", type=str, default="")
    # args = parser.parse_args()

    # try:
    #     os.mkdir(os.path.join(args.output_directory, "03_pdata"))
    # except:
    #     pass
    extract_pdata()
