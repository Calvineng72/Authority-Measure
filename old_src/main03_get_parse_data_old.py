# -*- coding: utf-8 -*-

from collections import Counter
import os
import argparse
import pandas as pd
import joblib
from tqdm import tqdm
## REMOVE THIS
import io
import json

                        
other = ['agreement',
         'day',
         'assignment','leave','payment',
         'he/she','they','party']

conditionals = ['if','when','unless']
	#'where', 'whereas','whenever', 'provided that', 'in case']
    
    
def extract_pdata(args):
    subcount = Counter()
    subnouncount = Counter()
    modalcount = Counter()

    ## REMOVE THIS
    slemcount = Counter()
    vlemcount = Counter()
    mlemcount = Counter()

    num_to_process = len(os.listdir(args.input_directory))

    iteration_num = 0
    chunk_num = 0
    pdata_rows = []

    files = os.listdir(os.path.join(args.output_directory, "02_parsed_articles"))
    filenames = [os.path.join(args.output_directory, "02_parsed_articles", fn) for fn in files]
    for filename in tqdm(filenames, total=len(filenames)):
        for statement_data in joblib.load(filename):
            contract_id = statement_data["contract_id"]
            # skip french contracts
            if not "eng" in contract_id:
                continue
            # Loop over each statement, getting the subject/subject_branch/subject_tag
            subject = statement_data['subject']
            statement_dict = {'contract_id':contract_id,
            'article_num':statement_data['article_num'],
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
            subjectnouns = sorted([x for x,t in zip(statement_data['subject_branch'], statement_data['subject_tags']) if t.startswith('N')])
            subcount[subject] += 1

            # TO REMOVE
            vlemcount[statement_dict['verb']] += 1
            mlemcount[statement_dict['modal']] += 1
            slemcount[statement_dict['subject']] += 1

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
                cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl"))
                chunk_num += 1
                pdata_rows.clear()

    # Make a Pandas df out of whatever's left in pdata_rows and save it
    cur_df = pd.DataFrame(pdata_rows)
    cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl"))
    sub_counts_filename = os.path.join(args.output_directory, "subject_counts.pkl")
    joblib.dump(subcount, sub_counts_filename)
    modal_counts_filename = os.path.join(args.output_directory, "modal_counts.pkl")
    joblib.dump(subcount, sub_counts_filename)    
    print("most common subjects", subcount.most_common()[:100])

    ## REMOVE THIS ONCE DONE
    slem_counts_filename = os.path.join(args.output_directory, "slem_counts.txt")
    # joblib.dump(slemcount, slem_counts_filename)    
    with io.open(slem_counts_filename, 'w', encoding='utf-8') as f:
        json.dump(slemcount.most_common(), f, ensure_ascii=False)
    vlem_counts_filename = os.path.join(args.output_directory, "vlem_counts.txt")
    # joblib.dump(vlemcount, vlem_counts_filename)    
    with io.open(vlem_counts_filename, 'w', encoding='utf-8') as f:
        json.dump(vlemcount.most_common(), f, ensure_ascii=False)
    mlem_counts_filename = os.path.join(args.output_directory, "mlem_counts.txt")
    # joblib.dump(mlemcount, mlem_counts_filename)    
    with io.open(mlem_counts_filename, 'w', encoding='utf-8') as f:
        json.dump(mlemcount.most_common(), f, ensure_ascii=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, default="")
    parser.add_argument("--output_directory", type=str, default="")
    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.output_directory, "03_pdata"))
    except:
        pass
    extract_pdata(args)
