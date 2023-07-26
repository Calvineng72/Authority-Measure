import argparse
from collections import Counter
import os
import pandas as pd
import joblib
import io
import json
from tqdm import tqdm

# command to run the file in the terminal
# python src/main03_get_parse_data.py --input_directory cleaned_cbas --output_directory output
    
def extract_pdata(args):
    """
    Extracts data from parsed articles and saves it into a Pandas DataFrame. Also produces text files with 
    counts of occurrences of modal verbs, subjects, and verbs lemmatized. 

    Args:
        args: object containing the required arguments and settings

    Returns:
        None
    """
    # counters for modals, verbs, and subjects lemmatized
    mlemcount = Counter()
    vlemcount = Counter()
    slemcount = Counter()

    iteration_num = 0
    chunk_num = 0
    pdata_rows = []

    # iterates through each clause and adds its data to Pandas DataFrame
    files = os.listdir(os.path.join(args.output_directory, "02_parsed_articles"))
    filenames = [os.path.join(args.output_directory, "02_parsed_articles", fn) for fn in files]
    for filename in tqdm(filenames, total=len(filenames)):
        for statement_data in joblib.load(filename):
            contract_id = statement_data["contract_id"]

            statement_dict = {'contract_id':contract_id,
                              'subject': statement_data['subject'], 'passive': statement_data['passive'],
                              'helping_verb': statement_data['helping_verb'],
                              'verb': statement_data['verb'], 'vlem': statement_data['vlem'], 
                              'modal': statement_data['modal'], 'mlem': statement_data['mlem'],
                              'md': statement_data['md'], 'neg': statement_data['neg'],
                              'slem': statement_data['slem']}
            if args.clause:
                statement_dict['clause_name'] = statement_data['clause_name']

            pdata_rows.append(statement_dict)

            vlemcount[statement_dict['vlem']] += 1
            mlemcount[statement_dict['mlem']] += 1
            slemcount[statement_dict['slem']] += 1
            
            iteration_num = iteration_num + 1
            if iteration_num % 100000 == 0:
                cur_df = pd.DataFrame(pdata_rows)
                cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl"))
                chunk_num += 1
                pdata_rows.clear()

    # makes a Pandas DataFrame from what is left and saves it
    cur_df = pd.DataFrame(pdata_rows)
    cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl"))

    # creates text files for counts of modals, subjects, and verbs lemmatized
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
    parser.add_argument("--clause", action='store_true')
    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.output_directory, "03_pdata"))
    except:
        pass
    extract_pdata(args)
