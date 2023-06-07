import argparse
from collections import Counter
import os
import pandas as pd
import joblib
import io
import json
from tqdm import tqdm

# python src/main03_get_parse_data.py --input_directory cleaned_cba_samples --output_directory output
    
def extract_pdata(args):
    """
    Extracts data from parsed articles and saves it into a Pandas DataFrame.

    Args:
        args: object containing the required arguments and settings

    Returns:
        None
    """
    subcount = Counter()
    subnouncount = Counter()
    modalcount = Counter()

    mlemcount = Counter() # TO REMOVE
    vlemcount = Counter() # TO REMOVE

    iteration_num = 0
    chunk_num = 0
    pdata_rows = []

    files = os.listdir(os.path.join(args.output_directory, "02_parsed_articles"))
    filenames = [os.path.join(args.output_directory, "02_parsed_articles", fn) for fn in files]
    for filename in tqdm(filenames, total=len(filenames)):
        for statement_data in joblib.load(filename):
            contract_id = statement_data["contract_id"]

            # loops over each statement, getting the subject/subject_branch/subject_tag
            subject = statement_data['subject']
            statement_dict = {'contract_id':contract_id,
                              'sentence_num':statement_data['sentence_num'],
                              'full_sentence':statement_data['full_sentence'],
                              'full_statement':statement_data['full_statement'],
                              'subject':statement_data['subject'], 'passive':statement_data['passive'],
                              'subject_tags':statement_data['subject_tags'],
                              'subject_branch':statement_data['subject_branch'],    
                              'object_tags':statement_data['object_tags'],
                              'verb':statement_data['vlem'], 'modal':statement_data['mlem'], # CHANGED TO MLEM FROM MODAL and VLEM FROM VERB
                              'md':statement_data['md'], 'neg':statement_data['neg'],
                              'slem':statement_data['slem'],
                              'object_branches':statement_data['object_branches']}

            pdata_rows.append(statement_dict)
            subjectnouns = sorted([x for x, t in zip(statement_data['subject_branch'], statement_data['subject_tags']) if t.startswith('N')])
            subcount[subject] += 1

            # TO REMOVE
            vlemcount[statement_dict['verb']] += 1
            mlemcount[statement_dict['modal']] += 1

            if statement_data['md'] == 1:
                modalcount[subject] += 1
            for x in subjectnouns:
                if x != subject:
                    subnouncount[x] += 1
            
            iteration_num = iteration_num + 1
            if iteration_num % 100000 == 0:
                print("Iteration ", iteration_num)
                cur_df = pd.DataFrame(pdata_rows)
                cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl"))
                chunk_num += 1
                pdata_rows.clear()

    # makes a Pandas df from what is left and saves it
    cur_df = pd.DataFrame(pdata_rows)
    cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl")) # CHANGE BACK TO PICKLE
    sub_counts_filename = os.path.join(args.output_directory, "subject_counts.pkl")
    joblib.dump(subcount, sub_counts_filename)
    # with io.open(sub_counts_filename, 'w', encoding='utf8') as f:
    #     json.dump(subcount, f)
    modal_counts_filename = os.path.join(args.output_directory, "modal_counts.pkl")
    joblib.dump(modalcount, modal_counts_filename)    
    # with io.open(modal_counts_filename, 'w', encoding='utf8') as f:
    #     json.dump(modalcount, f)
    print("most common subjects", subcount.most_common()[:100])
    print("most common modals", mlemcount.most_common()[:100]) # TO REMOVE
    print("most common verbs", vlemcount.most_common()[:100]) # TO REMOVE
    

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
