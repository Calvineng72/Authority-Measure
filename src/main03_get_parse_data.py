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
    modalcount = Counter()
    mlemcount = Counter()
    vlemcount = Counter()
    slemcount = Counter()

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
                            #   'sentence_num':statement_data['sentence_num'],
                            #   'full_sentence':statement_data['full_sentence'],
                            #   'full_statement':statement_data['full_statement'],
                              'subject': statement_data['subject'], 'passive': statement_data['passive'],
                            #   'subject_tags':statement_data['subject_tags'],
                            #   'subject_branch':statement_data['subject_branch'],    
                            #   'object_tags':statement_data['object_tags'],
                            #   'object_branches':statement_data['object_branches'],
                              'helping_verb': statement_data['helping_verb'],
                              'verb': statement_data['verb'], 'vlem': statement_data['vlem'], 
                              'modal': statement_data['modal'], 'mlem': statement_data['mlem'],
                              'md': statement_data['md'], 'neg': statement_data['neg'],
                              'slem': statement_data['slem']}

            pdata_rows.append(statement_dict)
            # subjectnouns = sorted([x for x, t in zip(statement_data['subject_branch'], statement_data['subject_tags']) if t.startswith('N')])
            subcount[subject] += 1

            # TO REMOVE
            vlemcount[statement_dict['vlem']] += 1
            mlemcount[statement_dict['mlem']] += 1
            slemcount[statement_dict['slem']] += 1

            if statement_data['md'] == 1:
                modalcount[subject] += 1
            # for x in subjectnouns:
            #     if x != subject:
            #         subnouncount[x] += 1
            
            iteration_num = iteration_num + 1
            if iteration_num % 100000 == 0:
                cur_df = pd.DataFrame(pdata_rows)
                cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl"))
                chunk_num += 1
                pdata_rows.clear()

    # makes a Pandas df from what is left and saves it
    cur_df = pd.DataFrame(pdata_rows)
    cur_df.to_pickle(os.path.join(args.output_directory, "03_pdata", "pdata_" + str(chunk_num) + ".pkl"))

    # sub_counts_filename = os.path.join(args.output_directory, "subject_counts.pkl")
    # # joblib.dump(subcount, sub_counts_filename)
    # with io.open(sub_counts_filename, 'w', encoding='utf-8') as f:
    #     json.dump(subcount, f, ensure_ascii=False)
    # modal_counts_filename = os.path.join(args.output_directory, "modal_counts.pkl")
    # # joblib.dump(modalcount, modal_counts_filename)    
    # with io.open(modal_counts_filename, 'w', encoding='utf-8') as f:
    #     json.dump(modalcount, f)

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

    print("most common subjects (slem)", slemcount.most_common()[:100]) 
    print("most common modals (mlem)", mlemcount.most_common()[:100])
    print("most common verbs (vlem)", vlemcount.most_common()[:100])


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
