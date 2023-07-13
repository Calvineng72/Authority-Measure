import argparse
import json
import os
import sys
from tqdm import tqdm
import joblib
import io
import spacy
from collections import defaultdict

# command to run the file in the terminal
# python src/main02_parse_articles.py --input_directory cleaned_cba_samples --output_directory output

# subject dependencies
subdeps = {'nsubj', 'nsubj:pass'}

# to be words (conjugations are included for possible lemmatization errors)
to_be = {'estar', 'estará', 'estarão', 'está', 'estão', 'estiverem', 'ser', 'será', 'serão', 'é', 'são', 'for', 'ficar', 'ficará', 'ficarão', 'fica', 'ficam'}

# modal verbs ('ter que' and 'ir' are checked for seperately)
modal_verbs = {'dever', 'deverá', 'deverão', 'deve', 'devem', 'poder', 'poderá', 'poderão', 'pode', 'podem'}

# auxillary verbs to check for
auxillary_verbs = {'ir', 'haver', 'houverem', 'ter', 'tiverem'}
 
def get_statements(art_nlp, contract_id, nlp):
    """
    Extracts statements from the given article's spaCy parsed document.

    Arguments:
        art_nlp: spaCy parsed document of the article
        contract_id: contract ID associated with the article
        art_num: article number

    Returns:
        list of dictionaries, each containing the extracted statement data
        - Each dictionary includes information such as the contract ID, article number,
          sentence number, statement number, and the full sentence text.
    """
    statement_list = []
    
    # for sentence_num, sent in enumerate(art_nlp.sents):
    for sent in art_nlp.sents:
        tokcheck = str(sent).split()

        # checks if statement is less than three tokens
        if len(tokcheck) < 3:
            continue
        
        sent_statements = parse_by_subject(sent, nlp)
        
        # for statement_num, statement_data in enumerate(sent_statements):
        for statement_data in sent_statements:
            full_data = statement_data.copy()
            full_data.update({
                'contract_id': contract_id,
            })
            statement_list.append(full_data)
            
    return statement_list

def parse_article(filename, nlp, args):
    """
    Parses an article file using a given NLP model and saves the extracted statements.

    Arguments:
        filename (str): name of the article file
        nlp (spacy.Language): Spacy NLP model for text processing.
        args (argparse.Namespace): command-line arguments

    Returns:
        None
    """
    statement_list = []
    filepath = os.path.join(args.input_directory, filename)

    if filename.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                art_nlp = nlp(f.read())
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                print(filename)

        contract_id = os.path.basename(filename) 
        art_statements = get_statements(art_nlp, contract_id, nlp)
        statement_list.extend(art_statements)            

    parses_fpath = os.path.join(args.output_directory, "02_parsed_articles", filename[:-3] + "pkl") 
    joblib.dump(statement_list, parses_fpath)
    # with io.open(parses_fpath, 'w', encoding='utf-8') as f:
    #     json.dump(statement_list, f)

def parse_by_subject(sent, nlp):
    """
    Parses a sentence based on its subject and extracts relevant information related to clauses, 
    such as the presence of modal verbs, a negation word, or passive voice constructions.

    Arguments:
        sent: input sentence to parse
        nlp: spaCy natural language processer

    Returns:
        list of dictionaries, each representing a statement with subject-related information

    """
    subjects = [t for t in sent if t.dep_ in subdeps]
    datalist = []

    for subject in subjects:   
        # stores original subject
        orignial_stext = subject.text       
        original_slem = subject.lemma_.lower()

        # stores verb data
        verb = subject.head
        helping_verb = None
        modal = None

        # checks for subject 'que' and skips it
        if original_slem == 'que':
            continue

        # checks for modal and passive verbs
        for child in verb.children:
            # searches for a modal verb and form of 'to be'
            if child.dep_ == 'xcomp' and verb.lemma_.lower() in modal_verbs:
                modal = verb
                verb = child
                for grandchild in child.children:
                    if grandchild.dep_.startswith('aux') or grandchild.dep_ == 'cop':
                        helping_verb = grandchild
                break
            # searches for a complement if the verb is a form of 'to be'
            elif child.dep_ == 'xcomp' and child.tag_ == 'VERB' and verb.lemma_.lower() in to_be:
                for grandchild in child.children:
                    if grandchild.tag_ == 'SCONJ':
                        break
                else:
                    helping_verb = verb
                    verb = child
                    break
                continue
            # searches for an auxillary verb that is a form of 'to be'
            elif child.dep_.startswith('aux') and child.lemma_.lower() in to_be:
                for grandchild in child.children:
                    if grandchild.tag_ == 'SCONJ':
                        break
                else:
                    helping_verb = child
                    break
                continue
            # searches for 'se'
            elif child.dep_ == 'expl' and child.lemma_.lower() == "se":
                helping_verb = child
                break
            # searches for a copulative verb ('to be')
            elif child.dep_ == 'cop':
                verb = child
                break

        vlem = verb.lemma_.lower()
        mlem = modal.lemma_.lower() if modal is not None else ""
        hlem = helping_verb.lemma_.lower() if helping_verb is not None else ""
        verb_text = verb.text
        modal_text = modal.text if modal is not None else ""
        helping_verb_text = helping_verb.text if helping_verb is not None else ""

        # checks for 'ter que' and 'ter de'
        if not mlem: 
            children_lemmas = [child.lemma_.lower() for child in verb.children]
            if 'ter' in children_lemmas and 'que' in children_lemmas:
                children_texts = [child.text for child in verb.children]
                ter_index = children_lemmas.index('ter')
                modal_text, mlem = children_texts[ter_index] + ' que', 'ter que'

        # checks for forms of 'ir,' 'haver,' and 'ter' modifying verb
        if not hlem and mlem != 'ter que':
            for child in verb.children:
                if child.dep_ == 'aux' and child.lemma_.lower() in auxillary_verbs:
                    helping_verb_text, hlem = child.text, child.lemma_.lower()

        # checks for -se-á and -se-ão at the end of a verb
        if not hlem and not mlem:
            if verb_text.endswith('-se-á') or verb_text.endswith('-se-ão'):
                mlem, hlem = 'ir', 'se'
                if verb_text.endswith('-se-á'):
                    vlem = verb_text.replace('-se-á', '').lower()
                else:
                    vlem = verb_text.replace('-se-ão', '').lower()
                # checks for irregular future tense verbs
                vlem = 'fazer' if vlem == 'far' else ('trazer' if vlem == 'trar' else ('dizer' if vlem == 'dir' else vlem))

        # checks for future tense verbs
        if not mlem:
            if verb_text.endswith('rá') or helping_verb_text.endswith('rá') or verb_text.endswith('-á') or  \
                    helping_verb_text.endswith('-á') or verb_text.endswith('rão') or helping_verb_text.endswith('rão') or \
                    verb_text.endswith('-ão') or helping_verb_text.endswith('-ão'):
                mlem = 'ir'

        # checks for -se at the end of or in the middle of a verb phrase
        if not hlem and '-se' in verb_text:
            hlem = 'se'
            verb_stem = verb_text.split('-')[0]
            try:
                # re-lemmatizes the verb without the '-se' ending
                vlem = nlp(verb_stem)[0].lemma_.lower()
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                print(sent)
                continue
                           
        # checks if the verb is negated 
        neg = ''
        neg = 'não' if any(t.text.lower() == 'não' for t in verb.children) else neg
        neg = 'não' if helping_verb and any(t.text.lower() == 'não' for t in helping_verb.children) else neg

        # data structure to store clause information
        data = {'subject': orignial_stext,
                'slem': original_slem,
                'neg': neg,
                'modal': modal_text,
                'mlem': mlem,
                'helping_verb': helping_verb_text,
                'hlem': hlem,
                'verb': verb_text,
                'vlem': vlem,
                'passive': 0,
                'md': 0}
        
        # checks if the sentence is passive
        # (ter + garantido is a common case counted as passive since it translates to 'to be guaranteed')
        if (subject.dep_ == 'nsubj:pass') or (hlem == 'se') or (hlem in to_be and not verb_text.endswith('ndo')) or \
                (hlem == 'ter' and vlem == 'garantir'):
            data['passive'] = 1
        
        # checks if the sentence contains a modal verb
        if mlem != "":
            data['md'] = 1

        datalist.append(data)
    
    return datalist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, default="")
    parser.add_argument("--output_directory", type=str, default="")
    args = parser.parse_args()

    try:
        os.mkdir(args.output_directory)
    except:
        pass
    try:
        os.mkdir(os.path.join(args.output_directory, "02_parsed_articles"))
    except:
        pass

    nlp = spacy.load('pt_core_news_sm', disable=["ner"])
    for filename in tqdm(os.listdir(args.input_directory)):
        parse_article(filename, nlp, args)
