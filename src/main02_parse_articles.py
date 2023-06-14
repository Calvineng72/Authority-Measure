import argparse
import json
import os
import sys
from tqdm import tqdm
import joblib
import io
import spacy
from collections import defaultdict

# python src/main02_parse_articles.py --input_directory cleaned_cba_samples --output_directory output

# subject dependencies
subdeps = ['nsubj', 'nsubj:pass']

# to be words 
to_be = ['estar', 'ser', 'é', 'são', 'será', 'ficar', 'ficam', 'fica']

# modal verbs ('ter que' and 'ir' are checked for seperately)
modal_verbs = ['dever', 'poder']
        
def get_branch(t, sent, include_self=True):       
    """
    Retrieves the branch of tokens associated with a given token in a sentence.

    Arguments:
        t: token to retrieve the branch for
        sent: sentence containing the tokens
        include_self: optional parameter to include the token itself in the branch (default=True)

    Returns:
        tuple of two lists: the lemmas and tags of the tokens in the branch
    """
    branch = recurse(t)
    
    if include_self:
        branch.append(t)

    lemmas = []
    tags = []
    
    for token in sent:
        if token in branch:               
            lemma = token.lemma_.lower()

            if not any(char.isdigit() for char in lemma) and not any(punc in lemma for punc in ['.', ',', ':', ';', '-']):
                lemmas.append(lemma)
                tags.append(token.tag_)
    
    return lemmas, tags

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
                # 'sentence_num': sentence_num,
                # 'statement_num': statement_num,
                # 'full_sentence': str(sent)
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
        with open(filepath, encoding='utf-8') as f:
            art_nlp = nlp(f.read())
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
        resolve_corefs: flag indicating whether to resolve corefs (default=False)

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

        # # stores new subject data (if needed)
        # correct_stext = orignial_stext
        # correct_slem = original_slem
        # coref_replaced = False

        # checks for subject 'que'
        if original_slem == 'que':
            continue
        # if subject.text == 'que':
        #     for ancestor in subject.ancestors:
        #         if (ancestor.tag_ == 'NOUN' or ancestor.tag_ == 'PRON' or ancestor.tag_ == 'PROPN') and (ancestor not in subjects):
        #             if ancestor in verb.ancestors and verb in ancestor.children:
        #                 correct_stext = ancestor.text
        #                 correct_slem = ancestor.lemma_.lower()
        #                 coref_replaced = True
        #     if not coref_replaced:
        #         continue

        # checks for modal and passive verbs
        for child in verb.children:
            if child.dep_ == 'xcomp' and verb.lemma_.lower() in modal_verbs:
                modal = verb
                verb = child
                for child_child in child.children:
                    if child_child.dep_.startswith('aux'):
                        helping_verb = child_child
                        break
                break
            elif child.dep_ == 'xcomp' and child.tag_ == 'VERB' and verb.lemma_.lower() in to_be:
                helping_verb = verb
                verb = child
            elif child.dep_.startswith('aux') and child.lemma_.lower() in to_be:
                helping_verb = child
                break
            elif child.dep_ == 'expl' and child.lemma_.lower() == "se":
                helping_verb = child
                break
            elif child.dep_ == 'cop':
                verb = child
                break

        vlem = verb.lemma_.lower()
        mlem = modal.lemma_.lower() if modal is not None else ""
        hlem = helping_verb.lemma_.lower() if helping_verb is not None else ""
        verb_text = verb.text
        modal_text = modal.text if modal is not None else ""
        helping_verb_text = helping_verb.text if helping_verb is not None else ""

        # checks for 'ter que'
        if modal is None: 
            children_lemmas = [child.lemma_.lower() for child in verb.children]
            children_texts = [child.text for child in verb.children]
            if 'ter' in children_lemmas and 'que' in children_lemmas:
                ter_index = children_lemmas.index('ter')
                modal_text, mlem = children_texts[ter_index] + ' que', 'ter que'

        # checks for future tense verbs
        if modal is None:
            if verb_text.endswith('rá') or helping_verb_text.endswith('rá'):
                mlem = 'ir'
            elif verb_text.endswith('rão') or helping_verb_text.endswith('rão'):
                mlem = 'ir'

        # checks for -se-á and -se-ão at the end of a verb
        if helping_verb is None and modal is None:
            if verb_text.endswith('-se-á'):
                mlem = 'ir'
                hlem = 'se'
                verb_stem = verb_text.replace('-se-á', '')
                vlem = nlp(verb_stem)[0].lemma_.lower()
            elif verb_text.endswith('-se-ão'):
                mlem = 'ir'
                hlem = 'se'
                verb_stem = verb_text.replace('-se-ão', '')
                vlem = nlp(verb_stem)[0].lemma_.lower()

        # checks for -se at the end of or in the middle of a verb phrase
        if helping_verb is None and '-se' in verb_text:
            hlem = 'se'
            verb_stem = verb_text.split('-')[0]
            vlem = nlp(verb_stem)[0].lemma_.lower()
        
        tokenlists = defaultdict(list)                        
        neg = ''
        for t in verb.children:
            dep = t.dep_
            if dep in ['punct','cc','det', 'meta', 'intj', 'dep']:
                continue
            elif t.text.lower() == 'não':
                neg = 'não'      
            else:
                tokenlists[dep].append(t)
        if helping_verb is not None:
            for t in helping_verb.children:
                if t.text.lower() == 'não':
                    neg = 'não'

        # data = {'orig_subject': orignial_stext,
        #         'orig_slem': original_slem,
        #         'subject': correct_stext,
        #         'slem': correct_slem,
        #         'coref_replaced': coref_replaced,
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
        
        if (hlem == 'se') or (hlem in to_be and not verb_text.endswith('ndo')):
            data['passive'] = 1
        if mlem != "":
            data['md'] = 1
        
        # subject and object branches
        # subphrase, subtags = get_branch(subject, sent)                                        
        
        # data['subject_branch'] = subphrase        
        # data['subject_tags'] = subtags
        
        # object_branches = []
        # object_tags = []
        
        # for dep, tokens in tokenlists.items():
        #     if dep in subdeps:
        #         continue
        #     for t in tokens:
        #         tbranch, ttags = get_branch(t,sent)                
        #         object_branches.append(tbranch)
        #         object_tags.append(ttags)

        # data['object_branches'] = object_branches
        # data['object_tags'] = object_tags
        # data['full_statement'] = ''

        datalist.append(data)
    
    return datalist

def recurse(*tokens):
    """
    Recursively collects the children of the given tokens.
    
    Arguments:
        *tokens: variable number of tokens to collect the children from
        
    Returns:
        list containing all the children of the input tokens
    """
    children = []

    def add(tok):
        sub = tok.children
        children.extend(sub)
        for item in sub:
            add(item)
 
    for token in tokens:
        add(token)

    return children


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
