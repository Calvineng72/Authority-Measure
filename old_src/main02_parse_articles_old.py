import argparse
import json
import os
from tqdm import tqdm
import joblib
import spacy
import sys
from collections import defaultdict

def articles_as_strlist(contract_data, tuples=True):
    """
    Takes in a list of article_data dictionaries and returns just a list of the
    plaintext of the articles
    """
    def line_as_str(line_dict):
        line_tokens = line_dict["tokens"]
        line_tokens_str = [token["text"] for token in line_tokens]
        return " ".join(line_tokens_str)

    def article_as_str(article_dict):
        art_lines = article_dict["lines"]
        art_lines_str = [line_as_str(cur_line) for cur_line in art_lines]
        return " ".join(art_lines_str)

    contract_id = contract_data["contract_id"]
    art_list = contract_data["articles"]
    if tuples:
        article_strs = [(article_as_str(cur_art), {'contract_id': contract_id, 'article_num':art_num}) 
                        for art_num, cur_art in enumerate(art_list)]
    else:
        article_strs = [article_as_str(cur_art) for cur_art in art_list]
    return article_strs

# subject dependencies
subdeps = ['nsubj','nsubjpass', 'expl']

# main dependencies
maindeps = ['nsubj','nsubjpass', 'expl', 'advmod', 'dobj', 'prep', 'xcomp', 
            'dative', 'advcl', 'agent', 'ccomp', 'acomp', 'attr']
        
def get_branch(t, sent, include_self=True):        
    branch = recurse(t)
    if include_self:
        branch += [t]
            
    branch = [w for w in sent if w in branch]# and w.dep_ in include]

    lemmas = []
    tags = []
    
    for token in branch:
        lemma = token.lemma_.lower()
        #if len(lemma) <= 2:
        #    continue
        if any([char.isdigit() for char in lemma]):
            continue
        if any(punc in lemma for punc in ['.',',',':',';', '-']):
            continue
        lemmas.append(lemma)
        tags.append(token.tag_)
    
    return lemmas, tags

def get_statements(art_nlp, contract_id, art_num, args):
    statement_list = []
    time_in_pbs = 0
    # For now, since spaCy neural coref is buggy, need to check if
    # there are any coref clusters in the doc
    if args.use_neural_coref:
        any_corefs = art_nlp._.coref_clusters is not None
    else:
        any_corefs = False
    
    for sentence_num, sent in enumerate(art_nlp.sents):
        tokcheck = str(sent).split()
        if any([x.isupper() and len(x) > 3 for x in tokcheck]):
            # Don't parse this sentence
            continue
        
        sent_statements = parse_by_subject(sent, resolve_corefs=any_corefs)
        
        for statement_num, statement_data in enumerate(sent_statements):
            full_data = statement_data.copy()
            full_data['contract_id'] = contract_id
            full_data['article_num'] = art_num
            full_data['sentence_num'] = sentence_num
            full_data['statement_num'] = statement_num
            full_data['full_sentence'] = str(sent)
            statement_list.append(full_data)
    return statement_list

def parse_article(filename, nlp, args):
    statement_list = []
    filepath = os.path.join(args.input_directory, filename)

    # if we have json contracts
    if filename.endswith(".json"):
        with open(filepath) as f:
            contract_data = json.load(f)
        # parse_article function, yields a list of tuples [(article_as_string, article_meta_information)]
        # should be overwritten for custom data
        art_data = articles_as_strlist(contract_data)

        # do the actual function
        for text, art_meta in art_data:
            art_nlp = nlp(text)
            contract_id = art_meta["contract_id"]
            art_num = art_meta["article_num"]
            art_statements = get_statements(art_nlp, contract_id, art_num, args)
            statement_list.extend(art_statements)            
    # otherwise, if we have contracts as .txt files
    elif filename.endswith(".txt"):
        with open(filepath) as f:
            art_nlp = nlp(f.read())
        art_num = 0 # all article_numbers are 0, because we haven't split contracts into articles
        contract_id = os.path.basename(filename) # contract_id is just the filename
        art_statements = get_statements(art_nlp, contract_id, art_num, args)
        statement_list.extend(art_statements)                   
    # fn[:-4] strips the "json" ending and replaces with ".pkl" ending
    parses_fpath = os.path.join(args.output_directory, "02_parsed_articles", filename[:-4] + "pkl") 
    joblib.dump(statement_list, parses_fpath)


def parse_by_subject(sent, resolve_corefs=True):
    subjects = [t for t in sent if t.dep_ in subdeps]

    ## Only for debugging
    #for cur_sub in subjects:
    #    if "Board" == str(cur_sub):
    #        print(all_tokens)

    datalist = []

    # Each subject corresponds to a statement that it is the subject of.
    # Hence this is a loop over *statements*
    for obnum, subject in enumerate(subjects):   
        subdep = subject.dep_        
        mlem = None
        verb = subject.head
        if not verb.dep_.startswith('V'):
            continue        
                
        vlem = verb.lemma_
        
        tokenlists = defaultdict(list)                        
        neg = ''
        for t in verb.children:
            if t.tag_ == 'MD':
                mlem = t.orth_.lower()
                continue
            dep = t.dep_
            if dep in ['punct','cc','det', 'meta', 'intj', 'dep']:
                continue
            if dep == 'neg':
                neg = 'not'                
            elif t.dep_ == 'prt':
                vlem = vlem + '_' + t.orth_.lower()                    
            else:
                tokenlists[dep].append(t)
                
        slem = subject.lemma_
        in_coref = False
        cr_subject = subject.text
        cr_slem = slem
        num_clusters = 0
        coref_replaced = False
        if resolve_corefs:
            in_coref = subject._.in_coref
            # Now check if it's *different* from the coref cluster's main coref
            if in_coref:
                coref_clusters = subject._.coref_clusters
                num_clusters = len(coref_clusters)
                first_cluster = coref_clusters[0]
                # Get the main of this first cluster
                #cluster_main_lem = first_cluster.main.lemma_
                cluster_main_lem = first_cluster.main.root.lemma_
                if slem != cluster_main_lem:
                    # Replace it with main!
                    cr_slem = cluster_main_lem
                    coref_replaced = True
                    cr_subject = cluster_main_lem

        data = {'orig_subject': subject.text,
                'orig_slem': slem,
                'in_coref': in_coref,
                'subject': cr_subject,
                'slem': cr_slem,
                'coref_replaced': coref_replaced,
                'modal':mlem,
                'neg': neg,
                'verb': vlem,
                #'full_sentence': str(sent),
                #'subfilter': 0,
                'passive': 0,
                'md': 0}
        
        if subdep == 'nsubjpass':
            data['passive'] = 1
        if mlem is not None:
            data['md'] = 1
        
        subphrase, subtags = get_branch(subject,sent)                                        
        
        data['subject_branch'] = subphrase        
        data['subject_tags'] = subtags
        
        object_branches = []
        object_tags = []
        
        for dep, tokens in tokenlists.items():
            if dep in subdeps:
                continue
            for t in tokens:
                tbranch, ttags = get_branch(t,sent)                
                object_branches.append(tbranch)
                object_tags.append(ttags)
        data['object_branches'] = object_branches
        data['object_tags'] = object_tags

        data['full_statement'] = ""
        
        # So upon being added to datalist, the "data" dictionary has the following
        # keys: 'orig_subject','orig_slem','in_coref','subject', 'slem',"modal",
        # "neg","verb","passive","md","subject_branch","subject_tags",
        # "object_branches", "object_tags", "full_statement" (empty string for now)

        datalist.append(data)
    
    return datalist

def recurse(*tokens):
    children = []
    def add(tok):       
        sub = tok.children
        for item in sub:
            children.append(item)
            add(item)
    for token in tokens:
        add(token)    
    return children



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, default="")
    parser.add_argument("--output_directory", type=str, default="")
    parser.add_argument("--use_neural_coref", action='store_true')
    args = parser.parse_args()

    try:
        os.mkdir(args.output_directory)
    except:
        pass
    try:
        os.mkdir(os.path.join(args.output_directory, "02_parsed_articles"))
    except:
        pass

    nlp = spacy.load('en_core_web_sm', disable=["ner"])
    if args.use_neural_coref:
        neuralcoref.add_to_pipe(nlp)
    for filename in tqdm(os.listdir(args.input_directory)):
        parse_article(filename, nlp, args)
