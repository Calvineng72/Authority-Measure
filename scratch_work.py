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
        # if modal_text == 'poderá' or modal_text == 'poderão':
        #     verb_text, vlem = modal_text, 'poder'
        #     modal_text, mlem = '', 'ir'

        # checks for -se-á and -se-ão at the end of a verb
        if helping_verb is None and modal is None:
            if verb_text.endswith('-se-á'):
                mlem = 'ir'
                hlem = 'se'
                verb_stem = verb_text.replace('-se-á', '')
                try:
                    vlem = nlp(verb_stem)[0].lemma_.lower()
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    print(sent)
                    continue
            elif verb_text.endswith('-se-ão'):
                mlem = 'ir'
                hlem = 'se'
                verb_stem = verb_text.replace('-se-ão', '')
                try:
                    vlem = nlp(verb_stem)[0].lemma_.lower()
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    print(sent)
                    continue

        # checks for -se at the end of or in the middle of a verb phrase
        if helping_verb is None and '-se' in verb_text:
            hlem = 'se'
            verb_stem = verb_text.split('-')[0]
            try:
                vlem = nlp(verb_stem)[0].lemma_.lower()
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                print(sent)
                continue

        
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

    # print("most common subjects (slem)", slemcount.most_common()[:100]) 
    # print("most common modals (mlem)", mlemcount.most_common()[:100])
    # print("most common verbs (vlem)", vlemcount.most_common()[:100])


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



import argparse
import os
import pandas as pd
from tqdm import tqdm

# python src/main04_compute_auth.py --input_directory cleaned_cba_samples --output_directory output

def combine_auth(args):
    """
    Combines authority data chunks into a single DataFrame and saves it.

    Arguments:
        args: object containing the required arguments and settings

    Returns:
        None
    """
    filepath = os.path.join(args.output_directory, "04_auth")
    chunks = os.listdir(filepath)
    chunks = sorted(chunks, key=lambda x: int(x.split("_")[-1][:-4]))

    auth_df = pd.DataFrame()
    for filename in chunks:
        cur_df = pd.read_pickle(os.path.join(filepath, filename))
        auth_df = pd.concat([auth_df,cur_df])

    auth_df.to_pickle(os.path.join(args.output_directory, "04_auth.pkl"))

# worker = ['empregado', 'trabalhador', 'motorista', 'funcionário', 'empregada', 'empregados-jornalistas', 
#           'equipe', 'empregados-jornalista', 'professor', 'enfermeiro', 'mecânico', 'operador', 'comissário', 'pessoal',
#           'docente', 'professor', 'contratado', 'jornalista', 'aprendiz', 'empregados', 'empregadas', 'médico', 'empregar']
# firm = ['empregador', 'empresa', 'conselho', 'hospital', 'corporação', 'proprietário', 'superintendente', 'empregadora', 
#         'companhia', 'firma', 'empresas', 'concessionária', 'empresário'] 
# union = ['sindicato', 'associação', 'membro', 'representante', 'dirigente']
# manager = ['gerente', 'gestão', 'administração', 'administrador', 'supervisor', 'diretor', 'principal', 'gestor']

worker = ['admitida', 'admitidas', 'admitido', 'admitidos', 'aposentada', 'aposentadas', 'aposentado', 'aposentados', 
          'aprendiz', 'aprendizes', 'contratada', 'contratadas', 'contratado', 'contratados', 'dimitida', 'dimitidas', 
          'dimitido', 'dimitidos', 'empregada', 'empregadas', 'empregado', 'empregados', 'empregar', 'estagiária', 
          'estagiárias', 'estagiário', 'estagiários', 'funcionária', 'funcionárias', 'funcionário', 'funcionários',
          'pessoal', 'suplente', 'suplentes', 'trabalhador', 'trabalhadora', 'trabalhadoras', 'trabalhadores', 'operário',
          'operários', 'operária', 'operárias', 'motorista', 'motoristas', 'professor', 'professores', 'professora', 'professoras',
          'gestante', 'gestantes', 'cobrador', 'cobradores', 'docente', 'docentes', 'estudante', 'estudantes', 
          'colaborador', 'colaboradora', 'colaboradores', 'colaboradoras', 'operador', 'operadora', 'operadores', 'operadoras',
          'auxiliar', 'auxiliares', 'jornalista', 'jornalistas', 'vendedor', 'vendedora', 'vendedores', 'vendedoras', 
          'servidor', 'servidora', 'servidores', 'servidoras', 'participante', 'participantes', 'dependente', 'dependentes',
          'comissionista', 'comissionistas', 'aposentado', 'aposentada', 'aposentados', 'aposentadas', 'acidentado', 'acidentada',
          'acidentados', 'acidentadas', 'substituto', 'substituta', 'substitutos', 'substitutas']
firm = ['companhia', 'companhias', 'concessionária', 'concessionárias', 'concessionário', 'concessionários', 'corporação', 
        'corporações', 'corporativa', 'corporativas', 'corporativo', 'corporativos', 'empregador', 'empregadora', 'empregadoras', 
        'empregadores', 'empresa', 'empresar', 'empresária', 'empresárias', 'empresário', 'empresários', 'empresas', 
        'estabelecimento', 'estabelecimentos', 'firma', 'firmas', 'patrão', 'patroa', 'patroas', 'patrões', 'proprietária',
        'proprietárias', 'proprietário', 'proprietários', 'contratante', 'contratantes', 'hospital', 'hospitais', 'escola',
        'escolas']
union = ['confederação', 'confederações', 'cooperativa', 'cooperativas', 'delegado', 'delegados', 'dirigente', 'dirigentes', 
         'federação', 'federações', 'grêmio', 'líder', 'líderes', 'representante', 'representantes', 'sindicato', 'sindicatos',
         'cipa', 'cipeiro', 'sindicalizado', 'sindicalizados', 'sindicalizada', 'sindicalizadas', 'assembleia', 'assembleias']
manager = ['chefe', 'chefes', 'diretor', 'diretora', 'diretoras', 'diretores', 'diretoria', 'diretorias', 'gerência', 'gerências',
           'gerenciador', 'gerenciadora', 'gerenciadoras', 'gerenciadores', 'gerente', 'gerentes', 'manager', 'managers', 
           'superintendência', 'superintendente', 'superintendentes', 'supervisor', 'supervisora', 'supervisoras', 'supervisores',
           'conselho', 'conselhos']

subnorm_map = {}
for i in worker:
    subnorm_map[i] = "worker"
for i in firm:
    subnorm_map[i] = "firm"
for i in union:
    subnorm_map[i] = "union"
for i in manager:
    subnorm_map[i] = "manager"

def normalize_subject(subject):
    """
    Normalizes a subject by mapping it to predefined subnorm_map.

    Arguments:
        subject: subject to be normalized

    Returns:
        The normalized form of the subject if it exists in the subnorm_map, otherwise returns "other".
    """
    if subject in subnorm_map:
        return subnorm_map[subject]
    else:
        return "other_agent"

def check_strict_modal(statement_row):
    """
    Checks if a statement row contains a strict modal verb.

    Arguments:
        statement_row: row of statement data containing information about a specific statement

    Returns:
        True if the statement row contains a strict modal verb, False otherwise.
    """
    strict_modal = False
    if statement_row['md']:
        strict_modal = statement_row['mlem'] in ['dever', 'ter que', 'ir']
    return strict_modal

def check_neg(statement_row):
    """
    Checks if a statement row contains a negation.

    Arguments:
        statement_row: row of statement data containing information about a specific statement

    Returns:
        True if the statement row contains a negation, False otherwise.
    """
    return statement_row['neg'] == 'não'

def compute_statement_auth(args, df, filename):
    """
    Computes the authority of each statement in the given DataFrame.

    Arguments:
        args: object containing additional arguments or configuration settings
        df: DataFrame containing the statement data
        filename: filename of the output file

    Returns:
        None
    """
    vars_to_keep = ["contract_id", "slem", "subject", "verb", "vlem",
                    "modal", "mlem", "md", "helping_verb", "passive", "neg"]

    df = df[vars_to_keep]
    df["md"] = df["md"].astype('bool')
    df["passive"] = df["passive"].astype('bool')
    df["subject"] = df["subject"].str.lower()
    df["subnorm"] = df["slem"].apply(normalize_subject)
    df["strict_modal"] = df.apply(check_strict_modal, axis=1).astype('bool')
    df["neg"] = df['neg'].apply(lambda x: x == 'não').astype('bool')

    # permissive modals are may and can
    df['permissive_modal'] = (df['md'] & ~df['strict_modal']).astype('bool')

    df_passive = df['passive']
    df_notpassive = ~df_passive
    df_neg = df['neg']
    df_notneg = ~df_neg

    # obligation verbs 
    df['obligation_verb'] = ((df_passive & df['vlem'].isin(['exigir', 'esperar', 'coagir', 'obrigar', 'compelir', 'obrigado',
                                                            'forçar', 'requerer'])) 
                             | (df_notpassive & df['vlem'].isin(['concordar', 'prometer', 'consentir', 'aquiescer']))).astype('bool')

    # constraint verbs 
    df['constraint_verb'] = (df_passive & 
                      df['vlem'].isin(['proibir', 'vedar', 'banir', 'impedir', 'restringir', 'proscrever', 'limitar',
                                       'impossibilitar', 'negar'])).astype('bool')
        
    # permissiion verbs
    df['permission_verb'] = (df_passive & df['vlem'].isin(['permitir', 'autorizar', 'aprovar', 'habilitar'])).astype('bool')

    # entitlement verbs    
    df['entitlement_verb'] =  ((df_notpassive & df['vlem'].isin(['receber', 'ganhar', 'obter', 'gozar', 'beneficiar', 'repousar'])) 
                               | (df_passive & df['vlem'].isin(['conceder', 'dar', 'oferecer', 'reembolsar', 'pagar', 'outorgar', 
                                                                'fornecer', 'compensar', 'garantir', 'contratar', 'treinar', 'suprir', 
                                                                'proteger', 'cobrir', 'informar', 'notificar', 'selecionar', 
                                                                'entregar', 'proteger', 'pagar', 'premiar',
                                                                'facultar', 'garantido', 'proporcionar', 'prestar', 'propiciar',
                                                                'providenciar', 'fornecir']))).astype('bool')

    # promise verbs
    df['promise_verb'] = (df_notpassive & 
                  df['vlem'].isin(['reconhecer',
                              'consentir', 'afirmar',
                              'garantir', 'segurar', 'assegurar', 'estipular',
                              'assumir'])).astype('bool')

    # special verbs
    df['special_verb'] = (df['obligation_verb'] | df['constraint_verb'] | df['permission_verb'] | df['entitlement_verb'] | df['promise_verb']).astype('bool')
      
    # active verbs 
    df['active_verb'] = (df_notpassive & ~df['special_verb']).astype('bool')
        
    # provisions
    df['obligation_1'] = (df_notneg & df['strict_modal'] & df['active_verb']).astype('bool')
    df['obligation_2'] = (df_notneg & ~df['permissive_modal'] & df['obligation_verb']).astype('bool')  
    df['obligation'] = (df['obligation_1'] | df['obligation_2']).astype('bool')
    df['constraint_1'] = (df_neg & df['md'] & df['active_verb']).astype('bool')
    df['constraint_2'] = (df_notneg & df['strict_modal'] & df['constraint_verb']).astype('bool')
    df['constraint_3'] = (df_neg & df_passive & (df['entitlement_verb'] | df['permission_verb'])).astype('bool')
    df['constraint'] = (df['constraint_1'] | df['constraint_2'] | df['constraint_3']).astype('bool')
    df['permission_1'] = (df_notneg & df['permissive_modal'] & df['active_verb']).astype('bool')
    df['permission_2'] = (df_notneg & df['permission_verb']).astype('bool')
    df['permission_3'] = (df['neg'] & df['constraint_verb']).astype('bool')
    df['permission'] = (df['permission_1'] | df['permission_2'] | df['permission_3']).astype('bool')
    df['entitlement_1'] = (df_notneg & df['entitlement_verb']).astype('bool')
    df['entitlement_2'] = (df_neg & df['obligation_verb']).astype('bool')
    df['entitlement'] = (df['entitlement_1'] | df['entitlement_2']).astype('bool')
    df['other_provision'] = ~(df['obligation'] | df['constraint'] | df['permission'] | df['entitlement']).astype('bool')

    # df['obligation'] = ((df_notneg & df['strict_modal'] & df['active_verb']) |     
    #                     (df_notneg & ~df['permissive_modal'] & df['obligation_verb'])).astype('bool')           
    # df['constraint'] = ((df_neg & df['md'] & df['active_verb']) |
    #                     (df_notneg & df['strict_modal'] & df['constraint_verb']) | 
    #                     (df_neg & df_passive & (df['entitlement_verb'] | df['permission_verb'] ))).astype('bool')
    # df['permission'] = ((df_notneg & ((df['permissive_modal'] & df['active_verb']) | df['permission_verb'])) | 
    #                     (df_neg & df['constraint_verb'])).astype('bool')
    # df['entitlement'] = ((df_notneg & df['entitlement_verb']) |
    #                      (df_neg & df['obligation_verb'])).astype('bool')  
    # df['other_provision'] = ~(df['obligation'] | df['constraint'] | df['permission'] | df['entitlement'])
    
    df.to_pickle(os.path.join(args.output_directory, "04_auth", filename.replace("pdata_", "auth_")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, default="")
    parser.add_argument("--output_directory", type=str, default="")
    args = parser.parse_args()
    try:
        os.mkdir(os.path.join(args.output_directory, "04_auth"))
    except:
        pass
    chunks = os.listdir(os.path.join(args.output_directory, "03_pdata"))
    chunks = sorted(chunks, key=lambda x: int(x.split("_")[-1][:-4]))
    for filename in tqdm(chunks):
        filepath = os.path.join(args.output_directory, "03_pdata", filename)
        cur_df = pd.read_pickle(filepath)
        compute_statement_auth(args, cur_df, filename)
    combine_auth(args)



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
		df_prefixes = df[['obligation', 'constraint', 'permission', 'entitlement', 'other_provision', 'vlem', 
		    'obligation_1', 'obligation_2', 'constraint_1', 'constraint_2', 'constraint_3', 'permission_1', 
			'permission_2', 'permission_3', 'entitlement_1', 'entitlement_2']]
		provisions = ['obligation', 'constraint', 'permission', 'entitlement', 'other_provision', 'obligation_1',
			'obligation_2', 'constraint_1', 'constraint_2', 'constraint_3', 'permission_1', 
			'permission_2', 'permission_3', 'entitlement_1', 'entitlement_2']
		df_prefixes[provisions] = df_prefixes[provisions].astype(int)
		
		# replaces boolean values with text
		df['neg'] = df['neg'].apply(lambda x: 'não' if x else '')

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
