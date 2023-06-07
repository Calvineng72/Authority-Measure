import argparse
import os
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

def combine_auth(args):
    filepath = os.path.join(args.output_directory, "04_auth")
    chunks = os.listdir(filepath)
    chunks = sorted(chunks, key=lambda x: int(x.split("_")[-1][:-4]))

    auth_df = pd.DataFrame()
    for filename in chunks:
        cur_df = pd.read_pickle(os.path.join(filepath, filename))
        auth_df = pd.concat([auth_df,cur_df])

    auth_df.to_pickle(os.path.join(args.output_directory, "04_auth.pkl"))

# worker="employee,worker,staff,teacher,nurse,mechanic,operator,steward,personnel".split(",")
worker = ['empregado', 'trabalhador', 'motorista', 'funcionário', 'empregada', 'empregados-jornalistas', 
          'equipe', 'empregados-jornalista', 'professor', 'enfermeiro', 'mecânico', 'operador', 'comissário', 'pessoal',
          'empregada']
# firm="employer,company,board,hospital,corporation,owner,superintendent".split(",")
firm = ['empregador', 'empresa', 'conselho', 'hospital', 'corporação', 'proprietário', 'superintendente', 'empregadora']
# union="union,association,member,representative".split(",")
union = ['sindicato', 'associação', 'membro', 'representante']
# manager="manager,management,administration,administrator,supervisor,director,principal".split(",")
manager = ['gerente', 'gestão', 'administração', 'administrador', 'supervisor', 'diretor', 'principal']

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
    subject = str(subject)
    subject = unidecode(subject).lower()
    if subject in subnorm_map:
        return subnorm_map[subject]
    else:
        return "other"


def check_strict_modal(statement_row):
    strict_modal = False
    if statement_row['md']:
        strict_modal = statement_row['modal'] in ['dever', 'ter_de', 'precisar'] # MUST INCLUDE WILL
        # strict_modal = statement_row['modal'] in ['shall','must','will']
    return strict_modal

def check_neg(statement_row):
    return statement_row['neg'] == 'não'

def compute_statement_auth(args, df, filename):
    ### auth measures below
    vars_to_keep = ["contract_id", "sentence_num", "slem", "subject", 
                    "md", "verb", "passive", "full_sentence", "neg", "modal"]

    df = df[vars_to_keep]
    df["md"] = df["md"].astype('bool')
    df["passive"] = df["passive"].astype('bool')
    df["subject"] = df["subject"].str.lower()
    df["subnorm"] = df["slem"].apply(normalize_subject)
    df["strict_modal"] = df.apply(check_strict_modal, axis=1).astype('bool')
    df["neg"] = df['neg'].apply(lambda x: x == 'não').astype('bool')

    # permissive modals are may and can
    df['permissive_modal'] = (df['md'] & ~df['strict_modal']).astype('bool')

    # obligation verbs 
    df_passive = df['passive']

    # df['obligation_verb'] = ((df_passive & 
    #                   df['verb'].isin(['require', 'expect', 'compel', 'oblige', 'obligate'])) |
    #                   (~df_passive & df['verb'].isin(['agree','promise']))).astype('bool')
    df['obligation_verb'] = ((df_passive & 
                      df['verb'].isin(['exigir', 'esperar', 'coagir', 'obrigar', 'compelir'])) |
                      (~df_passive & df['verb'].isin(['concordar', 'prometer']))).astype('bool')

    # constraint verbs 
    # df['constraint_verb'] = (df_passive & 
    #                   df['verb'].isin(['prohibit', 'forbid', 'ban', 'bar', 'restrict', 'proscribe'])).astype('bool')
    df['constraint_verb'] = (df_passive & 
                      df['verb'].isin(['proibir', 'vedar', 'banir', 'impedir', 'restringir', 'proscribir', 'limitir'])).astype('bool')
        
    # permissiion verbs are be allowed, be permitted, and be authorized
    # df['permission_verb'] =  (df_passive &  
    #                    df['verb'].isin(['allow', 'permit', 'authorize'])).astype('bool')
    df['permission_verb'] = (df_passive & 
                             df['verb'].isin(['allow', 'permit', 'authorize']) 
                             | (df['verb'].isin(['poder']))).astype('bool')
        
      
    df_notpassive = ~df_passive
    df_neg = df['neg']
    df_notneg = ~df_neg

    """
    # this was the original implementation
    df['entitlement_verb'] =  ((df_notpassive &  
                         df['verb'].isin(['have', 'receive','retain', ])) | (df_passive & df['verb'].isin(['entitle']))).astype('bool')
    """    
    # this is the new entitlement verblist    
    # df['entitlement_verb'] =  ((df_notpassive &   
    #                      df['verb'].isin(['receive', "gain", "earn"])) | (df_passive & df['verb'].isin(['entitle', 'give', "offer", "reimburse", "pay", "grant", "provide", "compensate", "guarantee", "hire", "train", "supply", "protect", "cover", "inform", "notify", "grant_off", "select", "allow_off", "award", "give_off", "protect", "pay_out", "allow_up"]))).astype('bool')
    df['entitlement_verb'] =  ((df_notpassive &   
                            df['verb'].isin(['receber', 'ganhar', 'obter'])) | (df_passive & df['verb'].isin(['conceder', 'dar', 'oferecer', 'reembolsar', 'pagar', 'outorgar', 'fornecer', 'compensar', 'garantir', 'contratar', 'treinar', 'suprir', 'proteger', 'cobrir', 'informar', 'notificar', 'conceder_permissão', 'selecionar', 'permitir', 'conceder_premiação', 'entregar', 'proteger', 'pagar', 'permitir']))).astype('bool')

    # df['promise_verb'] = (df_notpassive & 
    #               df['verb'].isin(['commit','recognize',
    #                           'consent','assent','affirm','assure',
    #                           'guarantee','insure','ensure','stipulate',
    #                           'undertake','pledge'])).astype('bool')
    df['promise_verb'] = (df_notpassive & 
                  df['verb'].isin(['comprometer', 'reconhecer',
                              'consentir', 'concordar', 'afirmar', 'assegurar',
                              'garantir', 'segurar', 'assegurar', 'estipular',
                              'assumir', 'prometer'])).astype('bool')

       
    df['special_verb'] = (df['obligation_verb'] | df['constraint_verb'] | df['permission_verb'] | df['entitlement_verb'] | df['promise_verb']).astype('bool')
      
    df['active_verb'] = (df_notpassive & ~df['special_verb']).astype('bool')
        
    df['obligation'] = ((df_notneg & df['strict_modal'] & df['active_verb']) |     #positive, strict modal, action verb
                    (df_notneg & df['strict_modal'] & df['obligation_verb']) | #positive, strict modal, obligation verb
                    (df_notneg & ~df['md'] & df['obligation_verb'])).astype('bool')           #positive, non-modal, obligation verb
        
    df['constraint'] = ((df_neg & df['md'] & df['active_verb']) | # negative, any modal, any verb except obligation verb
                    (df_notneg & df['strict_modal'] & df['constraint_verb']) | # positive, strict modal, constraint verb
                     (df_neg & df_passive & (df['entitlement_verb'] | df['permission_verb'] ))).astype('bool') # Negated passive verbs should not be constraints, unless it is a permission/entitlement verb
                
    df['permission'] = ((df_notneg & ( (df['permissive_modal'] & df['active_verb']) | 
                  df['permission_verb'])) | 
                  (df['neg'] & df['constraint_verb'])).astype('bool')
        
                          
    """          
    # this was the original implementation                
    df['entitlement'] = ((df_notneg & df['entitlement_verb']) |
                  (df_notneg & df['strict_modal'] & df['passive'] & ~df['obligation_verb'] &~df['permission_verb'] & ~df["constraint_verb"]) |
                  (df_neg & df['obligation_verb'])).astype('bool')  
    """
    

    df['entitlement'] = ((df_notneg & df['entitlement_verb']) |
                  # (df_notneg & df['strict_modal'] & df['passive'] & ~df['obligation_verb'] &~df['permission_verb'] & ~df["constraint_verb"]) | # we drop these because they produce e.g., worker must be laid off, which clearly is not an entitlement... We double-checked and included all the verbs (from the top-100 most frequent verbs) which are produced by this and are true entitlements to df["entitlement_verb"]
                  (df_neg & df['obligation_verb'])).astype('bool')  
    
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
