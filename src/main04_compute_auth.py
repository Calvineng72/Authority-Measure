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

worker = ['empregado', 'trabalhador', 'motorista', 'funcionário', 'empregada', 'empregados-jornalistas', 
          'equipe', 'empregados-jornalista', 'professor', 'enfermeiro', 'mecânico', 'operador', 'comissário', 'pessoal',
          'docente', 'professor', 'contratado', 'jornalista', 'aprendiz', 'empregados', 'empregadas', 'médico']
firm = ['empregador', 'empresa', 'conselho', 'hospital', 'corporação', 'proprietário', 'superintendente', 'empregadora', 
        'companhia', 'firma'] 
union = ['sindicato', 'associação', 'membro', 'representante']
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
        return "other"

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
        strict_modal = statement_row['modal'] in ['dever', 'ter_que', 'ir']
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
    vars_to_keep = ["contract_id", "slem", "subject", 
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

    df_passive = df['passive']
    df_notpassive = ~df_passive
    df_neg = df['neg']
    df_notneg = ~df_neg

    # obligation verbs 
    df['obligation_verb'] = ((df_passive & df['verb'].isin(['exigir', 'esperar', 'coagir', 'obrigar', 'compelir', 'obrigado'])) 
                             | (df_notpassive & df['verb'].isin(['concordar', 'prometer']))).astype('bool')

    # constraint verbs 
    df['constraint_verb'] = (df_passive & 
                      df['verb'].isin(['proibir', 'vedar', 'banir', 'impedir', 'restringir', 'proscrever', 'limitar'])).astype('bool')
        
    # permissiion verbs are be allowed, be permitted, and be authorized
    df['permission_verb'] = (df_passive & df['verb'].isin(['permitir', 'autorizar', 'aprovar', 'habilitar'])).astype('bool')

    # entitlement verbs    
    df['entitlement_verb'] =  ((df_notpassive & df['verb'].isin(['receber', 'ganhar', 'obter'])) 
                               | (df_passive & df['verb'].isin(['conceder', 'dar', 'oferecer', 'reembolsar', 'pagar', 'outorgar', 
                                                                'fornecer', 'compensar', 'garantir', 'contratar', 'treinar', 'suprir', 
                                                                'proteger', 'cobrir', 'informar', 'notificar', 'selecionar', 
                                                                'entregar', 'proteger', 'pagar', 'premiar',
                                                                'facultar', 'garantido', 'proporcionar', 'prestar', 'propiciar',
                                                                'providenciar', 'fornecir']))).astype('bool')

    # promise verbs
    df['promise_verb'] = (df_notpassive & 
                  df['verb'].isin(['reconhecer',
                              'consentir', 'afirmar',
                              'garantir', 'segurar', 'assegurar', 'estipular',
                              'assumir'])).astype('bool')

    # special verbs
    df['special_verb'] = (df['obligation_verb'] | df['constraint_verb'] | df['permission_verb'] | df['entitlement_verb'] | df['promise_verb']).astype('bool')
      
    # active verbs 
    df['active_verb'] = (df_notpassive & ~df['special_verb']).astype('bool')
        
    # provisions
    df['obligation'] = ((df_notneg & df['strict_modal'] & df['active_verb']) |     
                        (df_notneg & df['strict_modal'] & df['obligation_verb']) | 
                        (df_notneg & ~df['md'] & df['obligation_verb'])).astype('bool')           
    df['constraint'] = ((df_neg & df['md'] & df['active_verb']) |
                        (df_notneg & df['strict_modal'] & df['constraint_verb']) | 
                        (df_neg & df_passive & (df['entitlement_verb'] | df['permission_verb'] ))).astype('bool')
    df['permission'] = ((df_notneg & ((df['permissive_modal'] & df['active_verb']) | 
                                      df['permission_verb'])) | 
                                      (df['neg'] & df['constraint_verb'])).astype('bool')
    df['entitlement'] = ((df_notneg & df['entitlement_verb']) |
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
