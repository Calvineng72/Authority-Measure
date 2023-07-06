import argparse
import os
import pandas as pd
from tqdm import tqdm

# command to run the file in the terminal
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

# agent dictionaries
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
          'acidentados', 'acidentadas', 'substituto', 'substituta', 'substitutos', 'substitutas', 'pai', 'pais', 'mãe', 'mães',
          'beneficiário', 'beneficiários']
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

# hash map from possible agents to category
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
    Normalizes a subject by mapping it to agent category.

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
    strict_modals = {'dever', 'deverá', 'deverão', 'deve', 'devem', 'ter que', 'ir'}
    return statement_row['md'] and statement_row['mlem'] in strict_modals

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
    df['obligation_verb'] = ((df_passive & df['vlem'].isin({'exigir', 'esperar', 'coagir', 'compelir', 'obrigar', 'obrigado',
                                                            'forçar', 'requerer', 'comprometar', 'comprometer', 'responsabilizar'})) 
                             | (df_notpassive & df['vlem'].isin({'garantir', 'assegurar'}))).astype('bool')

    # constraint verbs 
    df['constraint_verb'] = (df_passive & df['vlem'].isin({'proibir', 'vedar', 'banir', 'impedir', 'impeder', 'restringir', 'proscrever', 
                                                           'limitar', 'impossibilitar', 'negar', 'abster'})).astype('bool')
        
    # permissiion verbs
    df['permission_verb'] = (df_passive & df['vlem'].isin({'permitir', 'autorizar', 'aprovar', 'habilitar'})).astype('bool')

    # entitlement verbs    
    df['entitlement_verb'] =  ((df_notpassive & df['vlem'].isin({'ter', 'receber', 'ganhar', 'obter', 'gozar', 'beneficiar', 'repousar'}))
                               | (df_passive & df['vlem'].isin({'conceder', 'dar', 'outorgar', 'fornecer', 'garantir', 'garantido', 
                                                                'proteger', 'cobrir', 'informar', 'notificar', 'assegurar',
                                                                'facultar', 'proporcionar', 'prestar', 'propiciar', 'providenciar', 
                                                                'fornecir', 'avisar'}))).astype('bool')

    # promise verbs
    df['promise_verb'] = (df_notpassive & df['vlem'].isin({'reconhecer', 'consentir', 'afirmar', 'segurar', 'estipular', 'assumir', 
                                                            'concordar', 'prometer', 'aquiescer'})).astype('bool')

    # negative verbs
    df['negative_verb'] = ((df_notpassive & df['vlem'].isin({'trabalhar', 'sofrer', 'perder'})) 
                           | df_passive & df['vlem'].isin({'despedir', 'despeder', 'dispensar', 'dispensado', 'dispensados'})).astype('bool') 

    # # verbs to be removed and classified as an 'other provision'
    # df['to_remove'] = (df_notpassive & df['vlem'].isin({'fazer', 'fará', 'farão', 'faz', 'fazem', 'estar', 'estará', 'estarão', 
    #                                                     'está', 'estão', 'ser', 'será', 'serão', 'é', 'são' 'ficar', 'ficará', 
    #                                                     'ficarão' 'fica', 'ficam'})).astype('bool') 

    # special verbs
    df['special_verb'] = (df['obligation_verb'] | df['constraint_verb'] | df['permission_verb'] | df['entitlement_verb'] | df['promise_verb']).astype('bool')
      
    # active verbs 
    df['active_verb'] = (df_notpassive & ~df['special_verb']).astype('bool') # & ~df['to_remove']
        
    # obligations
    df['obligation_1'] = (df_notneg & df['strict_modal'] & df['active_verb']).astype('bool') 
    df['obligation_2'] = (df_notneg & ~df['permissive_modal'] & (df['obligation_verb'] | df['promise_verb'])).astype('bool')  
    df['obligation'] = (df['obligation_1'] | df['obligation_2']).astype('bool')

    # constraints
    df['constraint_1'] = (df_neg & df['md'] & (~df['obligation_verb'] & ~df['negative_verb'] & ~df['constraint_verb'])).astype('bool')
    df['constraint_2'] = (df_notneg & df['strict_modal'] & df['constraint_verb']).astype('bool')
    df['constraint_3'] = (df_neg & df['permission_verb']).astype('bool')
    df['constraint'] = (df['constraint_1'] | df['constraint_2'] | df['constraint_3']).astype('bool')

    # permissions
    df['permission_1'] = (df_notneg & df['permissive_modal'] & ~df['special_verb']).astype('bool')
    df['permission_2'] = (df_notneg & df['permission_verb']).astype('bool')
    df['permission_3'] = (df_neg & df['constraint_verb']).astype('bool')
    df['permission'] = (df['permission_1'] | df['permission_2'] | df['permission_3']).astype('bool')

    # entitlements
    df['entitlement_1'] = (df_notneg & df['entitlement_verb']).astype('bool')
    df['entitlement_2'] = (df_notneg & df['strict_modal'] & df['passive'] & (~df['special_verb'] & ~df['negative_verb'])).astype('bool')
    df['entitlement_3'] = (df_neg & (df['obligation_verb'] | df['negative_verb'])).astype('bool')
    df['entitlement'] = (df['entitlement_1'] | df['entitlement_2'] | df['entitlement_3']).astype('bool')

    df['other_provision'] = ~(df['obligation'] | df['constraint'] | df['permission'] | df['entitlement']).astype('bool')
    
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
