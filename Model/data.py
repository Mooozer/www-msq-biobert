# !/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from datasets import Dataset

import torch
import collections
from transformers import BertForQuestionAnswering,BertTokenizer,BertModel,AutoTokenizer # AdamW, BertConfig
from datasets import Dataset, load_dataset, load_metric
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_biobert_large = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')

import urllib.request
from bs4 import BeautifulSoup
import re
import pandas as pd


def get_treatment_from_wiki(sym):
    '''
    sym: input, symptoms string 
    output: treatment text
    '''
    url = 'https://en.wikipedia.org/wiki/' +  sym
    response = urllib.request.urlopen(url)
    html_doc = response.read().decode(encoding='UTF-8')
    parsed = BeautifulSoup(html_doc, "html.parser")
    
    soup = parsed.find("span",{'class':'mw-headline', "id":"Treatment"})#First look at Treatment
    if not soup:
        soup = parsed.find("span",{'class':'mw-headline', "id":"Management"})#If there is no Treatment, use Management
        if not soup:
            soup = parsed.find("span",{'class':'mw-headline', "id":"Treatments"})
            if not soup: 
                treatment = ["NA"]
                return treatment
        
    last_parent = list(soup.parents)[0]
    close_siblings = list(last_parent.next_siblings)
    
    treatment = []
    for i in range(len(close_siblings)):
        if close_siblings[i].name == 'h2':  #do not include next chapter
            break
        if close_siblings[i].name == 'p':   #main body of Treatment/Management
            ori_text = close_siblings[i].text
            ori_text = re.sub(r"\xa0", "", ori_text)  #remove "\xa0"
            ori_text = re.sub(r"\[\d*\]", "", ori_text) #remove cite
            ori_text = re.sub(r"\n", "", ori_text) #remove "\n"
            treatment.append(ori_text)
        else:
            continue
            
    return treatment


Sym_Tre_dic = {}
Symptoms = ['Fever', 'Cough', 'Shortness_of_breath',  'Myalgia', 'Headache', 'Anosmia', 
            'Sore_throat', 'Nasal_congestion', 'Rhinorrhea', 'Nausea', 'Vomiting', 'Diarrhea',
           'Abdominal_pain','Blood_in_stool','Chest_pain','Constipation','Dysphagia',
           'Palpitations','Knee_pain','Low_back_pain','Neck_pain','Paresthesia','Rash','Hemoptysis',
            'Pneumonia','Delayed_onset_muscle_soreness','Back_pain','Xerostomia','Dry_eye_syndrome',
           'Insomnia','Sleep_deprivation','Cyanosis','Somnolence','Heartburn','Tremor','Chronic_pain'] 
for s in Symptoms:
    Sym_Tre_dic[s] = get_treatment_from_wiki(s)





# ST_QuAAD dataset
def padded_question(question_list, tokenizer):
    '''
    input: question list, tokenizer
           tokenizer
    output: padded question list
    '''
    QList = [question_list[i].strip() for i in range(len(question_list))] #lower case, remove last space    
    QList_token_len = [len(tokenizer(QList)['input_ids'][i]) for i in range(len(QList))]
    QList_token_len_max = max(QList_token_len)
    QList_padded = [["[PAD] " * (QList_token_len_max - QList_token_len[i]) + QList[i]] for i in range(len(QList))]
    return QList_padded




def generate_ST_QuAAD(tokenizer):
    '''
    Pad questions using tokenizer, then generate the new Wiki_QUAD
    '''
    Symptoms_words = ['Fever', 'Cough', 'Shortness of breath',  'Myalgia', 'Headache', 'Anosmia', 
                  'Sore throat', 'Nasal congestion', 'Rhinorrhea', 'Nausea', 'Vomiting', 'Diarrhea',
                  'Abdominal pain','Blood in stool','Chest pain','Constipation','Dysphagia',
                  'Palpitations','Knee pain','Low back pain','Neck pain','Paresthesia', 'Rash','Hemoptysis',
                  'Pneumonia','Delayed onset muscle soreness','Back pain','Xerostomia','Dry eye syndrome',
                  'Insomnia','Sleep deprivation','Cyanosis','Somnolence','Heartburn','Tremor','Chronic pain'] 
    Symptoms = ['Fever', 'Cough', 'Shortness_of_breath',  'Myalgia', 'Headache', 'Anosmia', 
            'Sore_throat', 'Nasal_congestion', 'Rhinorrhea', 'Nausea', 'Vomiting', 'Diarrhea',
           'Abdominal_pain','Blood_in_stool','Chest_pain','Constipation','Dysphagia',
           'Palpitations','Knee_pain','Low_back_pain','Neck_pain','Paresthesia','Rash','Hemoptysis',
            'Pneumonia','Delayed_onset_muscle_soreness','Back_pain','Xerostomia','Dry_eye_syndrome',
           'Insomnia','Sleep_deprivation','Cyanosis','Somnolence','Heartburn','Tremor','Chronic_pain']
    
    vaild_qs = [q.lower() for q in wiki_aug_ques_list['Ques_argument'] if 'aa' in q.lower()]
    aug_qs_temp = list(dict.fromkeys(vaild_qs)) #keep unique and keep order 
    aug_qs_temp = padded_question(aug_qs_temp, tokenizer)  #pad questions
    aug_qs_temp = sum(aug_qs_temp, []) #convert to a single list   
    
    Wiki_data_dic = {'answers':[], 'context':[], "id":[], "question":[], "title":[], "new_id":[]}
    passage_id = 0 
    
    for s in range(len(Symptoms)): 
        sym = Symptoms[s]
        tem_content =  wiki_contents_file[sym].dropna().tolist()
        tem_answers1 =  wiki_answers_file1[sym].dropna().tolist()
        tem_answers2 =  wiki_answers_file2[sym].dropna().tolist()
        tem_answers3 =  wiki_answers_file3[sym].dropna().tolist()
        for c in range(len(tem_content)):
            content_c = tem_content[c]
            
            for q in range(len(aug_qs_temp)):
                
                Wiki_data_dic['context'] += [str(tem_content[c])]
                
                Wiki_data_dic['question'] += [aug_qs_temp[q].replace("aa", Symptoms_words[s].lower())]
                
                ans_dic = {'answer_start': sum([[i for i in range(len(content_c)) if content_c.startswith(a, i)] 
                                                 for a in [tem_answers1[c],tem_answers2[c],tem_answers3[c]] if a],[]),
                                   'text': sum([[a for i in range(len(content_c)) if content_c.startswith(a, i)] 
                                                 for a in [tem_answers1[c],tem_answers2[c],tem_answers3[c]] if a],[])}
                Wiki_data_dic['answers'] += [ans_dic]

                Wiki_data_dic['id'] += [sym+str(c)+str(q)]
                Wiki_data_dic['title'] += [sym+str(c)+str(q)]
                Wiki_data_dic['new_id'] += [str(passage_id)+"_"+str(q)]
            passage_id+=1 
    new_Wiki_QUAD = Dataset.from_dict(Wiki_data_dic)
    return new_Wiki_QUAD 





#BioASQ Datasets

def padded_question(question_list, tokenizer = tokenizer_biobert_large):
    '''
    input: question list, tokenizer
           tokenizer
    output: padded question list
    '''
    QList = [question_list[i].strip() for i in range(len(question_list))] #lower case, remove last space    
    QList_token_len = [len(tokenizer(QList)['input_ids'][i]) for i in range(len(QList))]
    QList_token_len_max = max(QList_token_len)
    QList_padded = [["[PAD] " * (QList_token_len_max - QList_token_len[i]) + QList[i]] for i in range(len(QList))]
    return QList_padded

lang_tgt_list= ['af','sq','ar','hy','az','eu','be','bg','ca','zh','hr','cs','da','nl','et','tl','fi','fr','gl',
                'ka','de','el','ht','iw','hi','hu','is','id','ga','it','ja','ko','lv','lt','mk','ms','mt','no',
                'fa','pl','pt','ro','ru','sr','sk','sl','es','sw','sv','th','tr','uk','ur','vi','cy','yi']
language_list = ['Afrikaans','Albanian','Arabic','Armenian','Azerbaijani','Basque','Belarusian','Bulgarian',
                 'Catalan','Chinese','Croatian','Czech','Danish','Dutch','Estonian','Filipino','Finnish','French',
                 'Galician','Georgian','German','Greek','Haitian Creole','Hebrew','Hindi','Hungarian','Icelandic',
                 'Indonesian','Irish','Italian','Japanese','Korean','Latvian','Lithuanian','Macedonian','Malay',
                 'Maltese','Norwegian','Persian','Polish','Portuguese','Romanian','Russian','Serbian','Slovak',
                 'Slovenian','Spanish','Swahili','Swedish','Thai','Turkish','Ukrainian','Urdu','Vietnamese',
                 'Welsh','Yiddish']
language_dic = {lang_tgt_list[i]: language_list[i] for i in range(len(lang_tgt_list))}




import json
def create_BioASQ_SQuAD(x):
    '''
    x: string: 6~9, corresponding to 3B ~ 9B tasks
    '''
    BioASQ_xB = {'answers':[], 'context':[], 'id':[], 'question':[], 'title':[], 'new_id' :[]} 
    new_id = 0
    for f_num in range(1,6): #for each of the five json files:
        with open('./BioASQ/Task'+x+'BGoldenEnriched/'+x+'B'+str(f_num)+'_golden.json', 'r') as f:
            data = json.load(f)
            data = data['questions'] 
            for d in range(len(data)):
                if data[d]['type']!='factoid':
                    continue
                context_list = [i['text'].lower().strip() for i in data[d]['snippets']] #extract all the context
                #select context which include at least one exact_answer:                 
                context_list = [c for c in context_list 
                                if any([ea.lower().strip() in c for ea in data[d]['exact_answer'][0]])] 
                        
                context_list = list(set(context_list)) #remove repeated strings
                context_list = sorted(context_list, key=len)[::-1] #sort context by length decreasing

                for c in context_list:
                    # find all locations of exact_answer in context
                    ans_text = {'answer_start': sum( [[i for i in range(len(c)) if c.startswith(a, i)] for a in data[d]['exact_answer'][0] if a],[]),
                                        'text': sum( [[a for i in range(len(c)) if c.startswith(a, i)] for a in data[d]['exact_answer'][0] if a],[]) }
                    if ans_text == {'answer_start': [], 'text': []}:
                        continue     
                    BioASQ_xB['answers'] += [ans_text]
                    BioASQ_xB['context'] += [c]
                    BioASQ_xB['id'] += [data[d]['id']+"_"+str(new_id)]
                    BioASQ_xB['question'] += [data[d]['body']]
                    BioASQ_xB['title'] += [data[d]['id']] #using id as title 
                    BioASQ_xB['new_id'] += [str(new_id)] 
                    new_id += 1
    np.save('./BioASQ/Task'+x+'BGoldenEnriched/BioASQ_'+x+'B_SQuAD.npy',BioASQ_xB) 
    return Dataset.from_dict(BioASQ_xB) 



for b in ['6','7','8','9']:
    new_BioASQ_xB_SQuAD = {'answers':[], 'context':[], 'id':[], 'question':[], 'title':[], 'new_id' :[]} 
    combined_BioASQ_xB_augQ = pd.read_excel("./BioASQ/Task"+b+"BGoldenEnriched/combined_BioASQ_"+b+"B_augQ.xlsx", index_col=0) 
    BioASQ_xB_SQuAD = Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/BioASQ_'+b+'B_SQuAD.npy',allow_pickle='TRUE').item()) 

    for q_id in range(combined_BioASQ_xB_augQ.shape[0]): 
        ql = combined_BioASQ_xB_augQ.iloc[q_id].tolist()
        ql = [ql[i].lower().strip() for i in range(len(ql))] #lower case, remove last space   
        ql = list(dict.fromkeys(ql))  # unique and keep unique 
        padded_ql = padded_question(ql, tokenizer_biobert_large)
        padded_ql = sum(padded_ql, []) #convert to a single list 
        for j in range(len(padded_ql)):
            if BioASQ_xB_SQuAD[q_id]['answers'] == {'answer_start': [], 'text': []}:
                break 
            # find all locations of exact_answer in context:
            ans_modified = {'answer_start': BioASQ_xB_SQuAD[q_id]['answers']['answer_start'],
                            'text': BioASQ_xB_SQuAD[q_id]['answers']['text']}
            new_BioASQ_xB_SQuAD['answers'].append(ans_modified)
            new_BioASQ_xB_SQuAD['context'].append(BioASQ_xB_SQuAD[q_id]['context'])
            new_BioASQ_xB_SQuAD['id'].append(BioASQ_xB_SQuAD[q_id]['id'])
            new_BioASQ_xB_SQuAD['question'].append(padded_ql[j])
            new_BioASQ_xB_SQuAD['title'].append(BioASQ_xB_SQuAD[q_id]['title']) 
            new_BioASQ_xB_SQuAD['new_id'].append(str(q_id)+'_'+str(j))
    np.save('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',new_BioASQ_xB_SQuAD) 
