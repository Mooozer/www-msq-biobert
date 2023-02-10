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
tokenizer_biobert_large = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')



max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer_biobert_large.padding_side == "right"

def prepare_validation_features(examples, tokenizer = tokenizer_biobert_large):
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["new_id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples




from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer_biobert_large.padding_side == "right"

for b in ['6','7','8','9']:
    print('---------- predict new_BioASQ_',b,'B_SQuAD ----------',)
    new_BioASQ_xB_SQuAD = Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',allow_pickle='TRUE').item())
    new_BioASQ_xB_features = new_BioASQ_xB_SQuAD.map(
        lambda x: prepare_validation_features(x, tokenizer = tokenizer_biobert_large),   
        batched=True,
        remove_columns=new_BioASQ_xB_SQuAD.column_names) 
   
    biobert_large_finetuned_model = BertForQuestionAnswering.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')
    data_collator = default_data_collator
    
    trainer_BioBERTlarge = Trainer(
        biobert_large_finetuned_model,
        data_collator=data_collator,
        tokenizer=tokenizer_biobert_large,
    )

    raw_predictions_BioBERTlarge = trainer_BioBERTlarge.predict(new_BioASQ_xB_features) ## size (2, #xx, 384) = (start/end, #example features, length)
    np.save('./BioASQ/Task'+b+'BGoldenEnriched/raw_pred_BioBERTlarge_newBioASQ_'+b+'B.npy',raw_predictions_BioBERTlarge.predictions) 





#result
#LS=0.85: FS = 0.1 & 0.01, rank = 1 
table = [['Model','Data','F1','Exact_match','Word_match','String_match','Levenshtein similarity']]

for b in ['6','7','8','9']:
    new_BioASQ_xB_SQuAD=Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',
                                                    allow_pickle='TRUE').item())
    raw_pred_BioBERTlarge_newBioASQ_xB=np.load('./BioASQ/Task'+b+'BGoldenEnriched/raw_pred_BioBERTlarge_newBioASQ_'+b+'B.npy'
                                            ,allow_pickle ='TRUE') 
    #save: 
    #cID_with_fID_BioASQ_xB_BioBERT = mapping_cID_with_fID(new_BioASQ_xB_SQuAD, tokenizer = tokenizer_biobert_large)
    #np.save("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy", cID_with_fID_BioASQ_xB_BioBERT) 
    #load: 
    cID_with_fID_BioASQ_xB_BioBERT = np.load("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy",
                                     allow_pickle=True)
    
    cID_with_fID_BioASQ_xB_BioBERT_dic = dict(cID_with_fID_BioASQ_xB_BioBERT.item())
    cID_with_fID_BioASQ_xB_BioBERT_dic2 = {v[0]: v for k, v in cID_with_fID_BioASQ_xB_BioBERT_dic.items()}

    #Create dictionary to store question index, number of augumentation, number of splits
    indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
    Q_index_aug_split_dic = {} #record ori question index, number of aug q, and number of split
    allFeature_sofar = 0  #number of features so far 
    for q_i, aug in indexNum_dic.items(): #for each original question, its aug number
        split = len(cID_with_fID_BioASQ_xB_BioBERT_dic2[allFeature_sofar]) #how many splits for this q(note all aug qs have same length)
        Q_index_aug_split_dic[q_i] = {'aug':aug, 'split':split}
        allFeature_sofar += aug * split
    
    ori_scores, MSQ_scores, matrix_for_plot = multi_score_2_single_score(new_BioASQ_xB_SQuAD, raw_pred_BioBERTlarge_newBioASQ_xB, Q_index_aug_split_dic,
                                                                         rank = 1, augQ_fs0 = 0.01, LS_threshold = 0.85, structure = 'full')
        
    indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
    rangeIndex = np.cumsum([0]+list(indexNum_dic.values()))[:-1]
    print('------------------BioASQ_',b,'B SQuAD------------------')
    ori_result, ori_p, ori_r = calculate_ex_f1(ori_scores[0], ori_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)
    MSQ_result, msq_p, msq_r = calculate_ex_f1(MSQ_scores[0], MSQ_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)
    
    ### majority vote: 
    majority_vote_result = {'exact_match':[], 'f1':[]}
    majority_vote_text = []
    cumsum_num_of_aug = 0 
    total_split = 0
    for k in range(len(Q_index_aug_split_dic)):
        print(f"data {b}, {k}/{len(Q_index_aug_split_dic)}")
        num_of_aug = matrix_for_plot[0][0][total_split].shape[0]
        split = Q_index_aug_split_dic[str(k)]['split']
        start = np.vstack([matrix_for_plot[0][0][total_split+o] for o in range(split)])
        end = np.vstack([matrix_for_plot[1][0][total_split+o] for o in range(split)])  
        result, p, r = calculate_ex_f1(start, end, 
                                       Dataset.from_dict(new_BioASQ_xB_SQuAD[cumsum_num_of_aug:cumsum_num_of_aug+num_of_aug]), 
                                       tokenizer_biobert_large)  
        aug_texts = [j['prediction_text'] for j in p] #find all texts with aug questions
        mode_text = collections.Counter(aug_texts).most_common()[0][0] #majority vote text 
        single_index = aug_texts.index(mode_text) #find representation index of (one of the majority vote text )
        
        result1, p1, r1 = calculate_ex_f1(start[[single_index+o*num_of_aug for o in range(split)]], 
                                          end[[single_index+o*num_of_aug for o in range(split)]], 
                                          Dataset.from_dict(new_BioASQ_xB_SQuAD[cumsum_num_of_aug+single_index:cumsum_num_of_aug+single_index+1]), 
                                          tokenizer_biobert_large)
        
        cumsum_num_of_aug += num_of_aug
        majority_vote_result['exact_match'].append(result1['exact_match']) 
        majority_vote_result['f1'].append(result1['f1'])
        majority_vote_text.append(mode_text)
        
        total_split += split
    ###    
    
    ori_pred_text = [i['prediction_text'] for i in ori_p] 
    msq_pred_text = [i['prediction_text'] for i in msq_p]  
    groundtruth_text = [i['answers']['text'] for i in ori_r] 
    
    word_matching_ori = word_matching_scores(ori_pred_text ,groundtruth_text)
    word_matching_msq = word_matching_scores(msq_pred_text ,groundtruth_text)
    word_matching_vote = word_matching_scores(majority_vote_text ,groundtruth_text)

    str_matching_ori = string_matching_scores(ori_pred_text ,groundtruth_text)
    str_matching_msq = string_matching_scores(msq_pred_text ,groundtruth_text)
    str_matching_vote = string_matching_scores(majority_vote_text ,groundtruth_text)
    
    LS_ori = Levenshtein_similarity(ori_pred_text,groundtruth_text)
    LS_msq = Levenshtein_similarity(msq_pred_text,groundtruth_text)
    LS_vote = Levenshtein_similarity(majority_vote_text,groundtruth_text)
    
    table.append(['BioBERT','BioASQ'+b, str(round(ori_result['f1'],2))+"%",  
                                        str(round(ori_result['exact_match'],2))+"%",
                                        str(round(word_matching_ori,2))+"%",
                                        str(round(str_matching_ori,2))+"%",
                                        str(round(LS_ori,2))+"%"])
    table.append(['BioBERT-vote','BioASQ'+b, str(round(np.mean(majority_vote_result['f1']),2))+"%",  
                                        str(round(np.mean(majority_vote_result['exact_match']),2))+"%",
                                        str(round(word_matching_vote,2))+"%",
                                        str(round(str_matching_vote,2))+"%",
                                        str(round(LS_vote,2))+"%"])
    table.append(['MSQ-BioBERT','BioASQ'+b, str(round(MSQ_result['f1'],2))+"%", 
                                        str(round(MSQ_result['exact_match'],2))+"%",
                                        str(round(word_matching_msq,2))+"%",
                                        str(round(str_matching_msq,2))+"%",
                                        str(round(LS_msq,2))+"%"])
print(tabulate(table))

