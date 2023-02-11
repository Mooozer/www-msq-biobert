# !/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import collections
from transformers import BertForQuestionAnswering,BertTokenizer,BertModel,AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator
from transformers import, TrainingArguments, Trainer
from datasets import Dataset, load_dataset, load_metric
from tabulate import tabulate
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_biobert_large = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')




def prepare_validation_features(examples, tokenizer):
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





max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer_biobert_large.padding_side == "right"
new_Wiki_STQuAAD = generate_new_Wiki_QUAD(tokenizer_biobert_large)


print('---------- predict ST-QuAAD raw scores with BioBERT large ----------',)
biobert_large_finetuned_model = BertForQuestionAnswering.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')
data_collator = default_data_collator

new_Wiki_STQuAAD_features =  new_Wiki_STQuAAD.map(
    lambda x: prepare_validation_features(x, tokenizer = tokenizer_biobert_large),
    batched=True,
    remove_columns=new_Wiki_STQuAAD.column_names) 

args_biobert_large = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch",
    learning_rate= 3e-5,  
    per_device_train_batch_size = 16,  
    per_device_eval_batch_size= 16,   
    num_train_epochs= 2, 
    weight_decay=0.01,
)
trainer_BioBERTlarge = Trainer(
    biobert_large_finetuned_model,
    args_biobert_large,
    data_collator=data_collator,
    tokenizer=tokenizer_biobert_large,
)

raw_pred_BioBERTlarge_STQuAAD = trainer_BioBERTlarge.predict(new_Wiki_STQuAAD_features) ## size (2, #xx, 384) = (start/end, #example features, length)
np.save('./Wiki_ST-QuAAD/raw_pred_BioBERTlarge_STQuAAD.npy',raw_pred_BioBERTlarge_STQuAAD.predictions) 
    
    




# Results
table = [['Model','F1','Exact_match','Word_match','String_match','Levenshtein similarity']]


model_tokenizer = {'BERTlarge': tokenizer_bert, 'BioBERTlarge': tokenizer_biobert_large}
for model, tokenizer in model_tokenizer.items():
    print('----------------------',model,'----------------------')
    new_Wiki_STQuAAD = generate_new_Wiki_QUAD(tokenizer)
    max_length = 384 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
    pad_on_right = tokenizer.padding_side == "right"

    raw_pred_xmodel_newSTQuAAD = np.load('./Wiki_ST-QuAAD/raw_pred_'+model+'_STQuAAD.npy', allow_pickle ='TRUE')
#   #save: 
#     cID_with_fID_newSTQuAAD = mapping_cID_with_fID(new_Wiki_STQuAAD, tokenizer = tokenizer)
#     np.save('./Wiki_ST-QuAAD/cID_with_fID_'+model+'_STQuAAD.npy', cID_with_fID_newSTQuAAD) 
#   #load: 
    cID_with_fID_newSTQuAAD = np.load('./Wiki_ST-QuAAD/cID_with_fID_'+model+'_STQuAAD.npy', allow_pickle=True)

    cID_with_fID_newSTQuAAD_dic = dict(cID_with_fID_newSTQuAAD.item())
    cID_with_fID_newSTQuAAD_dic2 = {v[0]: v for k, v in cID_with_fID_newSTQuAAD_dic.items()}

    #Create dictionary to store question index, number of augumentation, number of splits
    indexNum_dic = Counter([i.split('_')[0] for i in new_Wiki_STQuAAD['new_id']]) 
    Q_index_aug_split_dic = {} #record ori question index, number of aug q, and number of split
    allFeature_sofar = 0  #number of features so far 
    for q_i, aug in indexNum_dic.items(): #for each original question, its aug number
        split = len(cID_with_fID_newSTQuAAD_dic2[allFeature_sofar]) #how many splits for this q(note all aug qs have same length)
        Q_index_aug_split_dic[q_i] = {'aug':aug, 'split':split}
        allFeature_sofar += aug * split

    ori_scores, MSQ_scores ,matrix_for_plot= multi_score_2_single_score(new_Wiki_STQuAAD, raw_pred_xmodel_newSTQuAAD, Q_index_aug_split_dic, 
                                                                        rank = 1,augQ_fs0 = 0.01, LS_threshold = 0.85, structure = 'full')

    indexNum_dic = Counter([i.split('_')[0] for i in new_Wiki_STQuAAD['new_id']]) 
    rangeIndex = np.cumsum([0]+list(indexNum_dic.values()))[:-1]

    ori_result, ori_p, ori_r = calculate_ex_f1(ori_scores[0], ori_scores[1], Dataset.from_dict(new_Wiki_STQuAAD[rangeIndex]), tokenizer=tokenizer)
    MSQ_result, msq_p, msq_r = calculate_ex_f1(MSQ_scores[0], MSQ_scores[1], Dataset.from_dict(new_Wiki_STQuAAD[rangeIndex]), tokenizer=tokenizer)

    ### majority vote
    majority_vote_result = {'exact_match':[], 'f1':[]}
    majority_vote_text = []
    cumsum_num_of_aug = 0 
    total_split = 0
    for k in range(len(Q_index_aug_split_dic)):
        print(f"data ST-QuAAD, {k}/{len(Q_index_aug_split_dic)}")
        num_of_aug = matrix_for_plot[0][0][total_split].shape[0]
        split = Q_index_aug_split_dic[str(k)]['split']
        start = np.vstack([matrix_for_plot[0][0][total_split+o] for o in range(split)])
        end = np.vstack([matrix_for_plot[1][0][total_split+o] for o in range(split)])  
        result, p, r = calculate_ex_f1(start, end, 
                                       Dataset.from_dict(new_Wiki_STQuAAD[cumsum_num_of_aug:cumsum_num_of_aug+num_of_aug]), 
                                       tokenizer)  
        aug_texts = [j['prediction_text'] for j in p] #find all texts with aug questions
        mode_text = collections.Counter(aug_texts).most_common()[0][0] #majority vote text 
        single_index = aug_texts.index(mode_text) #find representation index of (one of the majority vote text )
        
        result1, p1, r1 = calculate_ex_f1(start[[single_index+o*num_of_aug for o in range(split)]], 
                                          end[[single_index+o*num_of_aug for o in range(split)]], 
                                          Dataset.from_dict(new_Wiki_STQuAAD[cumsum_num_of_aug+single_index:cumsum_num_of_aug+single_index+1]), 
                                          tokenizer)
        
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

    table.append([model,  round(ori_result['f1'],2), 
                          round(ori_result['exact_match'],2),              
                          round(word_matching_ori,2),
                          round(str_matching_ori,2),
                          round(LS_ori,2)])
    table.append([model+"-vote",  round(np.mean(majority_vote_result['f1']),2), 
                          round(np.mean(ori_result['exact_match']),2),              
                          round(word_matching_vote,2),
                          round(str_matching_vote,2),
                          round(LS_vote,2)])
    table.append(['MSQ'+model, round(MSQ_result['f1'],2),
                                round(MSQ_result['exact_match'],2), 
                                round(word_matching_msq,2),
                                round(str_matching_msq,2),
                                round(LS_msq,2)])

print(tabulate(table))
