import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel

def distinct_n(preds,n):
    # print("preds",preds)
    n_gram_dict = {}
    for pred in preds:
        pred_ = list(jieba.cut(pred))
        # print("pred_",pred_)
        for n_gram in pred_:
            if len(n_gram) == n:
                if n_gram not in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                else:
                    n_gram_dict[n_gram] += 1
    # print("n_gram_dict",n_gram_dict)
    unigram = 0
    allgram = 0
    for n_gram, num in n_gram_dict.items():
        allgram += num
        if num == 1:
            unigram +=1 
    if allgram != 0:
        distinct  = unigram/allgram
    else:
        distinct = 0
    # print("disctic",n,":",distinct) 
    return distinct

def comp(pred, label):
    pred_ = list(jieba.cut(pred))
    label_ = list(jieba.cut(label))
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(pred_) , ' '.join(label_))
    result = scores[0]
    bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
    return result, bleu_score


score_dict_ori = {
    "rouge-1": [],
    "rouge-2": [],
    "rouge-l": [],
    "bleu-4": []
}
score_dict_FT = {
    "rouge-1": [],
    "rouge-2": [],
    "rouge-l": [],
    "bleu-4": []
}
score_dict_PPO = {
    "rouge-1": [],
    "rouge-2": [],
    "rouge-l": [],
    "bleu-4": []
}
with open("/home/xxx/data/ppo_weibo/weiboGen/test.json") as f_label, \
   open("/home/xxx/output/ptuningGen/Top3/generated_predictions.txt") as f_pred_ori, \
   open("/home/xxx/output/PPOGen_epoch2_gen1_return1/Top3/generated_predictions.txt") as f_pred_FT, \
   open("/home/xxx/output/PPOGen_ours_epoch2_gen5_return1/PPOTop3/generated_predictions.txt") as f_pred_PPO, \
   open("/home/xxx/output/compare_gen_label_top3.txt","w+") as f_compare:
    label_lines = f_label.readlines()
    predOri_lines = f_pred_ori.readlines()
    predFT_lines = f_pred_FT.readlines()
    predPPO_lines = f_pred_PPO.readlines()
    print(len(label_lines))
    print(len(predOri_lines))
    print(len(predFT_lines))
    print(len(predPPO_lines))
    posts_length = 0
    labels_length = 0
    labels_num = 0


    all_predPPO = []
    all_predFT = []
    all_predOri = []
    m_Distict1_predPPO_score = 0
    m_Distict1_predFT_score = 0
    m_Distict1_predOri_score = 0
    m_Distict2_predPPO_score = 0
    m_Distict2_predFT_score = 0
    m_Distict2_predOri_score = 0
    for i in range(1000):
        post = json.loads(label_lines[i])["weibo"]
        posts_length += len(post)
        labels = json.loads(label_lines[i])["resp"]
        predOri_1 = json.loads(predOri_lines[3*i])["predict"]
        predOri_2 = json.loads(predOri_lines[3*i+1])["predict"]
        predOri_3 = json.loads(predOri_lines[3*i+2])["predict"]
        predFT_1 = json.loads(predFT_lines[3*i])["predict"]
        predFT_2 = json.loads(predFT_lines[3*i+1])["predict"]
        predFT_3 = json.loads(predFT_lines[3*i+2])["predict"]
        predPPO_1 = json.loads(predPPO_lines[3*i])["predict"]
        predPPO_2 = json.loads(predPPO_lines[3*i+1])["predict"]
        predPPO_3 = json.loads(predPPO_lines[3*i+2])["predict"]

        all_predOri.append(predOri_1)
        all_predOri.append(predOri_2)
        all_predOri.append(predOri_3)
        m_Distict_predOri = [predOri_1,predOri_2,predOri_3]
        m_Distict1_predOri_score += distinct_n(m_Distict_predOri,1)
        m_Distict2_predOri_score += distinct_n(m_Distict_predOri,2)

        all_predFT.append(predFT_1)
        all_predFT.append(predFT_2)
        all_predFT.append(predFT_3)
        m_Distict_predFT = [predFT_1,predFT_2,predFT_3]
        m_Distict1_predFT_score += distinct_n(m_Distict_predFT,1)
        m_Distict2_predFT_score += distinct_n(m_Distict_predFT,2)

        all_predPPO.append(predPPO_1)
        all_predPPO.append(predPPO_2)
        all_predPPO.append(predPPO_3)
        m_Distict_predPPO = [predPPO_1,predPPO_2,predPPO_3]
        m_Distict1_predPPO_score += distinct_n(m_Distict_predPPO,1)
        m_Distict2_predPPO_score += distinct_n(m_Distict_predPPO,2)

        best_r1_f_ori = -1
        best_result_ori = {}
        best_bleu_score_ori = -1

        best_r1_f_FT = -1
        best_result_FT = {}
        best_bleu_score_FT = -1

        best_r1_f_PPO = -1
        best_result_PPO = {}
        best_bleu_score_PPO = -1
        for predOri, predFT, predPPO in zip(m_Distict_predOri,m_Distict_predFT,m_Distict_predPPO):
            for label in labels:
                if label == "":
                    continue
                labels_num += 1
                labels_length += len(label)
                result, bleu_score = comp(predOri, label) #predOri, pred
                r1_f = result["rouge-1"]["f"]
                if r1_f > best_r1_f_ori:
                    best_r1_f_ori = r1_f
                    best_result_ori = result
                    best_bleu_score_ori = bleu_score
                result, bleu_score = comp(predFT, label) 
                r1_f = result["rouge-1"]["f"]
                if r1_f > best_r1_f_FT:
                    best_r1_f_FT = r1_f
                    best_result_FT = result
                    best_bleu_score_FT = bleu_score
                result,bleu_score = comp(predPPO, label) 
                r1_f = result["rouge-1"]["f"]
                if r1_f > best_r1_f_PPO:
                    best_r1_f_PPO = r1_f
                    best_result_PPO = result
                    best_bleu_score_PPO = bleu_score
            for k, v in best_result_ori.items():
                score_dict_ori[k].append(round(v["f"] * 100, 4))  
                score_dict_ori["bleu-4"].append(round(best_bleu_score_ori * 100, 4))
            for k, v in best_result_FT.items():
                score_dict_FT[k].append(round(v["f"] * 100, 4))  
                score_dict_FT["bleu-4"].append(round(best_bleu_score_FT * 100, 4))
            for k, v in best_result_PPO.items():
                score_dict_PPO[k].append(round(v["f"] * 100, 4))  
                score_dict_PPO["bleu-4"].append(round(best_bleu_score_PPO * 100, 4))
        print("weibo: ", post, file=f_compare)
        for label in labels:
            print("label: ", label, file=f_compare)
        print("predOri_1:",predOri,file=f_compare)
        print("predFT_1: ",predFT,file=f_compare)
        print("predFT_2:", json.loads(predFT_lines[3*i+1])["predict"],file=f_compare)
        print("predFT_3:", json.loads(predFT_lines[3*i+2])["predict"],file=f_compare)
        print("predPPO_1:",predPPO_1,file=f_compare)
        print("predPPO_2:",json.loads(predPPO_lines[3*i+1])["predict"],file=f_compare)
        print("predPPO_3:",json.loads(predPPO_lines[3*i+2])["predict"],file=f_compare)
        print("--------------",file=f_compare)
    m_Distict1_predOri_score = m_Distict1_predOri_score/len(label_lines)
    m_Distict2_predOri_score = m_Distict2_predOri_score/len(label_lines)
    m_Distict1_predFT_score = m_Distict1_predFT_score/len(label_lines)
    m_Distict2_predFT_score = m_Distict2_predFT_score/len(label_lines)
    m_Distict1_predPPO_score = m_Distict1_predPPO_score/len(label_lines)
    m_Distict2_predPPO_score = m_Distict2_predPPO_score/len(label_lines)

    disctic1_PPO = distinct_n(all_predPPO,1)
    disctic1_FT = distinct_n(all_predFT,1)
    disctic1_Ori = distinct_n(all_predOri,1)
    disctic2_PPO = distinct_n(all_predPPO,2)
    disctic2_FT = distinct_n(all_predFT,2)
    disctic2_Ori = distinct_n(all_predOri,2)

    print("m_Distict1_predOri_score",m_Distict1_predOri_score)
    print("m_Distict1_predFT_score",m_Distict1_predFT_score)
    print("m_Distict1_predPPO_score",m_Distict1_predPPO_score,"\n")

    print("m_Distict2_predOri_score",m_Distict2_predOri_score)
    print("m_Distict2_predFT_score",m_Distict2_predFT_score)
    print("m_Distict2_predPPO_score",m_Distict2_predPPO_score,"\n")
    
    print("disctic1_Ori",disctic1_Ori)
    print("disctic1_FT",disctic1_FT)
    print("disctic1_PPO",disctic1_PPO,"\n")

    print("disctic2_Ori",disctic2_Ori)
    print("disctic2_FT",disctic2_FT)
    print("disctic2_PPO",disctic2_PPO)

    for k, v in score_dict_ori.items():
        score_dict_ori[k] = float(np.mean(v))       
    for k, v in score_dict_FT.items():
        score_dict_FT[k] = float(np.mean(v))
    for k, v in score_dict_PPO.items():
        score_dict_PPO[k] = float(np.mean(v))
    print("average post_length",posts_length/len(predOri_lines))
    print("average label_length",labels_length/labels_num)
    print(score_dict_ori)
    print(score_dict_FT)
    print(score_dict_PPO)
    
    
