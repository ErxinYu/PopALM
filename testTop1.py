import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from nlgeval import NLGEval


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
   open("/home/xxx/output/ptuningGen/Top1/generated_predictions.txt") as f_pred_ori, \
   open("/home/xxx/output/PPOGen_epoch2_gen1_return1/Top1/generated_predictions.txt") as f_pred_FT, \
   open("/home/xxx/output/PPOGen_ours_epoch2_gen5_return1/PPOTop1/generated_predictions.txt") as f_pred_PPO, \
   open("/home/xxx/output/compare_gen_label_top1.txt","w+") as f_compare:
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
    for i in range(len(label_lines)):
        post = json.loads(label_lines[i])["weibo"]
        posts_length += len(post)
        labels = json.loads(label_lines[i])["resp"]
        predOri = json.loads(predOri_lines[i])["predict"]
        if predOri == "":
            predOri = "None"
        predFT = json.loads(predFT_lines[i])["predict"]
        if predFT == "":
            predFT = "None"
        predPPO = json.loads(predPPO_lines[i])["predict"]
        if predPPO == "":
            predPPO = "None"

        best_r1_f_ori = -1
        best_result_ori = {}
        best_bleu_score_ori = -1

        best_r1_f_FT = -1
        best_result_FT = {}
        best_bleu_score_FT = -1

        best_r1_f_PPO = -1
        best_result_PPO = {}
        best_bleu_score_PPO = -1

        best_label = ""
        for label in labels:
            if label == "":
                continue
            labels_num += 1
            labels_length += len(label)
            result, bleu_score = comp(predOri, label) #predOri, pred
            r1_f = result["rouge-1"]["f"]
            if r1_f > best_r1_f_ori:
                best_label = label
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
        print("best_label: ",best_label,file=f_compare)
        print("predOri:",predOri,file=f_compare)
        print("predFT: ",predFT,file=f_compare)
        print("predPPO:",predPPO,file=f_compare)
        print("--------------",file=f_compare)
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
    
    
