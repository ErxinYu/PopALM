#!/usr/bin/env python
# -*- coding: utf-8 -*-

 
import sys
import math
from collections import Counter

def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    统计n-gram频率并用dict存储
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None: 
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict

def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    print("pred_dict",pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1 
        #if freq == 1:
        #    ngram_distinct_count += freq
    return ngram_distinct_count / ngram_total


def calc_distinct(pair_list):
    """
    calc_distinct
    """
    distinct1 = calc_distinct_ngram(pair_list, 1)
    distinct2 = calc_distinct_ngram(pair_list, 2)
    return [distinct1, distinct2]




# eval_file = "your generated and golden response"

# for line in open(eval_file):
#     tk = line.strip().split("\t")
#     if len(tk) < 2:
#         continue
#     pred_tokens = tk[0].strip().split(" ")
#     gold_tokens = tk[1].strip().split(" ")
#     sents.append([pred_tokens, gold_tokens])
# calc f1
sents = [["你好，你是谁", "你好，我来自中国"]]
# calc distinct
distinct1, distinct2 = calc_distinct(sents)

print("DISTINCT1: %.3f%%\n" % distinct1)
print("DISTINCT2: %.3f%%\n" % distinct2)

