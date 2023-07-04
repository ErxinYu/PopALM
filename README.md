# PopALM

## Overview

The code for "PopALM: Popularity-Aligned Language Models for Social Media Trendy Response Prediction."

## Requirements

``` 
conda create -n dis python=3.7 
conda activate dis
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -e .
pip install gensim jieba matplotlib overrides pyhocon allennlp accelerate tensorboard pandas datasets
```

## Description

Social media platforms are daily exhibiting millions of events. To better track public responses to these events, we study trendy response prediction to automatically generate top-liked user replies to social media events. While previous work focus on generating responses without factoring in popularity, we propose Popularity-Aligned Language Models (PopALM) to distinguish responses liked by a larger audience through reinforcement learning. Recognizing the noisy labels from user ``likes'', we tailor-make curriculum learning in proximal policy optimization (PPO) to help models capture the essential samples for easy-to-hard training. In experiments, we build a large-scale Weibo dataset for trendy response prediction, and its results show that PopALM can help boost the performance of advanced language models.

![avatar](framework.png)
