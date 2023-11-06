# PopALM

## Overview

The code for "PopALM: Popularity-Aligned Language Models for Social Media Trendy Response Prediction."

## Description

Social media platforms are daily exhibiting millions of events. To better track public responses to these events, we study trendy response prediction to automatically generate top-liked user replies to social media events. While previous work focus on generating responses without factoring in popularity, we propose Popularity-Aligned Language Models (PopALM) to distinguish responses liked by a larger audience through reinforcement learning. Recognizing the noisy labels from user ``likes'', we tailor-make curriculum learning in proximal policy optimization (PPO) to help models capture the essential samples for easy-to-hard training. In experiments, we build a large-scale Weibo dataset for trendy response prediction, and its results show that PopALM can help boost the performance of advanced language models.

<img src=framework.png width="200" height="150">

## Requirements

``` 
conda env create -f PopALM.yaml
```

## Training PopALM
### Step 1: Supervised Fine-tuning

``` 
bash train_chatglm_ptuning.sh
```

### Step 2: Reward Model Training

``` 
bash train_reward.sh
```

### Step 3: CL-PPO Training

``` 
bash train_chatglm_clppo_ptuning.sh
```

## Testing PoPALM

``` 
bash test_chatglm_ptuning.sh
```
