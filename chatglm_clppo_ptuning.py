# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModel, AutoConfig, AutoModelForSequenceClassification, DataCollatorForSeq2Seq
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, create_reference_model
from trl.core import LengthSampler, respond_to_batch, compute_rouge
from trl.core import respond_to_batch, top_k_top_p_filtering, print_trainable_parameters
import torch.nn.functional as F
import os

from chatGLM_local.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from chatGLM_local.tokenization_chatglm import ChatGLMTokenizer


#1. 定义参数和ChatGLMForCausalLMWithValueHead模型
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="THUDM/chatglm-6b", metadata={"help": "the model name"})
    model_path: Optional[str] = field(default=None, metadata={"help":""})
    train_file: Optional[str] = field(default=None, metadata={"help":""})
    save_file: Optional[str] = field(default=None, metadata={"help":""})
    reward_model_path: Optional[str] = field(default=None, metadata={"help":""})
    mode: Optional[str] = field(default=None, metadata={"help":""})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    num_return_sequences: Optional[int] = field(default=3, metadata={"help": "num_return_sequences"})
    rerank_return: Optional[int] = field(default=3, metadata={"help": "num_return_sequences"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
config = PPOConfig(model_name="ChatGLMForCausalLMWithValueHead",
                       ppo_epochs=1,
                       mini_batch_size=script_args.mini_batch_size,
                       learning_rate=script_args.learning_rate,
                       adap_kl_ctrl=True,
                       init_kl_coef=0.1,
                       batch_size=script_args.batch_size,
                       max_grad_norm=1,
                       seed=2023,
                       )
# print("config:", config)
class ChatGLMForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    transformers_parent_class = ChatGLMForConditionalGeneration
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = ("summary_dropout_prob",
                      "v_head_initializer_range",
                      "v_head_init_strategy",
                      )
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.is_peft_model = False
def data_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)


# 2. 导入模型
chatglm_config = ChatGLMConfig.from_pretrained("THUDM/chatglm-6b")
chatglm_config.num_layers = 28
chatglm_config.pre_seq_len = 128
chatglm_config.prefix_projection = False
tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b", config=chatglm_config)
if script_args.model_path is not None:
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", config=chatglm_config).cuda().half()
    post = "请给出对以下微博的评论: 头条新闻娱乐电影新奥特曼正式预告斋藤工主演电影《新·奥特曼》发布全新预告。本作是奥特曼系列55周年纪念电影，由庵野秀明担任企划、编剧，樋口真嗣指导，超强卡司阵容包括斋藤工、长泽雅美、西岛秀俊、有冈大贵等，米津玄师演唱本片主题曲，电影将于5月13日在日本上映"
    # response, history = model.chat(tokenizer, post)
    # print("11111")
    # print("model.transformer.prefix_encoder",model.state_dict())
    prefix_state_dict = torch.load(os.path.join(script_args.model_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.float()
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # print_trainable_parameters(model, "PPO load param")
    # print("model.transformer.prefix_encoder",model.state_dict()["transformer.prefix_encoder"])
    post = "请给出对以下微博的评论: 头条新闻娱乐电影新奥特曼正式预告斋藤工主演电影《新·奥特曼》发布全新预告。本作是奥特曼系列55周年纪念电影，由庵野秀明担任企划、编剧，樋口真嗣指导，超强卡司阵容包括斋藤工、长泽雅美、西岛秀俊、有冈大贵等，米津玄师演唱本片主题曲，电影将于5月13日在日本上映"
    # response, history = model.chat(tokenizer, post)
    # print("response_after", response)
else:
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", config=chatglm_config).cuda().half()
# chatglm model 最后要更新的模型
# print("11111111")
model = model.quantize(4)
model.transformer.prefix_encoder.float()
model = ChatGLMForCausalLMWithValueHead(pretrained_model=model)
model.to(torch.device("cuda:0"))
# print_trainable_parameters(model,"ppo_load")
# ref_model 
ref_model = create_reference_model(model)
# reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_path)
reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_model.config.pad_token_id = reward_tokenizer.eos_token_id
# print("2222222222")

#3.导入数据集
data_files = {}
data_files["train"] = script_args.train_file
raw_datasets = load_dataset(
    "json",
    data_files=data_files,
    cache_dir=None,
    num_proc=1
)
# print("333333333333")
def preprocess_function_train(examples):
    max_source_length = 128
    max_seq_length = max_source_length 
    prompt_column = "weibo"
    response_column = "resp"
    model_inputs = {}
    inputs = []
    labels = []
    for i in range(len(examples[prompt_column])):
        query = examples[prompt_column][i]
        response = examples[response_column][i]
        response = tokenizer(response, max_length=32, truncation=True, padding='max_length')
        query = "请对以下微博做出评论:" + query
        inputs.append(query)
        labels.append(response["input_ids"])
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')# 返回 input_ids, attention_mask, position_ids
    model_inputs["response"] = labels
    print("model_inputs",model_inputs)
    return model_inputs
# We retrieve the dataloader by calling the `build_dataset` function.
column_names = raw_datasets["train"].column_names
raw_datasets = raw_datasets["train"]
# print("444444")
datasets = raw_datasets.map(preprocess_function_train, batched=True, num_proc=1, remove_columns=column_names)
print("data example:", datasets["input_ids"][0],len(datasets["input_ids"]),datasets["response"][0])


# 4. 定义trainer
data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=None,
        padding=False,
    )
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=datasets, data_collator=data_collator)
device = ppo_trainer.accelerator.device
print("device",device)
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
print("device",device)
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
gen_kwargs = {"max_length": 160,  "do_sample": True, "top_p": 0.7,
                 "temperature": 0.95, "num_return_sequences":script_args.num_return_sequences,"output_scores":True}


# 5.训练
pace = 0.01
total_steps = 0
for epoch in range(2):
    for batch_id, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        #1. compute golden response and generate responses
        total_steps += 1
        golden_response_inputs = batch["response"]
        batch.pop("response")
        query_tensors = torch.as_tensor(batch["input_ids"]).cuda()
        batch = batch.to(torch.device("cuda:0"))
        print("batch",batch["input_ids"].device)
        with torch.autocast("cuda"):
            response_tensors,probs = ppo_trainer.ChatGLM_generate(tokenizer=tokenizer,inputs=batch,gen_kwargs=gen_kwargs)
        golden_responses = []
        batch["response"] = tokenizer.batch_decode(response_tensors)
        for golden_response_input in golden_response_inputs:
            golden_response = tokenizer.batch_decode(golden_response_input)
            for i in range(script_args.num_return_sequences):
                golden_responses.append(golden_response)
        query_tensors_diversity = [query_tensor for query_tensor in list(query_tensors) for i in range(script_args.num_return_sequences)]

        #2. compute the response+query texts
        texts = []
        inputs = tokenizer.batch_decode(batch["input_ids"])
        for i in range(len(inputs)):
            q = inputs[i][:] #######
            for j in range(script_args.num_return_sequences):
                index = i*script_args.num_return_sequences +j 
                r = batch["response"][index]
                texts.append(r + " " + tokenizer.bos_token + " " + q)            

        #3. compute rewards and rouge_rewards
        tokenized_texts = reward_tokenizer(texts, max_length=160,truncation=True)
        features = []
        for input_id, attention_mask in zip(tokenized_texts["input_ids"], tokenized_texts["attention_mask"]):
            features.append({"input_ids": input_id, "attention_mask": attention_mask})
            batch_features = reward_tokenizer.pad(
                features,
                padding=True,
                max_length=160,
                pad_to_multiple_of=None,
                return_tensors="pt",
            )
        rewards = reward_model(input_ids=batch_features["input_ids"], attention_mask=batch_features["attention_mask"])[0]
        if script_args.mode == "ppo_ori":
            ppo_trainer.step(query_tensors_diversity, list(response_tensors), list(rewards)) 
            print("saving model................",pace,total_steps)
            ppo_trainer.save_prefix(script_args.save_file) 
            continue    
        
        rouge_rewards = compute_rouge(batch["response"],golden_responses)
        for i in range(len(rewards)):
            rewards[i] += rouge_rewards[i]
        rewards = rewards.detach()

        
        ###4. Rewards ranking and fitltering
        train_index = []
        for i in range(script_args.batch_size):
            start = i * script_args.num_return_sequences
            end = start + script_args.num_return_sequences
            # train_rewards = np.array(rewards[start:end])
            train_rewards = [a[0] for a in rewards[start:end]]
            train_rewards = np.array(train_rewards)
            index = train_rewards.argsort()[-script_args.rerank_return:][::-1]
            index = [int(a)+start for a in index]
            train_index += index
        query_filter = []
        response_filter = []
        reward_filter = []
        probs_filter = []
        for num in train_index:
            query_filter.append(query_tensors_diversity[num])
            response_filter.append(response_tensors[num])
            reward_filter.append(rewards[num])
            probs_filter.append(probs[num])
        print("after reranking", len(query_filter),len(response_filter),len(reward_filter),len(probs_filter))
        
        #5. self_paced
        query_final = []
        response_final = []
        reward_final = []
        for num in range(len(query_filter)):
            pace_compare = abs(reward_filter[num].numpy()*probs_filter[num])
            print("pace_compare",pace_compare)
            if  pace_compare > pace:
                query_final.append(query_filter[num])
                response_final.append(response_filter[num])
                reward_final.append(reward_filter[num])
                # print("reward_final",reward_final)
        print("len(reward_final)",len(reward_final))
        if len(reward_final) >= script_args.mini_batch_size:
            if len(reward_final) % 2 != 0:
                query_final = query_final[:-1]
                response_final = response_final[:-1]
                reward_final = reward_final[:-1]
            print("after self_pace",len(query_final),len(response_final),len(reward_final))
            ppo_trainer.step(query_final, response_final, reward_final)

        
        pace = pace - 0.05 * pace
        # if total_steps % 100 == 0: 
        #     print("saving model................",pace,total_steps)
        #     ppo_trainer.save_prefix(f"{script_args.save_file}{total_steps}") 
ppo_trainer.save_prefix(script_args.save_file)