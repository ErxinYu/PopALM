import logging
import os
import sys
import json
import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import torch.nn.functional as F

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from chatGLM_local.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig, ChatGLMModel
from chatGLM_local.tokenization_chatglm import ChatGLMTokenizer
from trl.trainer.trainer_seq2seq import Seq2SeqTrainer
from trl.arguments import ModelArguments, DataTrainingArguments
from trl.core import print_trainable_parameters,print_dataset_example

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"

def main():

    # 1.preccss args and logs
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)) #可以从控制台中读参数，把对应的参数名的值传输到不同的argument中
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("data_args", data_args, "\n")
    print("model_args", model_args, "\n")
    print("training_args", training_args, "\n")
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    set_seed(training_args.seed)


    #2.Raw dataset process
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir, #None
        use_auth_token=True if model_args.use_auth_token else None, #False
    )


    #3.Load pretrained model and tokenizer
    config = ChatGLMConfig.from_pretrained(model_args.model_name_or_path)
    config.num_layers = 28
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    print("model_args.prefix_projection",model_args.prefix_projection)
    print("model_args.pre_seq_len",model_args.pre_seq_len)
    print("model_args.model_name_or_path",model_args.model_name_or_path)
    print("config",config)
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)
    print("model_args.ptuning_checkpoint",model_args.ptuning_checkpoint)
    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", config=config).cuda().half()
        print("model_before")
        # print_trainable_parameters(model)  
        post = "请给出对以下微博的评论: 头条新闻娱乐电影新奥特曼正式预告斋藤工主演电影《新·奥特曼》发布全新预告。本作是奥特曼系列55周年纪念电影，由庵野秀明担任企划、编剧，樋口真嗣指导，超强卡司阵容包括斋藤工、长泽雅美、西岛秀俊、有冈大贵等，米津玄师演唱本片主题曲，电影将于5月13日在日本上映"
        response, history = model.chat(tokenizer, post)
        print(response)
        print(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        print("new_prefix_state_dict:\n", new_prefix_state_dict)
        if model_args.pre_seq_len is not None: #先让模型参数和要加载的参数保持一致，都是float，model冻结部分不用改的原因是 都是从头加载，然后在quantize
            model.transformer.prefix_encoder.float()
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        # print_trainable_parameters(model)
        post = "请给出对以下微博的评论: 头条新闻娱乐电影新奥特曼正式预告斋藤工主演电影《新·奥特曼》发布全新预告。本作是奥特曼系列55周年纪念电影，由庵野秀明担任企划、编剧，樋口真嗣指导，超强卡司阵容包括斋藤工、长泽雅美、西岛秀俊、有冈大贵等，米津玄师演唱本片主题曲，电影将于5月13日在日本上映"
        response, history = model.chat(tokenizer, post)
        print(response)
    else:
        model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", config=config).cuda()
        model = model.half()
        post = "请给出对以下微博的评论: 头条新闻娱乐电影新奥特曼正式预告斋藤工主演电影《新·奥特曼》发布全新预告。本作是奥特曼系列55周年纪念电影，由庵野秀明担任企划、编剧，樋口真嗣指导，超强卡司阵容包括斋藤工、长泽雅美、西岛秀俊、有冈大贵等，米津玄师演唱本片主题曲，电影将于5月13日在日本上映"
        response, history = model.chat(tokenizer, post)
        print(response)
        # print_trainable_parameters(model,"model_load_from_ori")  
    if model_args.quantization_bit is not None:# 这时候模型应该是fp16
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit) # 这时候模型应该是INT4
        # print_trainable_parameters(model,"quantization_bit_load")
    if model_args.pre_seq_len is not None:  #为了从头加载模型的做法 如果load from checkpoint,这个就没啥用
        model.transformer.prefix_encoder.float()      
    print_trainable_parameters(model,"final_load")

    #4. Preprocessing the datasets.
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        print("column_names",column_names)
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    max_target_length = data_args.max_target_length    

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length 
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        total_source_length = 0
        total_target_length = 0
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]
                prompt = prefix + query
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
                total_source_length += len(a_ids)
                total_target_length += len(b_ids)

                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]
                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]
                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids) #a_ids + gmask_id + bos_id + b_id + eos_id
                source_length = input_ids.index(tokenizer.bos_token_id)
                labels = [-100] * source_length + input_ids[source_length:] #source length的位置用-100掩盖,计算loss的时候不用算
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len # pad_token_id = 3 [source,gmask,bos,target,eos,3333]
                labels = labels + [tokenizer.pad_token_id] * pad_len 
                # print("labels_before", labels)
                if data_args.ignore_pad_token_for_loss: #true
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels] # 把pad_id改成-100 [-100,-100,-100,gmask,bos,target,eos,-100,-100]
                # print("labels_after", labels)
                # print("input_ids",input_ids)
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                #input_ids = [source,        gmask,bos,target,eos,3,..3,..3]
                #labels =    [-100* source,  gmask,bos,target,eos,-100,-100]
            else:
                print("error, No prompt_column or response_column in preprocess_function_eval")
        print("avg_source_length",total_source_length/len(examples[prompt_column]))
        print("avg_target_length",total_target_length/len(examples[prompt_column]))
        return model_inputs
    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                prompt = prefix + query
                inputs.append(prompt)
                targets.append(examples[response_column][i][0])
            else:
                print("error, No prompt_column or response_column in preprocess_function_eval")
                print("examples[prompt_column][i] and examples[response_column][i]",examples[prompt_column][i],examples[response_column][i])
        print("inputs",len(inputs),inputs[0])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding='max_length')# 返回 input_ids, attention_mask, position_ids
        print("targets",len(targets),targets[0])
        inputs_targets = tokenizer(text_target=targets, max_length=32, truncation=True)
        if data_args.ignore_pad_token_for_loss:
            for target in inputs_targets["input_ids"]:
               target = [ [(t if t != tokenizer.pad_token_id else -100) for t in target]]
        model_inputs["labels"] = inputs_targets["input_ids"] 
        #model_input 有
        #input_ids: [3...3, 5, 65647, 80229, 102590, 65611, 64160, 70862, 63828, 6, 63958, 65740, 79958, 63926, 106086, 106086, 65611, 
        #74777, 117872, 63825, 68324, 440, 130001, 130004] 长度是max_source_length
        #attention_mask, position_ids 和 labels = [5, 65205, 64189, 67713, 64965, 64273, 130001, 130004]
        return model_inputs
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers, #None
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache, #False
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(tokenizer,train_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(tokenizer,predict_dataset[0])
 
    #5. Set up training
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    ) #在train的时候是None,在eval和predict的时候是max_target_length=32
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    ) #None
    # Initialize our Trainer
   
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )


    #6. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    #7.Prediction
    results = {}
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=160, do_sample=True, top_p=0.7, temperature=0.95, num_return_sequences=data_args.num_return_sequences)
        print("predict_results",predict_results.predictions,type(predict_results.predictions))
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p in predictions:
                        res = json.dumps({"predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
