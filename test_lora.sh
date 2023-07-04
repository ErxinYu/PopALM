lora_checkpoint=loraGen/checkpoint-3000
TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=1 python3 chatGLM_lora.py \
    --do_predict \
    --test_file /home/XXX/data/ppo_weibo/weiboGen/test.json \
    --overwrite_cache \
    --prompt_column weibo \
    --response_column resp \
    --model_name_or_path THUDM/chatglm-6b \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 32 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --quantization_bit 4 \
    --num_return_sequences 3\
    --output_dir /home/xxx/output/loraGen/Top3 \
    --fp16 True \
    --lora_checkpoint /home/xxx/output/$lora_checkpoint \
    
    
