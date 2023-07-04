TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=2 python3 chatGLM_lora.py \
    --do_predict \
    --test_file /home/xxx/data/ppo_weibo/weiboGen/test.json \
    --overwrite_cache \
    --prompt_column weibo \
    --response_column resp \
    --model_name_or_path THUDM/chatglm-6b \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 32 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --num_return_sequences 1\
    --output_dir /home/xxx/output/lora_PPOGen_epoch2_gen1/Top1 \
    --lora_checkpoint /home/xxx/output/lora_PPOGen_epoch2_gen1 \
    
    
