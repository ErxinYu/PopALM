PRE_SEQ_LEN=128
ptuning_checkpoint=tweetGen_ptuning/checkpoint-1
TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python3 chatGLM_ptuning.py \
    --do_predict \
    --test_file /home/xxx/data/ppo_tweet/test.json \
    --overwrite_cache \
    --prompt_column weibo \
    --response_column resp \
    --model_name_or_path THUDM/chatglm-6b \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 32 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --quantization_bit 4 \
    --num_return_sequences 1\
    --output_dir /home/xxx/output/tweetGen_ptuning/Top1\
    --ptuning_checkpoint /home/xxx/output/$ptuning_checkpoint \
    --pre_seq_len $PRE_SEQ_LEN \
    
    
