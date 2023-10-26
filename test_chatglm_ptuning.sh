TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python3 chatglm_ptuning.py \
    --do_predict \
    --test_file /home/your_dir/datasets/weiboGen/valid.json \
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
    --output_dir /home/your_dir/output/\
    --ptuning_checkpoint /home/your_dir/output/chatglm_clppo_ptuning/ \
    --pre_seq_len 128 \
    
    
