CUDA_VISIBLE_DEVICES=0 python chatGLM_ptuning.py \
    --do_train \
    --train_file /home/your_dir/datasets/weiboGen/train.json \
    --prompt_column weibo \
    --response_column resp \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir /home/your_dir/output/chatglm_ptuning/ \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 32 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --quantization_bit 4 \
    --source_prefix "请对以下微博做出评论"