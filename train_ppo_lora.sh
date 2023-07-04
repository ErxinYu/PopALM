TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES="0,1" python  PPO_lora.py \
    --batch_size 2 \
    --mini_batch_size 2 \
    --num_return_sequences 1 \
    --mode ppo_ori \
    --rerank_return 1 \
    --save_file /home/xxx/output/lora_PPOGen_epoch2_gen1/ \
    --model_path /home/xxx/output/loraGen/checkpoint-3000 \
    --train_file /home/xxx/data/ppo_weibo/weiboGen/train_ppo.json \
    --reward_model_path /home/xxx/output/gpt2_rewardModel/checkpoint-1406

    