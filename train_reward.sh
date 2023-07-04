CUDA_VISIBLE_DEVICES=1 python reward.py  \
    --num_train_epochs 10 \
    --train_file /home/xxx/data/ppo_weibo/weiboReward/train_reward.json \
    --dev_file /home/xxx/data/ppo_weibo/weiboReward/dev_reward.json \
    --output_dir /home/xxx/output/gpt2_rewardModel