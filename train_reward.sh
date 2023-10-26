CUDA_VISIBLE_DEVICES=1 python reward.py  \
    --model_name gpt2 \
    --num_train_epochs 20 \
    --train_file /home/yex/data/ppo_weibo/weiboReward/train_reward_v1.json \
    --dev_file /home/yex/data/ppo_weibo/weiboReward/test_reward_v1.json \
    --output_dir /home/yex/output/gpt2_rewardModel