CUDA_VISIBLE_DEVICES=1 python reward.py  \
    --model_name gpt2 \
    --num_train_epochs 20 \
    --train_file /home/your_dir/data/datasets/weiboReward/train_reward.json \
    --dev_file /home/your_dir/data/datasets/weiboReward/test_reward.json \
    --output_dir /home/your_dir/output/gpt2_rewardModel