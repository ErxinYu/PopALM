TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES="0" python  chatglm_ppo_ptuning.py \
    --batch_size 4 \
    --mini_batch_size 2 \
    --num_return_sequences 3 \
    --mode cl-ppo \
    --rerank_return 1 \
    --save_file /home/yex/output/chatglm_clppo_ptuning/ \
    --model_path /home/yex/output/chatglm_ptuning/checkpoint-3000 \
    --train_file /home/yex/data/ppo_weibo/weiboGen/train_ppo.json \
    --reward_model_path /home/yex/output/gpt2_rewardModel/checkpoint-1406

    