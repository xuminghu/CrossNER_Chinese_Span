
## Directly Fine-tune
CUDA_VISIBLE_DEVICES=0 nohup python main.py --num_target_tag 17 --num_source_tag 23 --batch_size 32 --tgt_dm address --seed 1 --model_name bert-base-chinese >> logs/address_seed1_directly.log 2>&1 &

## Pre-train then Fine-tune
CUDA_VISIBLE_DEVICES=0 nohup python main.py --num_target_tag 17 --num_source_tag 23 --batch_size 32 --tgt_dm address --seed 1 --source --model_name bert-base-chinese >> logs/address_seed1_source.log 2>&1 &
