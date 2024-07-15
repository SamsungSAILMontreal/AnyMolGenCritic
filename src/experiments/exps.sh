# It assumes 1 GPU; you can change it to as many as you want using "CUDA_VISIBLE_DEVICES=0,...,i-1" and --n_gpu i
# If sampling takes too long, use more GPUs and if you get sampling memory problems (Killed), set --sample_batch_size lower.

cd /src

## HIV, BACE and BBBP ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name hiv --num_layers 3 --tag exp_hiv10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name hiv --num_layers 3 --tag exp_hiv10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.5 --guidance_ood 1.5 --test
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name hiv --num_layers 3 --tag exp_hiv10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.5 --guidance_ood 1.5 --best_out_of_k 5 --test

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bace --num_layers 3 --tag exp_bace10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bace --num_layers 3 --tag exp_bace10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.5 --guidance_ood 1.5 --test
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bace --num_layers 3 --tag exp_bace10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.5 --guidance_ood 1.5 --best_out_of_k 5 --test

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bbbp --num_layers 3 --tag exp_bbbp10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bbbp --num_layers 3 --tag exp_bbbp10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.5 --guidance_ood 1.5 --test
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bbbp --num_layers 3 --tag exp_bbbp10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.5 --guidance_ood 1.5 --best_out_of_k 5 --test


#real data 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name hiv --num_layers 3 --tag exp_hiv10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.0 --guidance_ood 1.0 --test --test_on_train_data
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name hiv --num_layers 3 --tag exp_hiv10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.0 --guidance_ood 1.0 --test --test_on_test_data

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bace --num_layers 3 --tag exp_bace10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.0 --guidance_ood 1.0 --test --test_on_train_data
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bace --num_layers 3 --tag exp_bace10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.0 --guidance_ood 1.0 --test --test_on_test_data

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bbbp --num_layers 3 --tag exp_bbbp10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.0 --guidance_ood 1.0 --test --test_on_train_data
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name bbbp --num_layers 3 --tag exp_bbbp10kepoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 20 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 10000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--max_len 300 --gpt --no_bias --rmsnorm --rotary --nhead 16 --swiglu --expand_scale 2.0 \
--guidance 1.0 --guidance_ood 1.0 --test --test_on_test_data


## Zinc in-distribution ##

# Same pretrained model as Zinc out-of-distribution
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --num_layers 3 --tag exp_zinc50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond 

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --num_layers 3 --tag exp_zinc50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --no_ood \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --num_layers 3 --tag exp_zinc50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --best_out_of_k 5 --no_ood \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778


## Zinc out-of-distribution ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --num_layers 3 --tag exp_zinc50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --num_layers 3 --tag exp_zinc50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --guidance_rand \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --num_layers 3 --tag exp_zinc50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --best_out_of_k 5 \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --num_layers 3 --tag exp_zinc50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --best_out_of_k 5  --guidance_rand \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778

## QM9 in-distribution ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm950epoch_swiglu --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond 

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm950epoch_swiglu --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --no_ood \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm950epoch_swiglu --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5  --best_out_of_k 5 --no_ood \
--test --ood_values 580 84 8.194 -3.281 1.2861 0.1778

## QM9 reward maximization ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm9gflownet50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--gflownet --max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm9gflownet50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--gflownet --max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond \
--guidance 1.5 --guidance_ood 1.5 --test
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm9gflownet50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--gflownet --max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond --best_out_of_k 5 \
--guidance 1.5 --guidance_ood 1.5 --test
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm9gflownet50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--gflownet --max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond --best_out_of_k 20 \
--guidance 1.5 --guidance_ood 1.5 --test
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name qm9 --num_layers 3 --tag exp_qm9gflownet50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init \
--gflownet --max_len 150 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond --best_out_of_k 100 \
--guidance 1.5 --guidance_ood 1.5 --test

## Chromophore out-of-distribution

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --num_layers 3 --tag exp_chromophore50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 1000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary --log_every_n_steps 24

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --num_layers 3 --tag exp_chromophore1000epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 1000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --num_layers 3 --tag exp_chromophore50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 1000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 100

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --num_layers 3 --tag exp_chromophore50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 1000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --guidance_rand --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --num_layers 3 --tag exp_chromophore50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 1000 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --guidance_rand --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 100

## Zinc pretrain + Chromophore finetune

# pretrain
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name zinc --finetune_dataset_name chromophore --finetune_scaler_vocab --num_layers 3 --tag exp_zinc_pretrain_50epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 512 --lr 1e-3 --max_epochs 50 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond --save_checkpoint_dir your_dir

# fine-tune
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --finetune_dataset_name zinc --num_layers 3 --tag exp_chromophore_finetune100epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 50 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 100 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary --load_checkpoint_path your_dir/exp_zinc_pretrain_50epoch/last.ckpt

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --finetune_dataset_name zinc --num_layers 3 --tag exp_chromophore_finetune100epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 100 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --finetune_dataset_name zinc --num_layers 3 --tag exp_chromophore_finetune100epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 100 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 100

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --finetune_dataset_name zinc --num_layers 3 --tag exp_chromophore_finetune100epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 100 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --guidance_rand --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name chromophore --finetune_dataset_name zinc --num_layers 3 --tag exp_chromophore_finetune100epoch --bf16 \
--check_sample_every_n_epoch 999 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 \
--batch_size 128 --lr 2.5e-4 --max_epochs 100 --n_gpu 1 --randomize_order --start_random --scaling_type std --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 600 --gpt --no_bias --rmsnorm --rotary \
--guidance 1.5 --guidance_ood 1.5 --guidance_rand --only_ood --no_test_step --not_allow_empty_bond \
--test --ood_values 1538 -531 28.6915 -13.6292 1.2355 -0.5406 --sample_batch_size 100 --num_samples_ood 100 --best_out_of_k 100

