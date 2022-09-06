export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name balls \
    --train_data_paths /root/autodl-tmp/CoPhy_112/ballsCF/4 \
    --valid_data_paths /root/autodl-tmp/CoPhy_112/ballsCF/4 \
    --save_dir checkpoints/balls_predrnn_v2_run1 \
    --gen_frm_dir results/balls_predrnn_v2_run1 \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 112 \
    --img_channel 3 \
    --input_length 2 \
    --total_length 75 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 4 \
    --max_iterations 80000 \
    --display_interval 1 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
#    --pretrained_model ./checkpoints/mnist_predrnn_v2/mnist_model.ckpt