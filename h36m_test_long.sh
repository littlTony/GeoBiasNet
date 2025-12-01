# 单机多卡 DDP 训练
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_h36m_ddp_long.py \
    --world_size 4 \
    --data_dir "./dataset/h3.6m" \
    --kernel_size 10 \
    --dct_n 60 \
    --input_n 10 \
    --train_output_n 50 \
    --output_n 50 \
    --skip_rate 1 \
    --batch_size 16 \
    --test_batch_size 16 \
    --in_features 66 \
    --d_model 16 \
    --cuda_idx cuda:5 \
    --lr_now 0.005 \
    --epoch 50 \
    --test_sample_num -1 