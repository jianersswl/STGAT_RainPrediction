export CUDA_VISIBLE_DEVICES=0
python train_GBA.py --model="SwinUnet_Two" --device=0 --seed=10 --input_data="gdaps_kim" \
                --num_epochs=500 \
                --rain_thresholds 0.4 52.0 100.0 \
                --log_dir logs/logs_1111_China \
                --batch_size 10\
                --lr 0.0001 \
                --dropout 0.0 \
                --use_two \
                --loss ce+mse \
                --alpha 10 \
                --kernel_size 3 \
                --weight_version 5 \
                --wd_ep 100 \
                --custom_name="GBA_SwinUnet_bs128_ep500_v2_seed_10_alpha10_lrE-3_witht2p_matrix_0d452100_class4" &