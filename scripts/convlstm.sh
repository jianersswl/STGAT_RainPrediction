export CUDA_VISIBLE_DEVICES=0
python train_GBA.py --model="convlstm" --device=0 --seed=10 --input_data gdaps_kim \
                --num_epochs=200 \
                --rain_thresholds 0.4 52.0 100.0\
                --log_dir logs/logs_1105_GBA \
                --batch_size 10 \
                --window_size 1 \
                --lr 0.0001 \
                --dropout 0.0 \
                --loss ce+mse \
                --alpha 10 \
                --wd_ep 100 \
                --use_two \
                --seq_length 3 \
                --weight_version 6\
                --custom_name="China_convlstm_bs8_ep100_seed_10_alpha10_wd_ep100" &