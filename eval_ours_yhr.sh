#!/bin/sh




for no in 19 16 14 12 11 9 7 2; do
    CUDA_VISIBLE_DEVICES=2 python dreamplace/script_new_evaluate_yhr.py \
    --name test-11-23 --model pretrain_dac11_12_14_16_19 --no ${no}
done