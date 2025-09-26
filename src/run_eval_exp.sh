#!/bin/bash

###### visualization
#nohup python main_eval.py -p topk_avg -t 0.91 -fcv > logs/z_avg.log 2>&1 &
#nohup python main_eval.py -p topk_raw -t 0.91 -fcv > logs/z_raw.log 2>&1 &
#nohup python main_eval.py -p topk_cluster -t 0.91 -nc 5 -fcv > logs/z_cluster_5.log 2>&1 &
#nohup python main_eval.py -p topk_cluster -t 0.91 -nc 15 -fcv > logs/z_cluster_15.log 2>&1 &
#nohup python main_eval.py -p topk_cluster -t 0.91 -nc 30 -fcv > logs/z_cluster_30.log 2>&1 &

# Run experiments sequentially to avoid GPU conflicts
CUDA_VISIBLE_DEVICES=0 python main_eval.py -p topk_avg -t 0.91 -fcv -bc -did 0 > logs/z_avg_classifier.log 2>&1
CUDA_VISIBLE_DEVICES=0 python main_eval.py -p topk_raw -t 0.91 -fcv -bc -did 0 > logs/z_raw_classifier.log 2>&1
CUDA_VISIBLE_DEVICES=0 python main_eval.py -p topk_cluster -t 0.91 -nc 5 -fcv -bc -did 0 > logs/z_cluster_5_classifier.log 2>&1
CUDA_VISIBLE_DEVICES=0 python main_eval.py -p topk_cluster -t 0.91 -nc 15 -fcv -bc -did 0 > logs/z_cluster_15_classifier.log 2>&1
CUDA_VISIBLE_DEVICES=0 python main_eval.py -p topk_cluster -t 0.91 -nc 30 -fcv -bc -did 0 > logs/z_cluster_30_classifier.log 2>&1

#tau_list=(0.95 0.97 0.99)
#

#for TAU in "${tau_list[@]}"
#do
##  nohup python main_eval.py -p topk_avg -did 0 -t ${TAU} -fcv > avg.log 2>&1 &
##  nohup python main_eval.py -p topk_raw -did 0 -t ${TAU} -fcv > raw.log 2>&1 &
##  nohup python main_eval.py -p topk_cluster -did 0 -t ${TAU} -nc 5 -fcv > cluster5.log 2>&1 &
##  nohup python main_eval.py -p topk_cluster -did 1 -t ${TAU} -nc 10 -fcv > cluster10.log 2>&1 &
#  nohup python main_eval.py -p topk_cluster -did 1 -t ${TAU} -nc 15 -fcv > cluster15.log 2>&1 &
#  nohup python main_eval.py -p topk_cluster -did 1 -t ${TAU} -nc 20 -fcv > cluster20.log 2>&1 &
#done
