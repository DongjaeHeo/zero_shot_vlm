#!/bin/bash

export HF_HUB_OFFLINE=1

nohup python generate_crop_dataset.py 0 Headstone_without_Mound 42 > logs/z_crop_data_Headstone_without_Mound.log 2>&1 &
nohup python generate_crop_dataset.py 0 Mound_with_Headstone 42 > logs/z_crop_data_Mound_with_Headstone.log 2>&1 &
nohup python generate_crop_dataset.py 1 factory 42 > logs/z_crop_data_factory.log 2>&1 &
nohup python generate_crop_dataset.py 1 group_of_trees 42 > logs/z_crop_data_group_of_trees.log 2>&1 &
#
#wait

nohup python generate_crop_dataset.py 0 house 42 > logs/z_crop_data_house.log 2>&1 &
nohup python generate_crop_dataset.py 0 polytunnel_no_cover 42 > logs/z_crop_data_polytunnel_no_cover.log 2>&1 &

nohup python generate_crop_dataset.py 1 polytunnel_with_cover 42 > logs/z_crop_data_polytunnel_with_cover.log 2>&1 &
nohup python generate_crop_dataset.py 1 single_tree 42 > logs/z_crop_data_single_tree.log 2>&1 &

#wait

nohup python generate_crop_dataset.py 1 with_crop 42 > logs/z_crop_data_with_crop.log 2>&1 &
nohup python generate_crop_dataset.py 1 without_crop 42 > logs/z_crop_data_without_crop.log 2>&1 &
