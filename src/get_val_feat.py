#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
검증 특징 추출 스크립트 - Zero-shot 객체 검출 시스템의 검증 데이터 특징 추출

이 스크립트는 다음과 같은 기능을 수행합니다:
1. 검증 이미지에서 OWLv2 모델을 사용한 객체 검출 수행
2. 슬라이딩 윈도우를 통한 대형 이미지 처리 (타일링)
3. 검출된 바운딩 박스와 임베딩을 NPZ 파일로 저장
4. 클래스별 이미지 디렉토리 처리
5. 중복 처리 방지 (이미 NPZ 파일이 있으면 건너뛰기)

핵심 특징:
- OWLv2 모델 기반 객체 검출
- 효율적인 타일링 전략
- NPZ 형태의 특징 저장
- 클래스별 독립적 처리
"""

import argparse
import os, csv, random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Union
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from utils.infer_utils import *
from utils.eval_utils import *
from utils.model_utils import *

# ========= 설정 =========
OUT_IDX    = Path("../index").resolve()                                   # pos_<class>.pt 파일들이 저장된 위치
VIZ_DIR    = Path("../results/viz_fixed_5").resolve()                             # 시각화 결과 (k<K>별로 그룹화)
METRICS_DIR= Path("../results/metrics_fixed_5").resolve()                         # 클래스별 CSV (k<K>별로 그룹화)
MODEL_NAME = "google/owlv2-large-patch14-ensemble"

# 단일 운영점 설정
IOU_MATCH  = 0.30        # TP/FN을 위한 IoU 임계값 (원래 헤더의 coverage@0.5)
TOP_K      = 15          # 메트릭 경로용 선택적 상한 (NMS 후 상위 15개 유지)
MIN_AREA   = 0.0         # 사전에 작은 박스 제거하지 않음 (필요시 >0으로 설정)

# 시각화는 메트릭과 동일한 τ와 NMS 사용
TOP_K_VIS  = 30

# 타일링 설정 (마이너와 동일)
PATCH, OVERLAP = 1008, 500  # 패치 크기: 1008x1008, 오버랩: 500픽셀
STRIDE, CORE   = PATCH-OVERLAP, 250  # 스트라이드: 508픽셀, 코어 영역: 250픽셀
TILE_EXCEPT    = {"group_of_trees"}  # 타일링 제외 클래스 (대소문자 구분 없음)

# RELAXED 평가를 위한 클래스 그룹 (pos_*.pt 파일의 'class' 필드와 일치해야 함)
CLASS_GROUPS_RAW: Dict[str, List[str]] = {
    "building":   ["house", "factory", "shed"],  # 건물 관련 클래스들
    "vegetation": ["single_tree", "group_of_trees"],  # 식생 관련 클래스들
    "burial":     ["Mound_with_Headstone", "Headstone_without_Mound"],  # 묘지 관련 클래스들
    "polytunnel": ["polytunnel_with_cover", "polytunnel_no_cover"],  # 비닐하우스 관련 클래스들
    "field":      ["with_crop", "without_crop"],  # 농지 관련 클래스들
}

# 그룹별 ID 매핑
CLASS_GROUPS_ID: Dict[str, int] = {
    "burial": 0,      # 묘지 그룹
    "vegetation": 1,  # 식생 그룹
    "polytunnel": 2,  # 비닐하우스 그룹
    "building": 3,    # 건물 그룹
    "field": 4,       # 농지 그룹
}

# 전체 클래스명 리스트
CLASS_NAMES = [
    'Headstone_without_Mound',
    'Mound_with_Headstone',
    'factory',
    'group_of_trees',
    'house',
    'polytunnel_no_cover',
    'polytunnel_with_cover',
    'single_tree',
    'with_crop',
    'without_crop']

def run(args):
    """
    검증 특징 추출 메인 함수 - 클래스별로 이미지를 처리하여 특징 추출
    
    Args:
        args: 명령행 인수 (eval_sample, device_id 등)
    """
    # 데이터셋 경로 설정
    if args.eval_sample:
        SRC_ROOT = Path("../dataset/sample_data").resolve()  # 샘플 데이터셋: class/{images,labels}
    else:
        SRC_ROOT = Path("/home/dromii_shared/obstacle_subclass").resolve()  # 전체 데이터셋

    # 각 클래스별로 처리
    for cname in CLASS_NAMES:
        # 이미지 디렉토리 설정
        if args.eval_sample:
            img_dir = SRC_ROOT / "images"  # 샘플 데이터는 단일 이미지 디렉토리
        else:
            img_dir = SRC_ROOT / cname / "images"  # 클래스별 이미지 디렉토리

        # 타일링 여부 결정 (특정 클래스는 타일링 제외)
        do_tiles = (cname.lower() not in {s.lower() for s in TILE_EXCEPT})

        # 지원하는 이미지 확장자로 이미지 파일 수집
        unseen = []
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.tif", "*.tiff"):
            unseen += [p for p in img_dir.glob(ext)]

        unseen = sorted(unseen)  # 정렬하여 일관된 처리 순서 보장

        # 각 이미지에 대해 특징 추출 수행
        for img_path in unseen:
            img = load_image(img_path)  # EXIF 정보를 고려한 이미지 로드

            # NPZ 파일 경로 설정 (이미지와 동일한 이름, .npz 확장자)
            npz_file = img_path.with_suffix(".npz")

            if npz_file.exists():
                print(f'{npz_file.name} embedding already exists.')  # 이미 처리된 파일은 건너뛰기
            else:
                print(f'{npz_file.name} embedding extract.')
                # OWLv2 모델을 사용한 객체 검출 및 임베딩 추출
                boxes, embs = run_detector_with_tiling(img, MIN_AREA, torch.device("cuda" if torch.cuda.is_available() else "cpu"), do_tiles)
                # 바운딩 박스와 임베딩을 NPZ 파일로 저장
                np.savez(npz_file, boxes=boxes, embs=embs)


if __name__ == "__main__":
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', "--seed", type=int, default=42, help="재현 가능한 결과를 위한 시드")
    parser.add_argument('-did', "--device_id", type=str, default="0", help="사용할 GPU 디바이스 ID")
    parser.add_argument('-t', "--tau", type=float, default=0.95, help="임계값 (현재 사용되지 않음)")
    parser.add_argument('-nt', "--nms_iou", type=float, default=0.10, help="NMS IoU 임계값 (현재 사용되지 않음)")
    parser.add_argument('-p', "--proto", type=str, default='topk_avg', help="프로토타입 방식 ['topk_avg', 'topk_raw', 'topk_cluster'] (현재 사용되지 않음)")
    parser.add_argument('-nc', "--num_clusters", type=int, default=4, help="클러스터 수 (현재 사용되지 않음)")
    parser.add_argument('-sample', "--eval_sample", action='store_true', help="샘플 데이터셋 사용 여부")
    args = parser.parse_args()

    # 재현 가능한 결과를 위한 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # GPU 디바이스 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    # 메인 함수 실행
    run(args)