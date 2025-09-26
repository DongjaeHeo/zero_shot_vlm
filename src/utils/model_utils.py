#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모델 유틸리티 모듈 - Zero-shot 객체 검출 시스템의 모델 관련 유틸리티 함수들

이 모듈은 다음과 같은 기능을 제공합니다:
1. OWLv2 모델을 사용한 객체 검출 및 임베딩 추출
2. 슬라이딩 윈도우를 통한 대형 이미지 처리 (타일링)
3. 다양한 CNN 백본 모델 생성 (MobileNetV3, ResNet18, ConvNeXt)
4. 바이너리 분류기를 위한 모델 아키텍처 구성

핵심 특징:
- OWLv2 모델 기반 객체 검출
- 효율적인 타일링 전략
- 다양한 백본 아키텍처 지원
- GPU/CPU 호환성
"""

import os, csv, random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Union
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from utils.infer_utils import *
from utils.eval_utils import *

# OWLv2 모델 설정
MODEL_NAME = "google/owlv2-large-patch14-ensemble"

# 타일링 설정 (마이너와 동일)
PATCH, OVERLAP = 1008, 500  # 패치 크기: 1008x1008, 오버랩: 500픽셀
STRIDE, CORE   = PATCH-OVERLAP, 250  # 스트라이드: 508픽셀, 코어 영역: 250픽셀
TILE_EXCEPT    = {"group_of_trees"}  # 타일링 제외 클래스 (대소문자 구분 없음)


@torch.no_grad()
def infer_boxes_and_embs(img: Image.Image, DEVICE):
    """
    OWLv2 모델을 사용한 객체 검출 및 임베딩 추출
    
    Args:
        img: PIL 이미지 객체
        DEVICE: GPU/CPU 디바이스
    
    Returns:
        (boxes, embs): 바운딩 박스 배열 [N, 4], 임베딩 배열 [N, D]
    """
    # OWLv2 프로세서와 모델 로드
    proc = Owlv2Processor.from_pretrained(MODEL_NAME)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # 이미지 전처리 및 모델 입력 준비
    inp = proc(images=img, return_tensors="pt").to(DEVICE)
    fmap = model.image_embedder(inp.pixel_values)[0]  # 이미지 임베딩 맵 추출

    B, h, w, D = fmap.shape
    feats = fmap.reshape(1, h * w, D)  # [1, H*W, D] 형태로 변환

    # 바운딩 박스 예측
    pred = model.box_predictor(feats, feature_map=fmap)[0].detach().cpu().numpy()
    # 클래스 예측 (임베딩 추출용)
    _, cls = model.class_predictor(feats)

    # 좌표 변환 및 정규화
    boxes = np.clip(boxes_cxcywh_to_xyxy_norm(pred), 0.0, 1.0)
    # L2 정규화된 임베딩 추출
    embs = l2norm_t(cls[0].float(), dim=-1).cpu().numpy().astype(np.float32)
    return boxes, embs


def run_detector_with_tiling(img, MIN_AREA, DEVICE, do_tiles=True):
    """
    타일링을 사용한 객체 검출기 실행 - 대형 이미지를 작은 타일로 나누어 처리
    
    Args:
        img: PIL 이미지 객체
        MIN_AREA: 최소 박스 면적 (이보다 작은 박스는 제거)
        DEVICE: GPU/CPU 디바이스
        do_tiles: 타일링 수행 여부
    
    Returns:
        (boxes, embs): 전체 이미지에서 검출된 바운딩 박스와 임베딩
    """
    W, H = img.size
    all_boxes = []
    all_embs = []

    # 전체 이미지에서 검출
    b, e = infer_boxes_and_embs(img, DEVICE)
    all_boxes.append(b)
    all_embs.append(e)

    # 타일링 수행
    if do_tiles:
        # 타일 시작 위치 계산
        xs = list(range(0, max(W - PATCH, 0) + 1, STRIDE))
        ys = list(range(0, max(H - PATCH, 0) + 1, STRIDE))

        # 마지막 타일이 이미지 끝에 닿도록 조정
        if W > PATCH and xs[-1] != W - PATCH:
            xs.append(W - PATCH)

        if H > PATCH and ys[-1] != H - PATCH:
            ys.append(H - PATCH)

        # 이미지가 패치보다 작은 경우
        if W <= PATCH:
            xs = [0]

        if H <= PATCH:
            ys = [0]

        # 각 타일에서 검출 수행
        for y0 in ys:
            for x0 in xs:
                tile = img.crop((x0, y0, x0 + PATCH, y0 + PATCH))
                tb, te = infer_boxes_and_embs(tile, DEVICE)
                if tb.shape[0] == 0:
                    continue

                # 타일 좌표를 전체 이미지 좌표로 변환
                px = np.stack([tb[:, 0] * PATCH + x0, tb[:, 1] * PATCH + y0,
                               tb[:, 2] * PATCH + x0, tb[:, 3] * PATCH + y0], axis=1)
                gb = px / np.array([W, H, W, H], np.float32)  # 정규화
                
                # 코어 영역에 있는 박스만 선택 (경계 효과 방지)
                cx = (tb[:, 0] + tb[:, 2]) * 0.5 * PATCH
                cy = (tb[:, 1] + tb[:, 3]) * 0.5 * PATCH
                in_core = (cx >= CORE) & (cx <= PATCH - CORE) & (cy >= CORE) & (cy <= PATCH - CORE)
                if in_core.any():
                    gb = gb[in_core]
                    te = te[in_core]
                    all_boxes.append(gb)
                    all_embs.append(te)

    if len(all_boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, 1280), np.float32)

    # 모든 결과를 하나로 합치기
    B = np.concatenate(all_boxes, 0)
    E = np.concatenate(all_embs, 0)
    
    # 최소 면적 필터링
    if MIN_AREA > 0.0 and len(B) > 0:
        areas = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        keep = np.where(areas >= MIN_AREA)[0]
        B = B[keep]
        E = E[keep]
    return B, E


def make_backbone(backbone_name: str, num_classes=1):
    """
    CNN 백본 모델 생성 - 바이너리 분류기를 위한 다양한 백본 아키텍처 지원
    
    Args:
        backbone_name: 백본 모델명 ("mobilenet_v3_small", "resnet18", "convnext_tiny")
        num_classes: 출력 클래스 수 (기본값: 1, 바이너리 분류)
    
    Returns:
        수정된 백본 모델 (마지막 레이어가 num_classes로 변경됨)
    
    Raises:
        ValueError: 지원하지 않는 백본 모델명인 경우
    """
    name = backbone_name.lower()
    if name == "mobilenet_v3_small":
        # MobileNetV3-Small 백본
        m = torchvision.models.mobilenet_v3_small(weights=None)
        in_feat = m.classifier[0].in_features
        # 커스텀 분류기: 128차원 은닉층 + Hardswish 활성화 + Dropout
        m.classifier = nn.Sequential(
            nn.Linear(in_feat, 128),
            nn.Hardswish(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        return m
    elif name == "resnet18":
        # ResNet18 백본
        m = torchvision.models.resnet18(weights=None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)  # 마지막 레이어만 교체
        return m
    elif name == "convnext_tiny":
        # ConvNeXt-Tiny 백본
        m = torchvision.models.convnext_tiny(weights=None)
        in_feat = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_feat, num_classes)  # 분류기 마지막 레이어 교체
        return m
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")