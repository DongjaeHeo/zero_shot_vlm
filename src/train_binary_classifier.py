#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
바이너리 분류기 훈련 스크립트 - Zero-shot 객체 검출 시스템의 CNN 바이너리 분류기 훈련

이 스크립트는 다음과 같은 기능을 수행합니다:
1. 크롭 데이터셋에서 양성/음성 샘플 로드
2. 80/20 훈련/검증 분할 및 계층적 샘플링
3. 다양한 CNN 백본 모델 지원 (MobileNetV3, ResNet18, ConvNeXt)
4. Focal Loss를 사용한 불균형 데이터 처리
5. Early stopping 및 다양한 정규화 기법
6. 5-fold cross validation 지원

핵심 특징:
- ROI 기반 크롭 데이터셋 처리
- 하드 네거티브 마이닝
- 데이터 증강 (RandAugment, ColorJitter, Rotation)
- 다양한 백본 아키텍처 지원
- 효율적인 메모리 사용 (per-worker 이미지 캐싱)
"""

# =========================
# NOTEBOOK-READY FULL CODE
# =========================

# ==== CONFIG (EDIT ME) ====
from pathlib import Path
import torch
import os, re, sys

from utils.infer_utils import *

# ==== 데이터 경로 설정 ====
PILES_DIR = Path("../crop_dataset")               # 크롭 데이터셋 경로: piles_v2/<class>/{right,hard_wrong,wrong}.pt
OUT_DIR   = Path("../cnn_models_roi_on_fly")   # 모델 저장 경로 (클래스별)
CLASSES   = None                           # None => 자동 발견; 또는 ["house","factory"] 같은 리스트

# ==== 샘플링 설정 ====
# 에포크당 NEG:POS 비율 및 하드/이지 네거티브 비율
NEG_PER_POS   = 3        # 각 양성 샘플당 3개의 음성 샘플 샘플링
HARD_FRACTION = 2/3      # 음성 샘플 중 2/3은 hard_wrong, 1/3은 other wrong

# ==== 분할 및 평가 설정 ====
SPLIT_SEED     = 42  # 재현 가능한 분할을 위한 시드
VAL_THRESHOLD_GRID = [round(float(x), 2) for x in torch.linspace(0.05, 0.95, 19)]  # 검증 임계값 그리드

# ==== Early stopping 및 에포크 설정 ====
EARLY_STOP_PATIENCE  = 6      # 조기 종료 인내심 (에포크)
EARLY_STOP_MIN_DELTA = 1e-4   # 최소 개선 임계값
MAX_EPOCHS           = 50     # 최대 에포크 수

# ==== 이미지/크롭 파라미터 ====
CROP_SIZE     = 224    # 크롭 리사이즈 크기 (HxW)
BOX_CONTEXT   = 0.10   # 크롭 전 박스 확장 비율 (각 방향 10%)
SKIP_TINY_PX  = 4      # 정규화 해제 후 더 짧은 변이 이 값보다 작으면 샘플 건너뛰기

# ==== 모델/최적화 설정 ====
BACKBONE      = "mobilenet_v3_small"     # 백본 모델: "resnet18", "convnext_tiny"도 가능
FINETUNE_MODE = "partial"                # 파인튜닝 모드: "frozen" | "partial" (마지막 단계) | "full"
BATCH_SIZE    = 64                       # 배치 크기
LR            = 3e-4                     # 학습률 (partial/frozen용; full 파인튜닝은 1e-4 사용)
WEIGHT_DECAY  = 1e-4                     # 가중치 감쇠
LABEL_SMOOTH  = 0.0                      # 라벨 스무딩 (선택사항, focal loss용으로는 0 유지)
USE_IMAGENET_NORMALIZE = True            # ImageNet 정규화 사용 여부
USE_RANDAUG   = True                     # 단순 RandAugment로 견고성 향상

# ==== 데이터로더 설정 ====
NUM_WORKERS   = 4                        # 워커 수
PIN_MEMORY    = True                     # 메모리 고정

# ==== IMPORTS ====
import json, random, math
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


def pick_device(arg_idx: str | None):
    """
    GPU 디바이스 선택 - CLI 인자에서 GPU 인덱스 추출하여 적절한 디바이스 반환
    
    Args:
        arg_idx: CLI 인자 문자열 (GPU 인덱스 포함 가능)
    
    Returns:
        선택된 PyTorch 디바이스
    """
    # CLI 인자에 섞일 수 있는 개행/CR/제어문자 제거 후 숫자만 추출
    s = re.sub(r"[^0-9]", "", arg_idx or "")
    req = int(s) if s else 0

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        num = torch.cuda.device_count()
        if 0 <= req < num:
            dev = f"cuda:{req}"
        else:
            dev = "cuda:0"
        print(f"[INFO] device_count={num} -> using {dev}", flush=True)
        return torch.device(dev)
    print("[WARN] CUDA not available -> CPU", flush=True)
    return torch.device("cpu")

# ==== UTILS ====
def set_seed(seed: int):
    """
    재현 가능한 결과를 위한 시드 설정
    
    Args:
        seed: 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p: Path):
    """
    디렉토리 생성 (부모 디렉토리 포함)
    
    Args:
        p: 생성할 디렉토리 경로
    """
    p.mkdir(parents=True, exist_ok=True)

def load_pile(path: Path):
    """
    크롭 데이터 팩 로드
    
    Args:
        path: 팩 파일 경로
    
    Returns:
        로드된 팩 데이터
    
    Raises:
        FileNotFoundError: 파일이 없는 경우
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing pile: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)

def load_or_empty(path: Path, cname: str):
    """
    크롭 데이터 팩 로드 또는 빈 구조 반환
    
    Args:
        path: 팩 파일 경로
        cname: 클래스명
    
    Returns:
        로드된 팩 데이터 또는 빈 구조
    """
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return {
        "class": cname,
        "embeddings": torch.zeros((0, 768), dtype=torch.float32),
        "boxes_xyxy_norm": np.zeros((0,4), np.float32),
        "cosine_to_img_proto": np.zeros((0,), np.float32),
        "ious_to_best_gt": np.zeros((0,), np.float32),
        "meta": [],
    }

def discover_classes(piles_dir: Path):
    """
    크롭 데이터셋에서 클래스 자동 발견
    
    Args:
        piles_dir: 크롭 데이터셋 디렉토리
    
    Returns:
        발견된 클래스명 리스트 (정렬됨)
    """
    classes = []
    for p in sorted(piles_dir.iterdir()):
        if not p.is_dir(): continue
        # right.pt와 (hard_wrong.pt 또는 wrong.pt)가 모두 있는 클래스만 선택
        if (p/"right.pt").exists() and ((p/"hard_wrong.pt").exists() or (p/"wrong.pt").exists()):
            classes.append(p.name)
    return sorted(set(classes))

def gather_all_images(*objs):
    imgs = set()
    for obj in objs:
        for m in obj.get("meta", []):
            p = m.get("image_path","")
            if p: imgs.add(p)
    return sorted(imgs)

def stratified_images_80_20(obj_right, obj_hard, obj_other, seed: int):
    # positives per-image IDs
    pos_images = set()
    for m in obj_right["meta"]:
        ip = m.get("image_path","")
        if ip: pos_images.add(ip)
    all_images = set(gather_all_images(obj_right, obj_hard, obj_other))
    neg_only_images = [p for p in all_images if p not in pos_images]

    rng = np.random.RandomState(seed)
    pos_images = list(pos_images); rng.shuffle(pos_images)
    neg_only_images = list(neg_only_images); rng.shuffle(neg_only_images)

    def _split(lst):
        n = len(lst)
        n_tr = int(round(n * 0.80))
        # return set(lst[:n_tr]), set(lst[n_tr:])
        return set(lst[:]), set()

    pos_tr, pos_val = _split(pos_images)
    neg_tr, neg_val = _split(neg_only_images)

    train_imgs = set(list(pos_tr) + list(neg_tr))
    val_imgs   = set(list(pos_val) + list(neg_val))

    # ensure at least 1 positive in val if any positives exist
    if len(pos_images) > 0 and len(pos_val) == 0 and len(pos_tr) > 0:
        move_one = next(iter(pos_tr))
        train_imgs.discard(move_one)
        val_imgs.add(move_one)
    return train_imgs, val_imgs

def build_index(obj, allowed_images: set, label: int):
    idxs = []
    metas = obj.get("meta", [])
    boxes = obj.get("boxes_xyxy_norm", np.zeros((0,4), np.float32))
    for i, m in enumerate(metas):
        ip = m.get("image_path","")
        if ip and ip in allowed_images:
            idxs.append((ip, boxes[i], label))
    return idxs

def norm_to_abs(xyxy_norm, w, h):
    x1 = float(xyxy_norm[0])*w; y1 = float(xyxy_norm[1])*h
    x2 = float(xyxy_norm[2])*w; y2 = float(xyxy_norm[3])*h
    x1,x2 = min(x1,x2), max(x1,x2)
    y1,y2 = min(y1,y2), max(y1,y2)
    return [x1,y1,x2,y2]

def expand_and_clamp_box(xyxy, w, h, ctx=BOX_CONTEXT):
    x1,y1,x2,y2 = xyxy
    bw = max(1.0, x2-x1); bh = max(1.0, y2-y1)
    cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
    bw2 = bw * (1.0 + 2*ctx)
    bh2 = bh * (1.0 + 2*ctx)
    x1n = max(0.0, cx - bw2/2.0)
    y1n = max(0.0, cy - bh2/2.0)
    x2n = min(float(w-1), cx + bw2/2.0)
    y2n = min(float(h-1), cy + bh2/2.0)
    return [x1n,y1n,x2n,y2n]

# ==== DATASET (on-the-fly crops + per-worker image cache) ====
class ROICropsDataset(Dataset):
    _thread_local = threading.local()  # per worker cache

    def __init__(self, index_tuples, split="train"):
        """
        index_tuples: list of (image_path:str, xyxy_norm:np.array(4), label:int)
        """
        self.items = index_tuples
        if USE_IMAGENET_NORMALIZE:
            norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        else:
            norm = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

        aug = []
        if split == "train":
            aug = []
            if USE_RANDAUG:
                try:
                    from torchvision.transforms import RandAugment
                    aug.append(RandAugment(num_ops=2, magnitude=7))
                except Exception:
                    pass
            aug += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0),  # <-- hue off
                transforms.RandomRotation(10),
            ]
        else:
            aug = []

        self.tf = transforms.Compose(aug + [
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) if USE_IMAGENET_NORMALIZE
            else transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def _get_img_cache(self):
        if not hasattr(self._thread_local, "cache"):
            self._thread_local.cache = {}  # {path: PIL.Image}
        return self._thread_local.cache

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ipath, box_norm, label = self.items[i]
        cache = self._get_img_cache()

        img = cache.get(ipath)
        if img is None:
            img = ImageOps.exif_transpose(Image.open(ipath).convert("RGB"))
            cache[ipath] = img
        w, h = img.size
        x1,y1,x2,y2 = norm_to_abs(box_norm, w, h)
        if min(x2-x1, y2-y1) < SKIP_TINY_PX:
            # tiny box -> black tile to keep batching simple
            tile = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (0,0,0))
            x = self.tf(tile)
            return x, torch.tensor(float(label), dtype=torch.float32)

        x1n,y1n,x2n,y2n = expand_and_clamp_box([x1,y1,x2,y2], w, h, ctx=BOX_CONTEXT)
        crop = img.crop((x1n,y1n,x2n,y2n))
        x = self.tf(crop)
        return x, torch.tensor(float(label), dtype=torch.float32)

# Per-epoch NEG:POS with HARD_FRACTION
def make_epoch_index(pos_list, hard_list, other_list):
    n_pos = len(pos_list)
    if n_pos == 0:
        raise RuntimeError("No positives in training split.")
    total_neg_needed = max(1, int(round(NEG_PER_POS * n_pos)))
    n_hard = min(len(hard_list), int(round(HARD_FRACTION * total_neg_needed)))
    n_other = max(0, total_neg_needed - n_hard)

    rng = np.random.default_rng()
    idx = []
    idx.extend(pos_list)  # keep all positives

    if len(hard_list) > 0:
        sel_h = rng.choice(len(hard_list), size=n_hard, replace=(len(hard_list) < n_hard))
        idx.extend([hard_list[j] for j in sel_h])

    if len(other_list) > 0 and n_other > 0:
        sel_o = rng.choice(len(other_list), size=n_other, replace=(len(other_list) < n_other))
        idx.extend([other_list[j] for j in sel_o])

    rng.shuffle(idx)
    return idx

# ==== LOSS & METRICS ====
class FocalLoss(nn.Module):
    """
    Focal Loss - 불균형 데이터셋을 위한 손실 함수
    
    Focal Loss는 쉬운 샘플의 기여도를 줄이고 어려운 샘플에 더 집중하여
    클래스 불균형 문제를 해결합니다.
    
    Args:
        alpha: 클래스 가중치 (기본값: 0.25)
        gamma: 포커싱 파라미터 (기본값: 2.0)
        label_smoothing: 라벨 스무딩 비율 (기본값: 0.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Focal Loss 계산
        
        Args:
            logits: 모델 출력 로짓 [N]
            targets: 타겟 라벨 [N]
        
        Returns:
            Focal Loss 값
        """
        # 선택적 라벨 스무딩
        if self.label_smoothing > 0.0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p*targets + (1-p)*(1-targets)
        loss = ce * ((1 - p_t).clamp_min(1e-6) ** self.gamma)
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = alpha_t * loss
        return loss.mean()

def f1_from_counts(tp, fp, fn, eps=1e-9):
    prec = tp / (tp + fp + eps) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn + eps) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec / (prec + rec + eps) if (prec+rec)>0 else 0.0
    return float(prec), float(rec), float(f1)

@torch.no_grad()
def eval_on_loader(model, loader, device, threshold=0.5):
    model.eval()
    ys = []; ps = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ps.append(prob)
        ys.append(yb.detach().cpu().numpy())
    if not ys:
        return {"tp":0,"fp":0,"fn":0,"tn":0,"precision":0.0,"recall":0.0,"f1":0.0}
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = (p >= threshold).astype(np.float32)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    prec, rec, f1 = f1_from_counts(tp, fp, fn)
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":prec,"recall":rec,"f1":f1}

@torch.no_grad()
def best_threshold_from_loader(model, loader, device, grid=VAL_THRESHOLD_GRID):
    model.eval()
    ys = []; ps = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ps.append(prob); ys.append(yb.detach().cpu().numpy())
    if not ys:
        return 0.5, {"tp":0,"fp":0,"fn":0,"tn":0,"precision":0.0,"recall":0.0,"f1":0.0}
    y = np.concatenate(ys); p = np.concatenate(ps)

    best = (0.5, -1.0, None)
    for t in grid:
        pred = (p >= t).astype(np.float32)
        tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
        prec, rec, f1 = f1_from_counts(tp, fp, fn)
        if f1 > best[1]:
            best = (float(t), f1, {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":prec,"recall":rec,"f1":f1})
    return best[0], best[2]

# ==== MODEL ====
def make_backbone(backbone_name=BACKBONE, num_classes=1, pretrained=True):
    name = backbone_name.lower()
    try:
        if name == "mobilenet_v3_small":
            weights = "IMAGENET1K_V1" if pretrained else None
            m = torchvision.models.mobilenet_v3_small(weights=weights)
            in_feat = m.classifier[0].in_features
            m.classifier = nn.Sequential(
                nn.Linear(in_feat, 128),
                nn.Hardswish(),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )
            return m
        elif name == "resnet18":
            weights = "IMAGENET1K_V1" if pretrained else None
            m = torchvision.models.resnet18(weights=weights)
            in_feat = m.fc.in_features
            m.fc = nn.Linear(in_feat, num_classes)
            return m
        elif name == "convnext_tiny":
            weights = "IMAGENET1K_V1" if pretrained else None
            m = torchvision.models.convnext_tiny(weights=weights)
            in_feat = m.classifier[2].in_features
            m.classifier[2] = nn.Linear(in_feat, num_classes)
            return m
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    except Exception:
        # Fallback if weights download not available
        return make_backbone(backbone_name, num_classes, pretrained=False)

def set_finetune_mode(model, mode="partial"):
    mode = mode.lower()
    for p in model.parameters():
        p.requires_grad = (mode == "full")

    if isinstance(model, torchvision.models.MobileNetV3):
        # stages: features[0..N-1]
        if mode == "frozen":
            for p in model.parameters(): p.requires_grad = False
            for p in model.classifier.parameters(): p.requires_grad = True
        elif mode == "partial":
            # unfreeze last 2 blocks + classifier
            for p in model.parameters(): p.requires_grad = False
            for blk in model.features[-2:]:
                for p in blk.parameters(): p.requires_grad = True
            for p in model.classifier.parameters(): p.requires_grad = True

    elif isinstance(model, torchvision.models.ResNet):
        if mode == "frozen":
            for p in model.parameters(): p.requires_grad = False
            for p in model.fc.parameters(): p.requires_grad = True
        elif mode == "partial":
            for p in model.parameters(): p.requires_grad = False
            for p in model.layer4.parameters(): p.requires_grad = True
            for p in model.fc.parameters(): p.requires_grad = True

    elif isinstance(model, torchvision.models.ConvNeXt):
        if mode == "frozen":
            for p in model.parameters(): p.requires_grad = False
            for p in model.classifier.parameters(): p.requires_grad = True
        elif mode == "partial":
            for p in model.parameters(): p.requires_grad = False
            for p in model.features[-1].parameters(): p.requires_grad = True
            for p in model.classifier.parameters(): p.requires_grad = True

    return model

# ==== TRAIN ONE CLASS ====
def train_one_class(cname, right, hard, other, train_imgs, val_imgs, DEVICE):
    # Build index lists
    pos_train = build_index(right, train_imgs, 1)
    hard_train= build_index(hard,  train_imgs, 0)
    othr_train= build_index(other, train_imgs, 0)

    pos_val = build_index(right, val_imgs, 1)
    hard_val= build_index(hard,  val_imgs, 0)
    othr_val= build_index(other, val_imgs, 0)
    val_all = pos_val + hard_val + othr_val

    print(f"[{cname}] SPLIT  Train: Pos={len(pos_train)}  Hard={len(hard_train)}  Other={len(othr_train)}   |   "
          f"Val: Pos={len(pos_val)}  Hard={len(hard_val)}  Other={len(othr_val)}")

    if len(pos_val)==0 or len(val_all)==0 or len(pos_train)==0:
        return None, {"status":"insufficient data",
                      "counts":{"train_pos":len(pos_train),"train_hard":len(hard_train),"train_other":len(othr_train),
                                "val_pos":len(pos_val),"val_hard":len(hard_val),"val_other":len(othr_val)}}

    # Model, optim, loss
    model = make_backbone(BACKBONE, num_classes=1, pretrained=True).to(DEVICE)
    model = set_finetune_mode(model, FINETUNE_MODE)

    # lr hint
    lr_use = LR if FINETUNE_MODE in ("frozen","partial") else (1e-4 if LR > 1e-4 else LR)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_use, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=LABEL_SMOOTH)

    # Early stopping
    best = {"f1": -1.0, "epoch": -1, "metrics": None, "state_dict": None, "threshold": 0.5}
    no_improve = 0

    # Fixed val loader
    val_ds = ROICropsDataset(val_all, split="val")
    val_loader = DataLoader(val_ds, batch_size=max(32, BATCH_SIZE), shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)

    for ep in range(1, MAX_EPOCHS+1):
        model.train()

        # Per-epoch negative sampling
        epoch_idx = make_epoch_index(pos_train, hard_train, othr_train)
        tr_ds = ROICropsDataset(epoch_idx, split="train")
        eff_bs = max(8, min(BATCH_SIZE, len(tr_ds)))
        tr_loader = DataLoader(tr_ds, batch_size=eff_bs, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)

        running = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / max(1, len(tr_loader.dataset))

        # Threshold sweep on val
        t_best, metrics_at_t = best_threshold_from_loader(model, val_loader, DEVICE, grid=VAL_THRESHOLD_GRID)

        if metrics_at_t["f1"] > best["f1"] + EARLY_STOP_MIN_DELTA:
            best.update(f1=metrics_at_t["f1"], epoch=ep, metrics=metrics_at_t, threshold=t_best,
                        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"[{cname}] Early stop at epoch {ep}.")
                break

        print(f"[{cname}] ep {ep:02d}  loss {train_loss:.4f}  F1 {metrics_at_t['f1']:.4f}  "
              f"P {metrics_at_t['precision']:.3f}  R {metrics_at_t['recall']:.3f}  thr {t_best:.2f}")

    if best["state_dict"] is None:
        return None, {"status":"no improvement"}

    return best, {"status":"ok"}

# ==== DRIVER ====
def train_class_binary_classifier(cname, DEVICE, inter_path):
    # set_seed(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # classes = CLASSES if CLASSES is not None else discover_classes(PILES_DIR)
    # print("Classes:", classes)

    all_rows = []
    # for cname in classes:
    r_path = PILES_DIR/cname/inter_path/"right.pt"
    h_path = PILES_DIR/cname/inter_path/"hard_wrong.pt"
    o_path = PILES_DIR/cname/inter_path/"wrong.pt"

    if not r_path.exists():
        print(f"[skip] {cname}: missing right.pt.")
        return

    right = load_pile(r_path)
    hard  = load_or_empty(h_path, cname)
    other = load_or_empty(o_path, cname)

    total_true  = len(right["meta"])
    total_hard  = len(hard["meta"])
    total_other = len(other["meta"])
    print(f"\n[{cname}] TOTAL  True={total_true}  HardWrong={total_hard}  OtherWrong={total_other}")

    # by-image split
    train_imgs, val_imgs = stratified_images_80_20(right, hard, other, seed=SPLIT_SEED)

    best, info = train_one_class(cname, right, hard, other, train_imgs, val_imgs, DEVICE)

    row = {"class": cname,
           "neg_per_pos": NEG_PER_POS, "hard_fraction": HARD_FRACTION,
           "status": info.get("status",""),
           "split_seed": SPLIT_SEED, "crop_size": CROP_SIZE, "context": BOX_CONTEXT,
           "backbone": BACKBONE, "finetune_mode": FINETUNE_MODE,
           "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY}

    save_dir = OUT_DIR/cname/inter_path
    save_dir.mkdir(parents=True, exist_ok=True)

    if best is not None and best["state_dict"] is not None:
        model = make_backbone(BACKBONE, 1, pretrained=False)
        model = set_finetune_mode(model, "full")  # doesn’t matter for loading
        model.load_state_dict(best["state_dict"], strict=True)
        torch.save({"state_dict": best["state_dict"],
                    "backbone": BACKBONE,
                    "finetune_mode": FINETUNE_MODE,
                    "crop_size": CROP_SIZE,
                    "context": BOX_CONTEXT,
                    "normalize_imagenet": USE_IMAGENET_NORMALIZE},
                   save_dir/"model.pt")

        meta = {
            "class": cname,
            "best_f1": best["f1"],
            "best_epoch": best["epoch"],
            "metrics": best["metrics"],
            "threshold": best["threshold"],
            "split_seed": SPLIT_SEED,
            "neg_per_pos": NEG_PER_POS,
            "hard_fraction": HARD_FRACTION,
            "backbone": BACKBONE,
            "finetune_mode": FINETUNE_MODE,
            "train_cfg": {"batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                          "early_stopping": {"patience": EARLY_STOP_PATIENCE, "min_delta": EARLY_STOP_MIN_DELTA}},
            "crop": {"size": CROP_SIZE, "context": BOX_CONTEXT, "skip_tiny_px": SKIP_TINY_PX},
        }
        (save_dir/"best.json").write_text(json.dumps(meta, indent=2))

        row.update({"best_f1": best["f1"], "best_epoch": best["epoch"],
                    "precision": best["metrics"]["precision"], "recall": best["metrics"]["recall"],
                    "threshold": best["threshold"]})
        print(f"[{cname}] Saved best -> {save_dir/'model.pt'}  (thr={best['threshold']:.2f})")
    else:
        row.update({"best_f1": -1, "best_epoch": -1, "precision": 0.0, "recall": 0.0, "threshold": 0.5})

    # all_rows.append(row)

    # Save CSV of results
    results_df = pd.DataFrame([row])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = save_dir/f"cnn_results_{ts}_{cname}_{inter_path}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results CSV -> {results_csv}")
    # return results_df

if __name__ == "__main__":
    OUT_IDX = Path("../index").resolve()

    device_arg  = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else "0"
    class_label = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    seed        = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else 42

    set_seed(seed)

    # ❌ os.environ["CUDA_VISIBLE_DEVICES"] = device_id  -- 삭제!
    DEVICE = pick_device(device_arg)  # ✅ 안전한 torch.device 생성

    packs = load_pos_packs(OUT_IDX)
    splits = create_5fold_splits(packs, seed)

    if class_label is not None:
        print("Class:", class_label)
        for fold_idx, _ in splits.items():
            inter_path = f"{seed}_fold_{fold_idx}"
            _ = train_class_binary_classifier(class_label, DEVICE, inter_path)




