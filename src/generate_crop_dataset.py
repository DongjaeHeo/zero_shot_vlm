#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
크롭 데이터셋 생성 스크립트 - 바이너리 분류기 훈련을 위한 데이터 마이닝

이 스크립트는 다음과 같은 기능을 수행합니다:
1. OWLv2 모델을 사용하여 이미지에서 객체 검출 수행
2. 검출된 박스를 Ground Truth와 비교하여 3가지 카테고리로 분류:
   - TRUE (right.pt): IoU >= 0.50인 양성 샘플 (모든 샘플 보존, top-k 제한 없음)
   - HARD WRONG (hard_wrong.pt): 0.10 <= IoU < 0.50인 어려운 음성 샘플 (모든 샘플 보존)
   - OTHER WRONG (wrong.pt): IoU < 0.10이지만 이미지 프로토타입과 코사인 유사도 >= 0.80인 샘플
3. 이미지별 프로토타입 생성: 해당 이미지의 TRUE 임베딩들의 평균
4. 각 카테고리별로 임베딩, 박스, 메타데이터를 저장

핵심 특징:
- 사전 라벨 NMS 비활성화: 모든 제안을 보존하여 더 많은 데이터 확보
- 이미지별 프로토타입 기반 어려운 음성 샘플 선별
- 타일링을 통한 대형 이미지 처리
- 5-fold cross validation 지원

저장되는 필드:
- embeddings: L2 정규화된 임베딩
- boxes_xyxy_norm: 정규화된 바운딩 박스 좌표
- cosine_to_img_proto: 이미지 프로토타입과의 코사인 유사도 (해당 없으면 NaN)
- ious_to_best_gt: 가장 높은 IoU를 가진 GT와의 IoU 값
- meta: 이미지 경로, GT 정보, 제안 인덱스, 타일 여부, 이미지 크기, 소스 등
"""

import os
import re
import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from utils.infer_utils import *

# ========= 설정 =========
SRC_ROOT = Path("/home/dromii_shared/obstacle_subclass").resolve()  # 소스 데이터셋 경로
# SRC_ROOT = Path("../dataset/sample_data").resolve() # 샘플 데이터셋: class/{images,labels}
PILES_DIR = Path("../crop_dataset").resolve()  # 출력 파일: right.pt, hard_wrong.pt, wrong.pt
MODEL_NAME = "google/owlv2-large-patch14-ensemble"  # 사용할 OWLv2 모델명

# ========= 마이닝 임계값 =========
IOU_TRUE = 0.50  # TRUE (양성 샘플) 임계값
HARD_L = 0.10    # HARD WRONG 하한 (포함)
HARD_U = 0.50    # HARD WRONG 상한 (제외)
TAU_IMG = 0.80   # OTHER WRONG을 위한 이미지별 프로토타입 코사인 유사도 임계값

# ========= 사전 라벨 중복 제거 - 비활성화 (더 많은 데이터 확보를 위해) =========
# 명확성을 위해 스위치로 유지하지만 코드 경로는 제거됨
NMS_IOU = 0.0  # 0.0 => 비활성화 (사전 라벨 NMS 없음)

# ========= 타일링 파라미터 (이전과 동일) =========
PATCH, OVERLAP = 1008, 500  # 패치 크기와 겹침 크기
STRIDE, CORE = PATCH - OVERLAP, 250  # 스트라이드와 코어 영역 크기
TILE_EXCEPT = {"group_of_trees"}  # 타일링 제외 클래스 (대소문자 무관)

# # Repro
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)



\

def pick_device(arg_idx: str | None):
    """
    디바이스 선택 함수 - CUDA 디바이스 ID를 안전하게 파싱하여 torch.device 반환
    
    Args:
        arg_idx: 디바이스 ID 문자열 (예: "0", "1", "cuda:0" 등)
    
    Returns:
        torch.device: 선택된 디바이스
    """
    # 개행/CR/제어문자 등 불필요한 문자 제거
    s = re.sub(r"[^0-9]", "", arg_idx or "")
    req = int(s) if s else 0

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        num = torch.cuda.device_count()
        if 0 <= req < num:
            dev = f"cuda:{req}"
        else:
            dev = "cuda:0"  # 요청된 디바이스가 없으면 기본값 사용
        print(f"[INFO] device_count={num} -> using {dev}", flush=True)
        return torch.device(dev)
    print("[WARN] CUDA not available -> CPU", flush=True)
    return torch.device("cpu")

# ========= 유틸리티 함수들 =========
def ensure_dir(p: Path):
    """
    디렉토리 생성 함수 - 부모 디렉토리까지 포함하여 생성
    
    Args:
        p: 생성할 디렉토리 경로
    """
    p.mkdir(parents=True, exist_ok=True)


def l2norm_t(x: torch.Tensor, dim=-1, eps=1e-8):
    """
    L2 정규화 함수 - 텐서를 단위 벡터로 변환
    
    Args:
        x: 정규화할 텐서
        dim: 정규화할 차원
        eps: 0으로 나누기 방지를 위한 작은 값
    
    Returns:
        L2 정규화된 텐서
    """
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def boxes_cxcywh_to_xyxy_norm(a: np.ndarray) -> np.ndarray:
    """
    YOLO 형식 (center_x, center_y, width, height)을 xyxy 형식으로 변환
    
    Args:
        a: [N, 4] 형태의 바운딩 박스 배열 (cx, cy, w, h)
    
    Returns:
        [N, 4] 형태의 바운딩 박스 배열 (x1, y1, x2, y2)
    """
    c = a[:, 0]  # center_x
    d = a[:, 1]  # center_y
    w = a[:, 2]  # width
    h = a[:, 3]  # height
    return np.stack([c - w / 2, d - h / 2, c + w / 2, d + h / 2], axis=1).astype(np.float32)


def iou_xyxy(A, B):
    """
    두 바운딩 박스 배열 간의 IoU 매트릭스 계산
    
    Args:
        A: [N, 4] 형태의 첫 번째 바운딩 박스 배열
        B: [M, 4] 형태의 두 번째 바운딩 박스 배열
    
    Returns:
        [N, M] 형태의 IoU 매트릭스
    """
    if A.size == 0 or B.size == 0: 
        return np.zeros((A.shape[0], B.shape[0]), np.float32)
    
    a = A[:, None, :]  # [N, 1, 4]
    b = B[None, :, :]  # [1, M, 4]
    
    # 교집합 영역 계산
    ix1 = np.maximum(a[..., 0], b[..., 0])  # x1 좌표
    iy1 = np.maximum(a[..., 1], b[..., 1])  # y1 좌표
    ix2 = np.minimum(a[..., 2], b[..., 2])  # x2 좌표
    iy2 = np.minimum(a[..., 3], b[..., 3])  # y2 좌표
    
    iw = np.clip(ix2 - ix1, 0, None)  # 교집합 너비
    ih = np.clip(iy2 - iy1, 0, None)  # 교집합 높이
    inter = iw * ih  # 교집합 면적
    
    # 합집합 면적 계산
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (
                b[..., 3] - b[..., 1]) - inter + 1e-8
    return (inter / ua).astype(np.float32)


def read_gt_yolo(txt: Path) -> np.ndarray:
    """
    YOLO 형식의 라벨 파일을 읽어서 바운딩 박스 정보 반환
    
    Args:
        txt: YOLO 형식 라벨 파일 경로
    
    Returns:
        [N, 5] 형태의 배열 (class_id, cx, cy, w, h)
    """
    if not txt.exists(): 
        return np.zeros((0, 5), np.float32)
    
    out = []
    for ln in txt.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5: 
            continue
        try:
            cid = int(float(p[0]))  # 클래스 ID
            cx, cy, w, h = map(float, p[1:5])  # 바운딩 박스 좌표
            out.append((cid, cx, cy, w, h))
        except:
            pass
    return np.array(out, np.float32) if out else np.zeros((0, 5), np.float32)


def load_image(path: Path) -> Image.Image:
    """
    EXIF 정보를 고려한 이미지 로드 함수
    
    Args:
        path: 이미지 파일 경로
    
    Returns:
        RGB로 변환된 PIL Image 객체
    """
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


# ========= OWLv2 모델 관리 =========
# 전역 캐시 (모델 로드 시간 단축)
_PROC = None
_MODEL = None

def get_model_and_proc(DEVICE):
    """
    OWLv2 모델과 전처리기를 가져오는 함수 (캐시 사용)
    
    Args:
        DEVICE: 사용할 디바이스
    
    Returns:
        proc: OWLv2 전처리기
        model: OWLv2 모델
    """
    global _PROC, _MODEL
    if _PROC is None:
        _PROC = Owlv2Processor.from_pretrained(MODEL_NAME)
    if _MODEL is None:
        _MODEL = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return _PROC, _MODEL

@torch.no_grad()
def infer_boxes_and_embs(img: Image.Image, DEVICE) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지에서 바운딩 박스와 임베딩을 추출하는 추론 함수
    
    Args:
        img: 입력 이미지 (PIL Image)
        DEVICE: 사용할 디바이스
    
    Returns:
        boxes: [N, 4] 형태의 바운딩 박스 배열 (xyxy, 정규화됨)
        embs: [N, D] 형태의 임베딩 배열 (L2 정규화됨)
    """
    proc, model = get_model_and_proc(DEVICE)
    inp = proc(images=img, return_tensors="pt").to(DEVICE)
    
    # 이미지 임베딩 추출
    fmap = model.image_embedder(inp.pixel_values)[0]  # [1, h, w, D]
    _, h, w, D = fmap.shape
    feats = fmap.reshape(1, h * w, D)  # [1, N, D] - N = h*w
    
    # 바운딩 박스 예측
    pred = model.box_predictor(feats, feature_map=fmap)[0].detach().cpu().numpy()
    
    # 클래스 임베딩 추출
    _, cls = model.class_predictor(feats)
    
    # 바운딩 박스를 xyxy 형식으로 변환하고 정규화
    boxes = np.clip(boxes_cxcywh_to_xyxy_norm(pred), 0.0, 1.0)
    
    # 임베딩을 L2 정규화
    embs = l2norm_t(cls[0].float(), dim=-1).cpu().numpy().astype(np.float32)
    
    return boxes, embs



@torch.no_grad()
def run_detector_with_tiling(img: Image.Image, DEVICE, do_tiles: bool):
    """
    전체 이미지와 타일에서 검출 제안을 집계하는 함수 (타일 출처 플래그도 추적)
    
    Args:
        img: 입력 이미지
        DEVICE: 사용할 디바이스
        do_tiles: 타일링 수행 여부
    
    Returns:
        B: [N, 4] 바운딩 박스 배열 (xyxy, 정규화됨)
        E: [N, D] 임베딩 배열 (L2 정규화됨)
        F: [N] 타일 출처 플래그 배열 (True=타일에서, False=전체 이미지에서)
    """
    W, H = img.size
    all_boxes = []
    all_embs = []
    all_from_tile = []
    
    # ========= 전체 이미지에서 검출 =========
    b, e = infer_boxes_and_embs(img, DEVICE)
    all_boxes.append(b)
    all_embs.append(e)
    all_from_tile.append(np.zeros((b.shape[0],), dtype=bool))  # 전체 이미지에서 검출됨을 표시
    
    # ========= 타일링 처리 =========
    if do_tiles:
        # 타일 시작 위치 계산
        xs = list(range(0, max(W - PATCH, 0) + 1, STRIDE))
        ys = list(range(0, max(H - PATCH, 0) + 1, STRIDE))
        
        # 이미지 끝까지 완전히 커버하도록 마지막 타일 위치 조정
        if W > PATCH and xs[-1] != W - PATCH: 
            xs.append(W - PATCH)
        if H > PATCH and ys[-1] != H - PATCH: 
            ys.append(H - PATCH)
        if W <= PATCH: 
            xs = [0]  # 이미지가 패치보다 작으면 전체 이미지 사용
        if H <= PATCH: 
            ys = [0]
        
        # 각 타일에 대해 검출 수행
        for y0 in ys:
            for x0 in xs:
                tile = img.crop((x0, y0, x0 + PATCH, y0 + PATCH))
                tb, te = infer_boxes_and_embs(tile, DEVICE)
                if tb.shape[0] == 0: 
                    continue
                
                # ========= 타일 좌표를 전체 이미지 좌표로 변환 =========
                px = np.stack([tb[:, 0] * PATCH + x0, tb[:, 1] * PATCH + y0,
                               tb[:, 2] * PATCH + x0, tb[:, 3] * PATCH + y0], axis=1)
                gb = px / np.array([W, H, W, H], np.float32)  # 전체 이미지 기준으로 정규화
                
                # ========= 코어 영역 필터링 =========
                # 타일 경계 근처의 검출을 제거하여 중복 방지
                cx = (tb[:, 0] + tb[:, 2]) * 0.5 * PATCH  # 타일 내 중심 x
                cy = (tb[:, 1] + tb[:, 3]) * 0.5 * PATCH  # 타일 내 중심 y
                in_core = (cx >= CORE) & (cx <= PATCH - CORE) & (cy >= CORE) & (cy <= PATCH - CORE)
                
                if in_core.any():
                    gb = gb[in_core]
                    te = te[in_core]
                    all_boxes.append(gb)
                    all_embs.append(te)
                    all_from_tile.append(np.ones((gb.shape[0],), dtype=bool))  # 타일에서 검출됨을 표시
    
    # ========= 결과 집계 =========
    if len(all_boxes) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0, 768), np.float32), np.zeros((0,), dtype=bool))
    
    B = np.concatenate(all_boxes, 0)      # 모든 바운딩 박스 연결
    E = np.concatenate(all_embs, 0)       # 모든 임베딩 연결
    F = np.concatenate(all_from_tile, 0)  # 모든 타일 플래그 연결
    
    return B, E, F


# ========= 파일 헬퍼 함수들 =========
def init_pile_acc():
    """
    파일 누적기를 초기화하는 함수
    
    Returns:
        빈 파일 딕셔너리
    """
    return {
        "embeddings": [],           # 임베딩 리스트
        "boxes_xyxy_norm": [],      # 정규화된 바운딩 박스 리스트
        "cosine_to_img_proto": [],  # 이미지 프로토타입과의 코사인 유사도 리스트
        "ious_to_best_gt": [],      # 최고 IoU를 가진 GT와의 IoU 리스트
        "meta": []                  # 메타데이터 리스트
    }


def extend_pile(pile, emb, box, cos_img, iou, meta):
    """
    파일에 새로운 샘플을 추가하는 함수
    
    Args:
        pile: 파일 딕셔너리
        emb: 임베딩
        box: 바운딩 박스
        cos_img: 이미지 프로토타입과의 코사인 유사도 (None 가능)
        iou: 최고 IoU 값
        meta: 메타데이터
    """
    pile["embeddings"].append(emb.astype(np.float32))
    pile["boxes_xyxy_norm"].append(box.astype(np.float32))
    pile["cosine_to_img_proto"].append(float(cos_img) if cos_img is not None else float("nan"))
    pile["ious_to_best_gt"].append(float(iou))
    pile["meta"].append(meta)


def finalize_and_save_pile(pile, out_path: Path, cname: str):
    """
    파일을 최종화하고 저장하는 함수
    
    Args:
        pile: 파일 딕셔너리
        out_path: 저장할 파일 경로
        cname: 클래스 이름
    """
    if len(pile["embeddings"]) == 0:
        # 빈 파일인 경우 빈 객체 생성
        obj = {
            "class": cname,
            "embeddings": torch.zeros((0, 768), dtype=torch.float32),
            "boxes_xyxy_norm": np.zeros((0, 4), np.float32),
            "cosine_to_img_proto": np.zeros((0,), np.float32),
            "ious_to_best_gt": np.zeros((0,), np.float32),
            "meta": []
        }
        ensure_dir(out_path.parent)
        torch.save(obj, out_path)
        print(f"[save-empty] {out_path}")
        return
    
    # 데이터를 텐서/배열로 변환
    E = torch.from_numpy(np.stack(pile["embeddings"], 0))  # 임베딩 스택
    B = np.stack(pile["boxes_xyxy_norm"], 0)               # 바운딩 박스 스택
    C = np.array(pile["cosine_to_img_proto"], dtype=np.float32)  # 코사인 유사도 배열
    I = np.array(pile["ious_to_best_gt"], dtype=np.float32)      # IoU 배열
    
    obj = {
        "class": cname, 
        "embeddings": E, 
        "boxes_xyxy_norm": B,
        "cosine_to_img_proto": C, 
        "ious_to_best_gt": I, 
        "meta": pile["meta"]
    }
    
    ensure_dir(out_path.parent)
    torch.save(obj, out_path)
    print(f"[saved] {out_path}  (N={E.shape[0]})")

def get_img_path(cname):
    img_dir = SRC_ROOT / cname / "images"
    # img_dir = SRC_ROOT / "images"

    # gather ALL images (no split)
    all_imgs: List[Path] = []
    for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.tif", "*.tiff"):
        all_imgs += list(img_dir.glob(ext))
    all_imgs = sorted(all_imgs)
    return all_imgs


# ========= Mining per class =========
def mine_class(cname: str, all_imgs = None, DEVICE="0"):
    print(f"\n[mine] class={cname}")
    if DEVICE is None:
        DEVICE = pick_device("0")  # safe fallback
        
    if all_imgs is None:
        all_imgs = get_img_path(cname)
    else:
        all_imgs = {Path(p) for p in all_imgs}
        all_imgs = sorted(all_imgs)

    lbl_dir = SRC_ROOT / cname / "labels"
    do_tiles = (cname.lower() not in {s.lower() for s in TILE_EXCEPT})
    piles = {"right": init_pile_acc(), "hard_wrong": init_pile_acc(), "wrong": init_pile_acc()}

    if not all_imgs:
        print(f"[warn] no images for class {cname}")
        return piles

    for img_path in all_imgs:
        img = load_image(img_path)
        W, H = img.size
        gt = read_gt_yolo(lbl_dir / f"{img_path.stem}.txt")
        gt_xyxy = boxes_cxcywh_to_xyxy_norm(gt[:, 1:5]) if gt.shape[0] else np.zeros((0, 4), np.float32)

        boxes, embs, from_tile = run_detector_with_tiling(img, DEVICE, do_tiles=do_tiles)
        if len(embs) == 0:
            print(f"[{cname}/{img_path.name}] props=0")
            continue

        # ---- NO pre-label NMS (keep everything) ----

        # IoU to GT
        if gt_xyxy.shape[0] > 0:
            ious = iou_xyxy(boxes, gt_xyxy)  # [M, G]
            best_gt = ious.argmax(axis=1)  # [M]
            best_iou = ious.max(axis=1)  # [M]
        else:
            best_gt = np.full((len(boxes),), -1, dtype=np.int32)
            best_iou = np.zeros((len(boxes),), dtype=np.float32)

        # ------- TRUE (IoU >= 0.5): keep ALL -------
        mask_true = best_iou >= IOU_TRUE
        true_idx = np.where(mask_true)[0]

        # Build per-image prototype from TRUE embeddings (if any)
        if true_idx.size > 0:
            E_true = embs[true_idx]
            E_true = E_true / (np.linalg.norm(E_true, axis=1, keepdims=True) + 1e-12)
            img_proto = E_true.mean(axis=0)
            img_proto = img_proto / (np.linalg.norm(img_proto) + 1e-12)
        else:
            img_proto = None

        # ------- HARD WRONG (0.10 <= IoU < 0.50) -------
        mask_hard = (best_iou >= HARD_L) & (best_iou < HARD_U)
        hard_idx = np.where(mask_hard)[0]

        # ------- OTHER WRONG (cos >= 0.80 to per-image proto AND IoU < 0.10) -------
        if img_proto is not None:
            E_all = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
            cos_img = (E_all @ img_proto.reshape(-1, 1)).reshape(-1)
        else:
            cos_img = np.full((len(embs),), np.nan, dtype=np.float32)

        mask_not_true = best_iou < IOU_TRUE
        mask_easy_band = best_iou < HARD_L  # IoU < 0.10
        mask_other = mask_not_true & mask_easy_band
        if img_proto is not None:
            mask_other = mask_other & (cos_img >= TAU_IMG)
        other_idx = np.where(mask_other)[0]

        # -------- Save TRUE --------
        for j in true_idx:
            meta = {"image_path": str(img_path), "split": "all", "class": cname,
                    "matched_gt_idx": int(best_gt[j]),
                    "gt_box_xyxy_norm": gt_xyxy[int(best_gt[j])].astype(np.float32).tolist() if gt_xyxy.size else None,
                    "proposal_idx": int(j), "from_tile": bool(from_tile[j]),
                    "W": int(W), "H": int(H), "source": "harvest"}
            extend_pile(piles["right"], embs[j], boxes[j],
                        cos_img[j] if img_proto is not None else None,
                        float(best_iou[j]), meta)

        # -------- Save HARD WRONG --------
        for j in hard_idx:
            meta = {"image_path": str(img_path), "split": "all", "class": cname,
                    "matched_gt_idx": int(best_gt[j]),
                    "gt_box_xyxy_norm": gt_xyxy[int(best_gt[j])].astype(np.float32).tolist() if gt_xyxy.size else None,
                    "proposal_idx": int(j), "from_tile": bool(from_tile[j]),
                    "W": int(W), "H": int(H), "source": "harvest"}

            extend_pile(piles["hard_wrong"], embs[j], boxes[j],
                        cos_img[j] if img_proto is not None else None,
                        float(best_iou[j]), meta)

        # -------- Save OTHER WRONG --------
        for j in other_idx:
            meta = {"image_path": str(img_path), "split": "all", "class": cname,
                    "matched_gt_idx": -1,
                    "gt_box_xyxy_norm": None,
                    "proposal_idx": int(j), "from_tile": bool(from_tile[j]),
                    "W": int(W), "H": int(H), "source": "harvest_cos_img"}
            extend_pile(piles["wrong"], embs[j], boxes[j],
                        cos_img[j] if img_proto is not None else None,
                        float(best_iou[j]), meta)

        print(f"[{cname}/{img_path.name}] props={len(embs)}  "
              f"true+={len(true_idx)}  hard_wrong+={len(hard_idx)}  other_wrong+={len(other_idx)}")

    return piles


# ========= Orchestrate =========
def generate_crop_dataset(splits, DEVICE):
    ensure_dir(PILES_DIR)

    # discover classes by folder names that contain images/
    classes = [p.name for p in sorted(SRC_ROOT.iterdir()) if (p / "images").exists()]
    print("Classes:", classes)

    for cname in classes:
        for fold_idx, split in splits.items():
            all_images = split[class_label][0]

            piles = mine_class(cname, all_images, DEVICE)
            base = PILES_DIR / cname
            finalize_and_save_pile(piles["right"], base / f"{seed}_fold_{fold_idx}" / "right.pt", cname)
            finalize_and_save_pile(piles["hard_wrong"], base / f"{seed}_fold_{fold_idx}" / "hard_wrong.pt", cname)
            finalize_and_save_pile(piles["wrong"], base / f"{seed}_fold_{fold_idx}" / "wrong.pt", cname)

    print("\nDone. Per-class piles saved under 'crop_dataset/<class>/{right,hard_wrong,wrong}.pt'.")


# if __name__ == "__main__":
#     OUT_IDX = Path("../index").resolve()

#     device_id = sys.argv[1] if sys.argv[1] != "-" else "0"
#     class_label = sys.argv[2] if sys.argv[2] != "-" else None
#     seed = int(sys.argv[3]) if sys.argv[3] != "-" else 42

#     # device_id = "0"
#     # class_label = "Headstone_without_Mound"
#     # seed = 42

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     os.environ["CUDA_VISIBLE_DEVICES"] = device_id
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     packs = load_pos_packs(OUT_IDX)
#     splits = create_5fold_splits(packs, seed)

#     if class_label is not None:
#         print("Class:", class_label)

#         for fold_idx, split in splits.items():
#             all_images = split[class_label][0]

#             piles = mine_class(class_label, all_images, DEVICE)
#             base = PILES_DIR / class_label
#             finalize_and_save_pile(piles["right"], base / f"{seed}_fold_{fold_idx}" / "right.pt", class_label)
#             finalize_and_save_pile(piles["hard_wrong"], base / f"{seed}_fold_{fold_idx}" / "hard_wrong.pt", class_label)
#             finalize_and_save_pile(piles["wrong"], base / f"{seed}_fold_{fold_idx}" / "wrong.pt", class_label)

#         print(f"\nDone. {class_label} crop dataset saved under 'crop_dataset/{class_label}/right,hard_wrong,wrong.pt'.")
#     else:
#         generate_crop_dataset(splits, DEVICE)


if __name__ == "__main__":
    OUT_IDX = Path("../index").resolve()

    # read CLI
    gpu_arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else "0"
    class_label = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else 42

    # never set CUDA_VISIBLE_DEVICES here; pick a concrete device instead
    DEVICE = pick_device(gpu_arg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    packs = load_pos_packs(OUT_IDX)
    splits = create_5fold_splits(packs, seed)

    if class_label is not None:
        print("Class:", class_label)
        for fold_idx, split in splits.items():
            all_images = split[class_label][0]
            piles = mine_class(class_label, all_images, DEVICE)  # pass a torch.device
            base = PILES_DIR / class_label
            finalize_and_save_pile(piles["right"],      base / f"{seed}_fold_{fold_idx}" / "right.pt",       class_label)
            finalize_and_save_pile(piles["hard_wrong"], base / f"{seed}_fold_{fold_idx}" / "hard_wrong.pt",  class_label)
            finalize_and_save_pile(piles["wrong"],      base / f"{seed}_fold_{fold_idx}" / "wrong.pt",       class_label)
        print(f"\nDone. {class_label} crop dataset saved under 'crop_dataset/{class_label}/right,hard_wrong,wrong.pt'.")
    else:
        generate_crop_dataset(splits, DEVICE)
