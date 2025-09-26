"""
추론 유틸리티 모듈 - Zero-shot 객체 검출 시스템의 핵심 유틸리티 함수들

이 모듈은 다음과 같은 기능을 제공합니다:
1. 임베딩 정규화 및 코사인 유사도 계산
2. 바운딩 박스 좌표 변환 (cxcywh <-> xyxy)
3. IoU 계산 및 NMS (Non-Maximum Suppression)
4. YOLO 형식 Ground Truth 읽기
5. 이미지 로드 및 전처리
6. 클래스별 프로토타입 생성 (top-K, clustering, averaging)
7. 5-fold cross validation 데이터 분할
8. CNN 바이너리 분류기 로드
9. 크롭 이미지 텐서 변환

핵심 특징:
- 다양한 프로토타입 생성 방식 지원 (raw, cluster, average)
- 클래스별/그룹별 프로토타입 관리
- 효율적인 크롭 이미지 처리
- GPU/CPU 호환 텐서 연산
"""

import os, csv, random, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable, Union
import numpy as np
import argparse
from collections import defaultdict

from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

from pycparser.ply.yacc import default_lr
from sklearn.cluster import KMeans
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from utils.model_utils import *


def l2norm_t(x: torch.Tensor, dim=-1, eps=1e-8):
    """
    L2 정규화 함수 - 텐서를 단위 벡터로 정규화
    
    Args:
        x: 정규화할 텐서
        dim: 정규화할 차원 (기본값: -1, 마지막 차원)
        eps: 0으로 나누는 것을 방지하는 작은 값
    
    Returns:
        L2 정규화된 텐서 (각 벡터의 크기가 1이 됨)
    """
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def cosine_mat(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    코사인 유사도 행렬 계산 - L2 정규화된 임베딩들 간의 코사인 유사도
    
    Args:
        E: 이미지 임베딩 행렬 [N, D] (N개 패치, D차원)
        P: 프로토타입 임베딩 행렬 [C, D] (C개 클래스, D차원)
        두 행렬 모두 L2 정규화되어 있어야 함
    
    Returns:
        코사인 유사도 행렬 [N, C] - 각 이미지 패치와 각 클래스 프로토타입 간의 유사도
    """
    return (E @ P.T).astype(np.float32)


def boxes_cxcywh_to_xyxy_norm(a: np.ndarray) -> np.ndarray:
    """
    바운딩 박스 좌표 변환 - (center_x, center_y, width, height) -> (x1, y1, x2, y2)
    
    Args:
        a: 정규화된 cxcywh 형식 바운딩 박스 [N, 4]
           - c: 중심 x 좌표 (0~1)
           - d: 중심 y 좌표 (0~1) 
           - w: 박스 너비 (0~1)
           - h: 박스 높이 (0~1)
    
    Returns:
        정규화된 xyxy 형식 바운딩 박스 [N, 4] (x1, y1, x2, y2)
    """
    c = a[:, 0]  # center_x
    d = a[:, 1]  # center_y
    w = a[:, 2]  # width
    h = a[:, 3]  # height
    return np.stack([c - w / 2, d - h / 2, c + w / 2, d + h / 2], axis=1).astype(np.float32)

def boxes_cxcywh_to_xyxy_norm_tensor(a: torch.Tensor) -> torch.Tensor:
    """
    바운딩 박스 좌표 변환 (PyTorch 텐서 버전) - (cx, cy, w, h) -> (x1, y1, x2, y2)
    
    Args:
        a: 정규화된 cxcywh 형식 바운딩 박스 텐서 [N, 4] (GPU/CPU)
    
    Returns:
        정규화된 xyxy 형식 바운딩 박스 텐서 [N, 4] (x1, y1, x2, y2)
    """
    c = a[:, 0]  # center_x
    d = a[:, 1]  # center_y
    w = a[:, 2]  # width
    h = a[:, 3]  # height

    x1 = c - w / 2  # left
    y1 = d - h / 2  # top
    x2 = c + w / 2  # right
    y2 = d + h / 2  # bottom

    return torch.stack([x1, y1, x2, y2], dim=1)


def iou_xyxy(A, B):
    """
    IoU (Intersection over Union) 계산 - xyxy 형식 바운딩 박스들 간의 겹침 비율
    
    Args:
        A: 첫 번째 바운딩 박스 배열 [N, 4] (x1, y1, x2, y2)
        B: 두 번째 바운딩 박스 배열 [M, 4] (x1, y1, x2, y2)
    
    Returns:
        IoU 행렬 [N, M] - A의 각 박스와 B의 각 박스 간의 IoU 값
    """
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), np.float32)

    # 브로드캐스팅을 위한 차원 확장
    a = A[:, None, :]  # [N, 1, 4]
    b = B[None, :, :]  # [1, M, 4]
    
    # 교집합 영역 계산
    ix1 = np.maximum(a[..., 0], b[..., 0])  # 교집합 왼쪽
    iy1 = np.maximum(a[..., 1], b[..., 1])  # 교집합 위쪽
    ix2 = np.minimum(a[..., 2], b[..., 2])  # 교집합 오른쪽
    iy2 = np.minimum(a[..., 3], b[..., 3])  # 교집합 아래쪽

    # 교집합 너비/높이 (음수면 겹치지 않음)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih  # 교집합 면적

    # 합집합 면적 계산
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (
                b[..., 3] - b[..., 1]) - inter + 1e-8
    return (inter / ua).astype(np.float32)


def nms_xyxy(boxes, scores, iou_thr):
    """
    NMS (Non-Maximum Suppression) - 겹치는 바운딩 박스들 중 최고 점수만 유지
    
    Args:
        boxes: 바운딩 박스 배열 [N, 4] (x1, y1, x2, y2)
        scores: 각 박스의 신뢰도 점수 [N]
        iou_thr: IoU 임계값 (이 값보다 높게 겹치면 제거)
    
    Returns:
        유지할 박스들의 인덱스 리스트
    """
    if len(boxes) == 0:
        return []

    # 점수 순으로 내림차순 정렬
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        # 가장 높은 점수의 박스 선택
        i = order[0]
        keep.append(i)
        rest = order[1:]
        
        if rest.size == 0:
            break

        # 선택된 박스와 나머지 박스들 간의 IoU 계산
        ious = iou_xyxy(boxes[i:i + 1], boxes[rest]).reshape(-1)
        
        # IoU 임계값보다 낮은 박스들만 유지 (겹치지 않는 박스들)
        rest = rest[ious <= iou_thr]
        order = rest
        
    return keep


def read_gt_yolo(txt: Path) -> np.ndarray:
    """
    YOLO 형식 Ground Truth 파일 읽기 - 클래스 ID와 정규화된 바운딩 박스 좌표
    
    Args:
        txt: YOLO 형식 라벨 파일 경로 (각 줄: class_id cx cy w h)
    
    Returns:
        Ground Truth 배열 [N, 5] - (class_id, cx, cy, w, h)
        파일이 없거나 빈 경우 빈 배열 반환
    """
    if not txt.exists():
        return np.zeros((0, 5), np.float32)

    out = []
    for ln in txt.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:  # 최소 5개 값 필요 (class_id, cx, cy, w, h)
            continue

        try:
            cid = int(float(p[0]))  # 클래스 ID
            cx, cy, w, h = map(float, p[1:5])  # 정규화된 바운딩 박스 좌표
            out.append((cid, cx, cy, w, h))
        except:
            pass  # 잘못된 형식의 줄은 무시
            
    return np.array(out, np.float32) if out else np.zeros((0, 5), np.float32)


def load_image(path: Path) -> Image.Image:
    """
    이미지 로드 및 EXIF 정보 처리 - 회전 정보를 자동으로 적용하여 올바른 방향으로 로드
    
    Args:
        path: 이미지 파일 경로
    
    Returns:
        RGB 형식으로 변환되고 EXIF 회전 정보가 적용된 PIL Image 객체
    """
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


def canonical(s: str) -> str:
    """
    문자열 정규화 - 앞뒤 공백 제거 및 소문자 변환
    
    Args:
        s: 정규화할 문자열
    
    Returns:
        정규화된 문자열 (소문자, 공백 제거)
    """
    return s.strip().lower()


# ========= Packs / Splits / Prototypes =========
def load_pos_packs(idx_dir: Path):
    """
    양성 샘플 팩 로드 - 클래스별로 저장된 임베딩과 메타데이터 로드
    
    Args:
        idx_dir: pos_*.pt 파일들이 저장된 디렉토리
    
    Returns:
        클래스명을 키로 하는 딕셔너리 {class_name: pack_data}
        각 팩은 embeddings, metadata 등을 포함
    
    Raises:
        RuntimeError: pos_*.pt 파일이 없는 경우
    """
    packs = {}
    for p in sorted(idx_dir.glob("pos_*.pt")):
        obj = torch.load(p, map_location="cpu")
        cname = obj.get("class", p.stem.replace("pos_", ""))
        packs[cname] = obj

    if not packs:
        raise RuntimeError(f"No pos_*.pt found in {idx_dir}")

    return packs


def split_80_20(imgs: List[str]) -> Tuple[set, set]:
    """
    80/20 데이터 분할 - 이미지 리스트를 훈련(80%)과 테스트(20%)로 무작위 분할
    
    Args:
        imgs: 분할할 이미지 경로 리스트
    
    Returns:
        (train_set, test_set): 훈련용과 테스트용 이미지 경로 세트
    """
    shuffled = list(imgs)
    random.shuffle(shuffled)  # 무작위 섞기
    n = len(shuffled)
    n_train = int(round(n * 0.80))  # 80%를 훈련용으로
    return set(shuffled[:n_train]), set(shuffled[n_train:])


def get_consensus(args, class_emb_dicts):
    """
    프로토타입 생성 - 클래스별 임베딩들로부터 다양한 방식으로 프로토타입 생성
    
    Args:
        args: 프로토타입 생성 방식 설정 (proto, num_clusters)
        class_emb_dicts: 클래스별 임베딩 딕셔너리 {class_name: embeddings_tensor}
    
    Returns:
        클래스별 프로토타입 딕셔너리 {class_name: prototype}
    
    프로토타입 생성 방식:
    - topk_raw: 모든 임베딩을 L2 정규화하여 그대로 사용
    - topk_cluster: K-means 클러스터링 후 각 클러스터의 중심점 사용
    - topk_avg: 모든 임베딩의 평균을 L2 정규화하여 단일 프로토타입 생성
    """
    protos = {}
    for cname, E in class_emb_dicts.items():
        if args.proto == "topk_raw":
            # 모든 임베딩을 L2 정규화하여 그대로 사용 (다중 프로토타입)
            e = l2norm_t(E, dim=-1).cpu().numpy().astype(np.float32)
            protos[cname] = e

        elif args.proto == "topk_cluster":
            # K-means 클러스터링을 통한 다중 프로토타입 생성
            num_clusters = args.num_clusters
            E_np = E.cpu().numpy().astype(np.float32)

            if len(E_np) <= num_clusters:
                num_clusters = len(E_np)

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(E_np)
            labels = kmeans.fit_predict(E_np)

            centroids_mean = []
            for i in range(num_clusters):
                cluster_members = E_np[labels == i]
                if len(cluster_members) == 0:
                    continue
                else:
                    cluster_avg = cluster_members.mean(axis=0)
                centroids_mean.append(cluster_avg)

            protos[cname] = np.stack(centroids_mean, axis=0)

        elif args.proto == "topk_avg":
            # 모든 임베딩의 평균을 L2 정규화하여 단일 프로토타입 생성
            e = l2norm_t(E, dim=-1).cpu().numpy().astype(np.float32)
            p = e.mean(axis=0)
            p /= (np.linalg.norm(p) + 1e-12)
            protos[cname] = p.astype(np.float32)

    return protos


def build_class_protos_topk(
        packs: Dict[str, dict],
        K: int,
        splits: Dict[str, Tuple[set, set]],
        group_name: Dict[str, List[str]],
        args: Optional[argparse.Namespace] = None,
) -> [Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    클래스별 프로토타입 생성 (Top-K 방식) - 각 Ground Truth당 상위 K개 IoU 임베딩 사용
    
    각 클래스에 대해 훈련 이미지(80%)에서만 프로토타입을 생성합니다.
    각 GT당 IoU가 높은 상위 K개 임베딩을 선택하여 클래스별/그룹별 프로토타입을 만듭니다.
    
    Args:
        packs: 클래스별 임베딩 팩 딕셔너리 {class_name: {embeddings, metadata}}
        K: 각 GT당 선택할 최대 임베딩 수
        splits: 클래스별 훈련/테스트 분할 {class_name: (train_set, test_set)}
        group_name: 클래스 그룹 정의 {group_name: [class_names]}
        args: 프로토타입 생성 방식 설정
    
    Returns:
        (protos_group, protos_class): 그룹별/클래스별 프로토타입 딕셔너리
    """
    group_emb_tot = defaultdict(list)  # 그룹별 임베딩 수집
    class_emb_tot = {}  # 클래스별 임베딩 수집
    
    for cname, obj in packs.items():
        embs: torch.Tensor = obj["embeddings"].float()
        meta: List[dict] = obj.get("metadata", [])
        
        # 임베딩이나 메타데이터가 없는 경우 처리
        if embs.numel() == 0 or not meta:
            # 폴백: 모든 임베딩의 평균 사용 (L2->mean->L2)
            e = l2norm_t(embs, dim=-1).cpu().numpy().astype(np.float32)
            if e.size == 0:
                print(f"[proto/K={K}] {cname}: no embeddings; skipping.")
                continue
            p = e.mean(axis=0)
            p /= (np.linalg.norm(p) + 1e-12)
            print(f"[proto/K={K}] {cname}: fallback mean over ALL ({len(e)})")
            continue

        train_set, _ = splits[cname]
        # 훈련 이미지에 속하는 임베딩 인덱스 찾기
        idx_train = [i for i, m in enumerate(meta) if m.get("image_path") in train_set]
        if not idx_train:
            # 폴백: 모든 임베딩의 평균 사용
            e = l2norm_t(embs, dim=-1).cpu().numpy().astype(np.float32)
            p = e.mean(axis=0)
            p /= (np.linalg.norm(p) + 1e-12)
            print(f"[proto/K={K}] {cname}: no train embs; fallback ALL ({len(e)})")
            continue

        # (이미지경로, GT_ID)별로 임베딩과 IoU 수집
        per_group: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
        for i in idx_train:
            m = meta[i]
            imgp = m.get("image_path", "")
            gid = int(m.get("gt_id", -1))
            iou = float(m.get("iou", 0.0))
            per_group.setdefault((imgp, gid), []).append((i, iou))

        # 각 GT당 상위 K개 IoU 임베딩 선택
        selected: List[int] = []
        for key, lst in per_group.items():
            lst.sort(key=lambda t: t[1], reverse=True)  # IoU 내림차순 정렬
            take = lst[:max(1, K)]  # 최소 1개, 최대 K개 선택
            selected.extend([i for (i, _) in take])

        if not selected:
            # 최종 폴백: 훈련 서브셋에서 선택
            selected = idx_train[:max(1, K)]

        # 그룹별/클래스별 임베딩 수집
        gname = next(group for group, members in group_name.items() if cname in members)
        group_emb_tot[gname].append(embs.index_select(0, torch.tensor(selected, dtype=torch.long)))
        class_emb_tot[cname] = embs.index_select(0, torch.tensor(selected, dtype=torch.long))

    # 그룹별 임베딩들을 연결하여 하나의 텐서로 만들기
    new_group_emb_tot = {}
    for tmp_gname, emb_list in group_emb_tot.items():
        new_group_emb_tot[tmp_gname] = torch.cat(emb_list, dim=0)
    
    # 그룹별/클래스별 프로토타입 생성
    protos_group = get_consensus(args, new_group_emb_tot)
    protos_class = get_consensus(args, class_emb_tot)

    print(
        f"[proto/K={K}] {cname}: train_imgs={len(train_set)}  groups={len(per_group)}  used_embs={len(selected)}")

    return protos_group, protos_class


def split_per_class_80_20(packs: Dict[str, dict]) -> Dict[str, Tuple[set, set]]:
    """
    클래스별 80/20 데이터 분할 - 각 클래스의 이미지들을 독립적으로 80/20으로 분할
    
    Args:
        packs: 클래스별 임베딩 팩 딕셔너리 {class_name: {embeddings, metadata}}
    
    Returns:
        클래스별 훈련/테스트 분할 딕셔너리 {class_name: (train_set, test_set)}
    """
    splits = {}
    for cname, obj in packs.items():
        meta: List[dict] = obj.get("metadata", [])
        images = sorted({m["image_path"] for m in meta})  # 중복 제거 후 정렬
        train, test = split_80_20(images)
        splits[cname] = (train, test)
        print(f"[split] {cname}: {len(train)}/{len(images)} train (80%)")
    return splits


def create_5fold_splits(packs: Dict[str, dict], seed: int = 42) -> Dict[int, Dict[str, Tuple[set, set]]]:
    """
    5-fold Cross Validation 데이터 분할 - 각 클래스의 이미지들을 5개 폴드로 분할
    
    Args:
        packs: 클래스별 임베딩 팩 딕셔너리 {class_name: {embeddings, metadata}}
        seed: 무작위 시드 (재현 가능한 분할을 위해)
    
    Returns:
        5-fold 분할 딕셔너리 {fold_id: {class_name: (train_set, val_set)}}
    """
    class_imgs = {}
    for cname, obj in packs.items():
        meta: List[dict] = obj.get("metadata", [])
        images = sorted({m["image_path"] for m in meta})
        class_imgs[cname] = images

    splits = {i: {} for i in range(5)}

    for cname, images in class_imgs.items():
        n = len(images)
        fold_size = n // 5  # 각 폴드 크기
        shuffled = list(images)
        random.Random(seed).shuffle(shuffled)  # 재현 가능한 셔플

        for i in range(5):
            val_start = i * fold_size
            val_end = n if i == 4 else (i + 1) * fold_size  # 마지막 폴드는 나머지 모두 포함
            val = set(shuffled[val_start:val_end])
            train = set(shuffled[:val_start] + shuffled[val_end:])
            splits[i][cname] = (train, val)
            print(f"[split] fold {i}, {cname}: {len(train)}/{n} train, {len(val)}/{n} val")

    return splits


def resolve_groups(packs, class_groups_raw):
    """
    클래스 그룹 해석 - 원시 그룹 정의를 실제 클래스명과 매칭하여 해석
    
    Args:
        packs: 클래스별 임베딩 팩 딕셔너리 (실제 존재하는 클래스들)
        class_groups_raw: 원시 그룹 정의 딕셔너리 {group_name: [class_names]}
    
    Returns:
        해석된 그룹 딕셔너리 {group_name: [resolved_class_names]}
    """
    def canonical(s: str) -> str:
        return s.strip().lower()

    # 실제 존재하는 클래스들의 정규화된 이름 매핑
    name_map = {canonical(k): k for k in packs.keys()}
    groups_resolved = {}
    print("\n[debug] Available classes:", ", ".join(sorted(packs.keys())))
    
    for gname, members in class_groups_raw.items():
        ok, miss = [], []
        for m in members:
            cm = canonical(m)
            if cm in name_map:
                ok.append(name_map[cm])
            else:
                # 별칭 매핑 (예: "tree" -> "single_tree")
                aliases = {"tree": "single_tree", "group-of-trees": "group_of_trees"}
                if cm in aliases and aliases[cm] in name_map:
                    ok.append(name_map[aliases[cm]])
                else:
                    miss.append(m)
        if ok: 
            groups_resolved[gname] = sorted(list(set(ok)))
        # print(f"[groups] {gname}: members={ok if ok else []}  missing={miss if miss else []}")
    return groups_resolved


def load_cnn_for_class(cnn_dir: Path, cname: str, device):
    """
    클래스별 CNN 바이너리 분류기 로드 - 모델, 메타데이터, 전처리 설정 로드
    
    Args:
        cnn_dir: CNN 모델이 저장된 디렉토리
        cname: 클래스명
        device: GPU/CPU 디바이스
    
    Returns:
        (model, transform, crop_size, context): 로드된 모델과 전처리 설정
    
    Raises:
        FileNotFoundError: 모델 파일이 없는 경우
    """
    ckpt_path = cnn_dir/"model.pt"
    meta_path = cnn_dir/"best.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing CNN for {cname}: {ckpt_path}")
    
    # 체크포인트와 메타데이터 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # 모델 설정 추출 (체크포인트 우선, 메타데이터 폴백)
    backbone = ckpt.get("backbone", meta.get("backbone", "mobilenet_v3_small"))
    crop_size = int(ckpt.get("crop_size", meta.get("crop",{}).get("size", 224)))
    context   = float(ckpt.get("context", meta.get("crop",{}).get("context", 0.10)))
    norm_imnet= bool(ckpt.get("normalize_imagenet", meta.get("normalize_imagenet", True)))

    # 모델 생성 및 가중치 로드
    from utils.model_utils import make_backbone
    model = make_backbone(backbone, 1).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # 평가용 전처리 파이프라인 구성
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) if norm_imnet \
           else transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        norm,
    ])
    print(f"[CNN] {cname}: backbone={backbone} crop={crop_size} ctx={context:.2f} norm_imnet={norm_imnet}")
    return model, tf, crop_size, context

def norm_to_abs(xyxy_norm, w, h):
    """
    정규화된 좌표를 절대 좌표로 변환 - (0~1) -> (0~width, 0~height)
    
    Args:
        xyxy_norm: 정규화된 바운딩 박스 좌표 [x1, y1, x2, y2] (0~1)
        w: 이미지 너비
        h: 이미지 높이
    
    Returns:
        절대 좌표 바운딩 박스 [x1, y1, x2, y2] (픽셀 단위)
    """
    x1 = float(xyxy_norm[0])*w; y1 = float(xyxy_norm[1])*h
    x2 = float(xyxy_norm[2])*w; y2 = float(xyxy_norm[3])*h
    x1,x2 = min(x1,x2), max(x1,x2)  # 좌우 순서 보장
    y1,y2 = min(y1,y2), max(y1,y2)  # 상하 순서 보장
    return [x1,y1,x2,y2]

def expand_and_clamp_box(xyxy, w, h, ctx=0.10):
    """
    바운딩 박스 확장 및 경계 제한 - 컨텍스트를 추가하고 이미지 경계 내로 제한
    
    Args:
        xyxy: 원본 바운딩 박스 좌표 [x1, y1, x2, y2]
        w: 이미지 너비
        h: 이미지 높이
        ctx: 컨텍스트 확장 비율 (기본값: 0.10, 10% 확장)
    
    Returns:
        확장되고 경계가 제한된 바운딩 박스 [x1, y1, x2, y2]
    """
    x1,y1,x2,y2 = xyxy
    bw = max(1.0, x2-x1); bh = max(1.0, y2-y1)  # 박스 너비/높이 (최소 1픽셀)
    cx = (x1+x2)/2.0; cy=(y1+y2)/2.0  # 중심점
    
    # 컨텍스트를 고려한 확장된 크기
    bw2 = bw * (1.0 + 2*ctx); bh2 = bh * (1.0 + 2*ctx)
    
    # 확장된 박스의 좌표 (이미지 경계 내로 제한)
    x1n = max(0.0, cx - bw2/2.0)
    y1n = max(0.0, cy - bh2/2.0)
    x2n = min(float(w-1), cx + bw2/2.0)
    y2n = min(float(h-1), cy + bh2/2.0)
    return [x1n,y1n,x2n,y2n]


# original version
def crops_to_tensor(img: Image.Image, boxes_xyxy_norm: np.ndarray, tf: transforms.Compose, context: float) -> torch.Tensor:
    """
    바운딩 박스들을 크롭하여 배치 텐서로 변환 - 컨텍스트를 포함한 크롭 이미지 생성
    
    각 바운딩 박스를 컨텍스트와 함께 크롭하고, 전처리를 적용하여 배치 텐서로 변환합니다.
    너무 작은 박스의 경우 검은색 패딩을 추가합니다.
    
    Args:
        img: 원본 PIL 이미지
        boxes_xyxy_norm: 정규화된 바운딩 박스 배열 [N, 4] (x1, y1, x2, y2)
        tf: 이미지 전처리 변환 (resize, to_tensor, normalize)
        context: 컨텍스트 확장 비율
    
    Returns:
        크롭된 이미지들의 배치 텐서 [N, 3, H, W]
    """
    w,h = img.size
    crops = []
    for b in boxes_xyxy_norm:
        x1,y1,x2,y2 = norm_to_abs(b, w, h)  # 정규화된 좌표를 절대 좌표로 변환
        
        # 너무 작은 박스 처리 (최소 4픽셀)
        if min(x2-x1, y2-y1) < 4:
            tile = Image.new("RGB", (max(4, int(x2-x1)), max(4, int(y2-y1))), (0,0,0))
            crops.append(tf(tile))
            continue
            
        # 컨텍스트를 포함한 확장된 박스로 크롭
        x1n,y1n,x2n,y2n = expand_and_clamp_box([x1,y1,x2,y2], w, h, ctx=context)
        crop = img.crop((x1n,y1n,x2n,y2n))
        crops.append(tf(crop))
        
    if len(crops)==0:
        return torch.zeros((0,3,224,224), dtype=torch.float32)
    return torch.stack(crops, dim=0)

# def crops_to_tensor(img: Image.Image, boxes_xyxy_norm: np.ndarray, tf: transforms.Compose, DEVICE, context: float) -> torch.Tensor:
#     w, h = img.size
#     # PIL -> Tensor
#     img_tensor = F.to_tensor(img).unsqueeze(0) .to(DEVICE) # [1,3,H,W]
#
#     boxes_abs = []
#     for b in boxes_xyxy_norm:
#         x1,y1,x2,y2 = norm_to_abs(b, w, h)
#         x1n,y1n,x2n,y2n = expand_and_clamp_box([x1,y1,x2,y2], w, h, ctx=context)
#         boxes_abs.append([x1n, y1n, x2n, y2n])
#     if len(boxes_abs) == 0:
#         return torch.zeros((0,3,224,224), dtype=torch.float32, device=DEVICE)
#
#     boxes = torch.tensor(boxes_abs, dtype=torch.float32, device=DEVICE)
#     batch_idx = torch.zeros(len(boxes), dtype=torch.int32, device=DEVICE)
#     crops_out = []
#     chunk_size = 8192  # GPU 메모리에 맞게 조정
#     with torch.no_grad():
#         for i in range(0, len(boxes), chunk_size):
#             chunk = boxes[i:i + chunk_size]
#             crops_chunk = torchvision.ops.roi_align(
#                 img_tensor, [chunk],
#                 output_size=(224, 224),
#                 spatial_scale=1.0,
#                 aligned=True
#             ).cpu()
#             crops_out.append(crops_chunk)
#     crops = torch.cat(crops_out, dim=0)
#
#     return crops