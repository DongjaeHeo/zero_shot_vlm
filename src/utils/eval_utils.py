#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
평가 유틸리티 모듈 - Zero-shot 객체 검출 시스템의 성능 평가 및 시각화

이 모듈은 다음과 같은 기능을 제공합니다:
1. 객체 검출 성능 메트릭 계산 (Precision, Recall, F1, AP@0.5, Coverage@0.5, FPPI)
2. 예측과 Ground Truth 간의 매칭 및 점수 계산
3. AP@0.5 계산 (연속 보간법 사용)
4. CSV 형태로 결과 저장
5. 검출 결과 시각화 (바운딩 박스, 점수, GT 표시)
6. 이미지 리사이징 및 패딩 처리

핵심 특징:
- STRICT/RELAXED 평가 모드 지원
- 클래스별/그룹별 성능 측정
- 5-fold cross validation 지원
- 다양한 시각화 옵션
- 효율적인 메트릭 계산
"""

import os, csv, random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Union
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# ========= Metrics helpers =========
def mask_scores(S: np.ndarray, allowed_ix: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    점수 마스킹 - 허용된 클래스 인덱스만 고려하여 최대 점수와 예측 레이블 계산
    
    Args:
        S: 점수 행렬 [N, C] (N개 예측, C개 클래스)
        allowed_ix: 허용된 클래스 인덱스 리스트
    
    Returns:
        (s_max, pl): 최대 점수 배열 [N], 예측 레이블 배열 [N]
        허용되지 않은 클래스는 -1e9로 마스킹
    """
    if not allowed_ix:
        return np.full((S.shape[0],), -1e9, dtype=np.float32), np.full((S.shape[0],), -1, dtype=np.int32)

    S_masked = S.copy()
    disallowed = np.ones(S.shape[1], dtype=bool)
    disallowed[allowed_ix] = False  # 허용된 클래스만 False
    S_masked[:, disallowed] = -1e9  # 허용되지 않은 클래스는 매우 낮은 점수로 마스킹
    s_max = S_masked.max(axis=1)  # 각 예측의 최대 점수
    pl = S_masked.argmax(axis=1)  # 각 예측의 최대 점수 클래스 인덱스
    return s_max, pl

def match_and_score(pb, ps, gt_boxes, iou_thr):
    """
    예측과 Ground Truth 매칭 및 성능 점수 계산 - IoU 기반 매칭으로 TP/FP/FN 계산
    
    Args:
        pb: 예측 바운딩 박스 배열 [N, 4]
        ps: 예측 점수 배열 [N]
        gt_boxes: Ground Truth 바운딩 박스 배열 [M, 4]
        iou_thr: IoU 임계값 (이 값 이상이면 매칭으로 간주)
    
    Returns:
        (tp, fp, fn, covered): True Positive, False Positive, False Negative, 커버된 GT 수
    """
    if len(pb)==0 and len(gt_boxes)==0: return 0,0,0,0
    if len(pb)==0: return 0,0,len(gt_boxes),0  # 모든 GT가 FN
    if len(gt_boxes)==0: return 0,len(pb),0,0  # 모든 예측이 FP
    
    # 점수 순으로 내림차순 정렬 (높은 점수부터 매칭)
    order=ps.argsort()[::-1]
    pb=pb[order]

    from utils.infer_utils import iou_xyxy
    ious=iou_xyxy(pb,gt_boxes)  # [N, M] IoU 행렬
    matched=set(); tp=fp=0

    for p in range(len(pb)):
        g=int(np.argmax(ious[p]))  # 가장 높은 IoU를 가진 GT
        if ious[p,g]>=iou_thr and g not in matched:
            matched.add(g); tp+=1  # 매칭 성공 (TP)
        else:
            fp+=1  # 매칭 실패 (FP)
    fn=len(gt_boxes)-len(matched); covered=len(matched)  # 매칭되지 않은 GT는 FN
    return tp,fp,fn,covered

def init_res_row():
    """
    결과 행 초기화 - 성능 메트릭을 저장할 딕셔너리 생성
    
    Returns:
        초기화된 메트릭 딕셔너리
    """
    return {"TP":0,"FP":0,"FN":0,"covered":0,"total_gt":0,"images":0,"fp_count":0}

def metrics_from_row(R):
    """
    결과 행에서 성능 메트릭 계산 - TP/FP/FN으로부터 Precision, Recall, F1 등 계산
    
    Args:
        R: 결과 행 딕셔너리 (TP, FP, FN, covered, total_gt, images, fp_count 포함)
    
    Returns:
        (coverage, fppi, precision, recall, f1): 계산된 성능 메트릭들
    """
    TP,FP,FN = R["TP"],R["FP"],R["FN"]
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0  # Precision = TP / (TP + FP)
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0  # Recall = TP / (TP + FN)
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0  # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    cov  = R["covered"]/R["total_gt"] if R["total_gt"]>0 else 0.0  # Coverage = covered GT / total GT
    fppi = R["fp_count"]/R["images"] if R["images"]>0 else 0.0  # FPPI = false positives per image
    return cov, fppi, prec, rec, f1

def ensure_dirs(base_viz: Path, base_metrics: Path):
    """
    결과 저장 디렉토리 생성 - 시각화와 메트릭 결과를 위한 디렉토리 구조 생성
    
    Args:
        base_viz: 시각화 결과 저장 기본 디렉토리
        base_metrics: 메트릭 결과 저장 기본 디렉토리
    
    생성되는 디렉토리 구조:
    - base_viz/STRICT/, base_viz/RELAXED/
    - base_metrics/STRICT/, base_metrics/RELAXED/
    """
    base_viz.mkdir(parents=True, exist_ok=True)
    (base_viz/"STRICT").mkdir(parents=True, exist_ok=True)
    (base_viz/"RELAXED").mkdir(parents=True, exist_ok=True)
    base_metrics.mkdir(parents=True, exist_ok=True)
    (base_metrics/"STRICT").mkdir(parents=True, exist_ok=True)
    (base_metrics/"RELAXED").mkdir(parents=True, exist_ok=True)

# ----- AP@0.5 helpers -----
def compute_ap50(preds: List[Tuple[str, float, np.ndarray]],
                 gts_per_image: Dict[str, np.ndarray],
                 iou_thr: float = 0.3) -> float:
    """
    AP@0.5 계산 - 연속 보간법을 사용한 Average Precision 계산
    
    Args:
        preds: 예측 리스트 [(img_id, score, box[4]), ...] - 박스는 [0,1] 정규화된 xyxy 형식
        gts_per_image: 이미지별 GT 딕셔너리 {img_id: array[G,4]} - 박스는 [0,1] 정규화된 xyxy 형식
        iou_thr: IoU 임계값 (기본값: 0.3)
    
    Returns:
        주어진 IoU 임계값에서의 Average Precision 값
    """
    if not preds:
        return 0.0
        
    # 점수 순으로 내림차순 정렬
    preds_sorted = sorted(preds, key=lambda t: t[1], reverse=True)
    tp = np.zeros(len(preds_sorted), dtype=np.float32)
    fp = np.zeros(len(preds_sorted), dtype=np.float32)

    # 이미지별 매칭된 GT 플래그
    matched: Dict[str, set] = {}

    total_gts = sum(len(gts_per_image.get(img_id, [])) for img_id in gts_per_image)
    if total_gts == 0:
        return 0.0

    # 각 예측에 대해 TP/FP 결정
    for i, (img_id, score, box) in enumerate(preds_sorted):
        gts = gts_per_image.get(img_id, np.zeros((0,4), np.float32))
        if gts.shape[0] == 0:
            fp[i] = 1.0  # GT가 없으면 FP
            continue

        from utils.infer_utils import iou_xyxy
        ious = iou_xyxy(box[None, :], gts).reshape(-1)  # [G] - 예측 박스와 모든 GT 간의 IoU
        g = int(np.argmax(ious))  # 가장 높은 IoU를 가진 GT
        if ious[g] >= iou_thr:
            used = matched.setdefault(img_id, set())
            if g not in used:
                tp[i] = 1.0  # 매칭 성공 (TP)
                used.add(g)
            else:
                fp[i] = 1.0  # 이미 매칭된 GT (FP)
        else:
            fp[i] = 1.0  # IoU 임계값 미달 (FP)

    # Precision-Recall 곡선 계산
    cum_tp = np.cumsum(tp)  # 누적 TP
    cum_fp = np.cumsum(fp)  # 누적 FP
    recall = cum_tp / (total_gts + 1e-12)  # Recall = TP / (TP + FN)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)  # Precision = TP / (TP + FP)

    # AP 계산: Precision envelope을 Recall에 대해 적분
    # Precision을 비증가 함수로 만들기 (monotonic decreasing)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    
    # Recall이 증가하는 구간에서의 적분
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def write_csv_tot(
        kind,
        row_dict, ap50_dict,
        g_row_dict, g_ap50_dict,
        g_post_row_dict, g_post_ap50_dict,
        class_names, groups_name,
        args):
    """
    CSV 결과 파일 작성 - 클래스별/그룹별 성능 메트릭을 CSV 형태로 저장
    
    Args:
        kind: 평가 종류 ("STRICT" 또는 "RELAXED")
        row_dict: 클래스별 결과 행 딕셔너리
        ap50_dict: 클래스별 AP@0.5 딕셔너리
        g_row_dict: 그룹별 결과 행 딕셔너리
        g_ap50_dict: 그룹별 AP@0.5 딕셔너리
        g_post_row_dict: 그룹별 후처리 결과 행 딕셔너리
        g_post_ap50_dict: 그룹별 후처리 AP@0.5 딕셔너리
        class_names: 클래스명 리스트
        groups_name: 그룹명 리스트
        args: 설정 인수 (K, proto, fold_idx, tau 등)
    """
    base_metrics = args.metric_dir / f"k{args.K}"

    # 바이너리 분류기 사용 여부에 따른 파일명 접미사 결정
    add_classifier = getattr(args, "binary_classifier", False)
    suffix = f"_classifier_{args.thr_use}.csv" if add_classifier else ".csv"

    # 프로토타입 방식에 따른 출력 파일 경로 설정
    if args.proto == 'topk_cluster':
        outp = (base_metrics / kind / f"K_{args.K}_{args.proto}_C_{args.num_clusters}_ap50_total{suffix}")
        outpg = (base_metrics / kind / f"K_{args.K}_{args.proto}_C_{args.num_clusters}_ap50_total_group{suffix}")
        outpg_post = (base_metrics / kind / f"K_{args.K}_{args.proto}_C_{args.num_clusters}_ap50_total_group_post{suffix}")
    else:
        outp = (base_metrics / kind / f"K_{args.K}_{args.proto}_ap50_total{suffix}")
        outpg = (base_metrics / kind / f"K_{args.K}_{args.proto}_ap50_total_group{suffix}")
        outpg_post = (base_metrics / kind / f"K_{args.K}_{args.proto}_ap50_total_group_post{suffix}")

    # 출력 디렉토리 생성
    outp.parent.mkdir(parents=True, exist_ok=True)
    outpg.parent.mkdir(parents=True, exist_ok=True)
    outpg_post.parent.mkdir(parents=True, exist_ok=True)

    # 클래스별 결과 저장
    if not outp.exists():
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Class", "fold_idx", "tau", "coverage@0.5", "fppi", "precision", "recall",
                        "f1@0.5", "ap50", "tp", "fp", "fn", "total_gt", "images",
                        "nms_iou", "K_per_GT"])

    for cname in class_names:
        row = row_dict[cname]
        ap50_value = ap50_dict[cname]
        cov, fppi, prec, rec, f1 = metrics_from_row(row)

        with outp.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{cname}", f"{args.fold_idx}", f"{args.tau:.2f}", f"{cov:.6f}", f"{fppi:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}",
                        f"{ap50_value:.6f}", row["TP"], row["FP"], row["FN"], row["total_gt"], row["images"],
                        f"{args.nms_iou:.2f}", args.K])

    # 그룹별 결과 저장
    if not outpg.exists():
        with outpg.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Group", "fold_idx", "tau", "coverage@0.5", "fppi", "precision", "recall",
                        "f1@0.5", "ap50", "tp", "fp", "fn", "total_gt", "images",
                        "nms_iou", "K_per_GT"])

    for gname in groups_name:
        row = g_row_dict[gname]
        ap50_value = g_ap50_dict[gname]
        cov, fppi, prec, rec, f1 = metrics_from_row(row)

        with outpg.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{gname}", f"{args.fold_idx}", f"{args.tau:.2f}", f"{cov:.6f}", f"{fppi:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}",
                        f"{ap50_value:.6f}", row["TP"], row["FP"], row["FN"], row["total_gt"], row["images"],
                        f"{args.nms_iou:.2f}", args.K])

    # 그룹별 후처리 결과 저장
    if not outpg_post.exists():
        with outpg_post.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Group", "fold_idx", "tau", "coverage@0.5", "fppi", "precision", "recall",
                        "f1@0.5", "ap50", "tp", "fp", "fn", "total_gt", "images",
                        "nms_iou", "K_per_GT"])

    for gname in groups_name:
        row = g_post_row_dict[gname]
        ap50_value = g_post_ap50_dict[gname]
        cov, fppi, prec, rec, f1 = metrics_from_row(row)

        with outpg_post.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [f"{gname}", f"{args.fold_idx}", f"{args.tau:.2f}", f"{cov:.6f}", f"{fppi:.6f}", f"{prec:.6f}",
                 f"{rec:.6f}", f"{f1:.6f}",
                 f"{ap50_value:.6f}", row["TP"], row["FP"], row["FN"], row["total_gt"], row["images"],
                 f"{args.nms_iou:.2f}", args.K])

    if args.proto == 'topk_cluster':
        print(f"Per-class ap results for K_{args.K}_{args.proto}_C_{args.num_clusters}_fold_{args.fold_idx} [saved] {kind} metrics -> {outp}")
    else:
        print(f"Per-class ap results for K_{args.K}_{args.proto}_fold_{args.fold_idx} [saved] {kind} metrics -> {outp}")


# ========= Viz fallback (draw best box even if below tau) =========
def draw_top1_fallback(img, boxes, scores, tau, font, color=(255,0,0)):
    """
    Top-1 폴백 시각화 - 임계값 미달 시에도 최고 점수 박스 표시
    
    Args:
        img: PIL 이미지 객체
        boxes: 정규화된 바운딩 박스 배열 [N, 4]
        scores: 점수 배열 [N]
        tau: 임계값
        font: 텍스트 폰트
        color: 박스 색상 (기본값: 빨간색)
    
    Returns:
        박스가 그려진 이미지
    """
    if scores.size == 0 or len(boxes) == 0:
        return img
    j = int(scores.argmax())  # 최고 점수 박스 인덱스
    b, s = boxes[j], float(scores[j])
    W, H = img.size
    x1,y1,x2,y2 = (b * np.array([W,H,W,H], np.float32)).tolist()  # 정규화된 좌표를 픽셀 좌표로 변환
    dr = ImageDraw.Draw(img)
    dr.rectangle([x1,y1,x2,y2], outline=color, width=2)
    dr.text((x1, max(0,y1-16)), f"top:{s:.2f} (<{tau:.2f})", fill=color, font=font)
    return img


def draw_top1_fallback_scale(img: Image.Image, boxes: np.ndarray, scores: np.ndarray, tau: float, font, color=(255,0,0)):
    """
    Top-1 폴백 시각화 (스케일링 버전) - 768x768로 리사이징 후 최고 점수 박스 표시
    
    Args:
        img: PIL 이미지 객체
        boxes: 정규화된 바운딩 박스 배열 [N, 4]
        scores: 점수 배열 [N]
        tau: 임계값
        font: 텍스트 폰트
        color: 박스 색상 (기본값: 빨간색)
    
    Returns:
        768x768로 리사이징되고 박스가 그려진 이미지
    """
    if scores.size == 0 or len(boxes) == 0:
        return img

    # 768x768로 리사이징 (OwlV2 비율 유지 padding 포함)
    orig_w, orig_h = img.size
    target_size = 768
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    img_resized_padded = Image.new("RGB", (target_size, target_size), (0,0,0))
    img_resized_padded.paste(img_resized, (pad_left, pad_top))

    # top-1 fallback
    j = int(scores.argmax())
    b, s = boxes[j], float(scores[j])

    # 좌표를 리사이즈된 이미지 기준으로 변환
    x1 = b[0] * new_w + pad_left
    y1 = b[1] * new_h + pad_top
    x2 = b[2] * new_w + pad_left
    y2 = b[3] * new_h + pad_top

    draw = ImageDraw.Draw(img_resized_padded)
    draw_box_with_text(draw, np.array([x1, y1, x2, y2]), f"top:{s:.2f} (<{tau:.2f})", color)

    return img_resized_padded


def resize_with_padding(img: Image.Image, target_size=(768, 768)):
    """
    이미지 리사이징 및 패딩 - 비율을 유지하면서 목표 크기로 리사이징
    
    Args:
        img: PIL 이미지 객체
        target_size: 목표 크기 (기본값: (768, 768))
    
    Returns:
        (img_padded, scale, pad_x, pad_y): 패딩된 이미지, 스케일 비율, 패딩 좌표
    """
    ow, oh = img.size
    tw, th = target_size
    scale = min(tw / ow, th / oh)  # 비율 유지를 위한 스케일
    new_w, new_h = int(ow * scale), int(oh * scale)
    img_resized = img.resize((new_w, new_h))

    # 패딩 이미지 생성
    img_padded = Image.new("RGB", (tw, th), (0, 0, 0))
    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2
    img_padded.paste(img_resized, (pad_x, pad_y))

    return img_padded, scale, pad_x, pad_y


def scale_boxes_to_padded_image(boxes, orig_size, scale, pad_x, pad_y):
    """
    바운딩 박스를 패딩된 이미지 좌표로 변환
    
    Args:
        boxes: 정규화된 바운딩 박스 배열 [N, 4]
        orig_size: 원본 이미지 크기 (width, height)
        scale: 스케일 비율
        pad_x: X축 패딩
        pad_y: Y축 패딩
    
    Returns:
        패딩된 이미지 기준 바운딩 박스 배열 [N, 4]
    """
    ow, oh = orig_size
    boxes_scaled = boxes.copy()
    boxes_scaled[:, [0, 2]] = boxes[:, [0, 2]] * ow * scale + pad_x  # x1, x2 변환
    boxes_scaled[:, [1, 3]] = boxes[:, [1, 3]] * oh * scale + pad_y  # y1, y2 변환
    return boxes_scaled


def draw_box_with_text(draw, box, text, color):
    """
    바운딩 박스와 텍스트 그리기 - 박스와 라벨을 함께 그리는 유틸리티 함수
    
    Args:
        draw: PIL ImageDraw 객체
        box: 바운딩 박스 좌표 [x1, y1, x2, y2]
        text: 표시할 텍스트
        color: 박스 색상
    """
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    x1, y1, x2, y2 = box.tolist()

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # 텍스트 크기 계산 및 배경 박스 그리기
    bbox = font.getbbox(text)  # (left, top, right, bottom)
    text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    draw.rectangle([x1, max(0, y1 - text_size[1]), x1 + text_size[0], y1], fill=color)

    draw.text((x1, max(0, y1 - text_size[1])), text, fill=(255,255,255), font=font)


def visualize_result(args, img_path, cname, img, gt_xyxy, s_strict, boxes):
    """
    검출 결과 시각화 - Ground Truth와 예측 결과를 함께 표시하는 이미지 생성
    
    Args:
        args: 설정 인수 (vis_dir, K, tau, nms_iou, proto 등)
        img_path: 이미지 파일 경로
        cname: 클래스명
        img: PIL 이미지 객체
        gt_xyxy: Ground Truth 바운딩 박스 배열 [N, 4]
        s_strict: 예측 점수 배열 [M]
        boxes: 예측 바운딩 박스 배열 [M, 4]
    """
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    base_viz = args.vis_dir / f"k{args.K}"
    TAU_VIZ = args.tau
    NMS_IOU_VIS = args.nms_iou

    # 시각화는 메트릭과 동일한 τ와 NMS 사용
    TOP_K_VIS = 30

    # 임계값 이상의 예측만 선택
    keep_v = np.where(s_strict >= TAU_VIZ)[0]
    vb, vs = (boxes[keep_v], s_strict[keep_v]) if keep_v.size > 0 else (
        np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    )
    
    if len(vb) > 0:
        from utils.infer_utils import iou_xyxy, nms_xyxy
        kept = nms_xyxy(vb, vs, NMS_IOU_VIS)  # NMS 적용
        vb, vs = vb[kept], vs[kept]
        if TOP_K_VIS and len(vs) > TOP_K_VIS:  # 상위 K개만 표시
            topk = vs.argsort()[::-1][:TOP_K_VIS]
            vb, vs = vb[topk], vs[topk]

    # 이미지를 768x768로 리사이징 및 패딩
    img_s, scale, pad_x, pad_y = resize_with_padding(img, (768, 768))

    # GT와 예측 박스를 패딩된 이미지 좌표로 변환
    gt_scaled = scale_boxes_to_padded_image(gt_xyxy, img.size, scale, pad_x, pad_y)
    vb_scaled = scale_boxes_to_padded_image(vb, img.size, scale, pad_x, pad_y)

    draw = ImageDraw.Draw(img_s)
    W, H = img_s.size

    # Ground Truth 박스 그리기 (녹색)
    for g in gt_scaled:
        draw_box_with_text(draw, g, "GT", (0, 255, 0))

    # 예측 결과 그리기
    if len(vb_scaled) == 0:
        # 예측이 없으면 최고 점수 박스 표시 (폴백)
        img_s = draw_top1_fallback_scale(img_s, boxes, s_strict, TAU_VIZ, font, color=(255, 0, 0))
    else:
        # 예측 박스들 그리기 (파란색)
        for b, s in zip(vb_scaled, vs):
            draw_box_with_text(draw, b, f"{cname}:{s:.2f}", (0, 0, 255))

    # 결과 이미지 저장
    (base_viz / "STRICT" / cname).mkdir(parents=True, exist_ok=True)
    if args.proto == 'topk_cluster':
        img_s.save(
            base_viz / "STRICT" / cname / f"{img_path.stem}_{args.proto}_cluster_{args.num_clusters}_tau{TAU_VIZ:.2f}_nms{NMS_IOU_VIS:.2f}.jpg")
    else:
        img_s.save(
            base_viz / "STRICT" / cname / f"{img_path.stem}_{args.proto}_tau{TAU_VIZ:.2f}_nms{NMS_IOU_VIS:.2f}.jpg")
