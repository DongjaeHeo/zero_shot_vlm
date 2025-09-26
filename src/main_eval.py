#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
메인 평가 스크립트 - Zero-shot 객체 검출 시스템의 핵심 평가 모듈

이 스크립트는 다음과 같은 기능을 수행합니다:
1. 저장된 임베딩에서 클래스별 프로토타입 생성 (top-K per GT 방식)
2. 테스트 이미지에 대한 임베딩 기반 객체 검출 수행
3. 바이너리 분류기를 통한 검출 결과 정제 (선택적)
4. STRICT/RELAXED 평가 모드로 성능 측정
5. AP@0.5, Coverage@0.5, FPPI 등 다양한 메트릭 계산
6. 시각화 결과 생성 및 CSV 형태로 결과 저장

핵심 특징:
- 5-fold cross validation 지원
- 슬라이딩 윈도우를 통한 대형 이미지 처리
- 클래스별/그룹별 프로토타입 기반 검출
- 바이너리 분류기를 통한 false positive 제거
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
from utils.define_class_group import *

# ========= 슬라이딩 윈도우 설정 =========
# 타일링 파라미터 (임베딩 추출기와 동일)
PATCH, OVERLAP = 1008, 500  # 패치 크기와 겹침 크기
STRIDE, CORE   = PATCH-OVERLAP, 250  # 스트라이드와 코어 영역 크기
TILE_EXCEPT    = {"group_of_trees"}  # 타일링을 하지 않을 클래스 (대소문자 무관)


def run(args, splits):
    """
    메인 평가 함수 - 클래스별 프로토타입을 사용한 zero-shot 객체 검출 평가
    
    Args:
        args: 명령행 인자 (임계값, K값, 프로토타입 방식 등)
        splits: 5-fold cross validation을 위한 데이터 분할 정보
    """
    # ========= 데이터셋 경로 설정 =========
    if args.eval_sample:
        SRC_ROOT = Path("../dataset/sample_data").resolve()  # 샘플 데이터셋 경로
    else:
        SRC_ROOT = Path("/home/dromii_shared/obstacle_subclass").resolve()  # 전체 데이터셋 경로

    # ========= 출력 디렉토리 설정 =========
    base_viz = args.vis_dir / f"k{args.K}"  # 시각화 결과 저장 경로
    base_metrics = args.metric_dir / f"k{args.K}"  # 메트릭 결과 저장 경로
    ensure_dirs(base_viz, base_metrics)  # 디렉토리 생성

    # ========= 클래스별 프로토타입 생성 =========
    # 훈련 데이터(80%)에서 GT당 top-K 임베딩을 사용하여 프로토타입 생성
    # class_protos: 클래스별 평균 프로토타입 (top-K per GT의 평균)
    group_protos, class_protos = build_class_protos_topk(
        packs, K=args.K, splits=splits, group_name=args.class_groups_raw, args=args
    )
    class_names = sorted(class_protos.keys())
    if not class_names:
        print(f"[K={args.K}] No classes with prototypes found.")
        return

    # topk_avg 방식의 경우 모든 클래스 프로토타입을 스택으로 구성
    if args.proto == 'topk_avg':
        P = np.stack([class_protos[c] for c in class_names], 0)  # [C,D] (L2 정규화됨)

    # 클래스명을 인덱스로 매핑
    name_to_idx = {c: i for i, c in enumerate(class_names)}

    # ========= 그룹 설정 (RELAXED 평가용) =========
    # 클래스명을 기반으로 그룹 해결
    groups_resolved = resolve_groups(packs, args.class_groups_raw)
    groups_name = list(groups_resolved.keys())

    # ========= 메트릭 누적기 초기화 =========
    # 클래스별 누적기 (각 모드당 클래스별로 하나의 행)
    strict_rows: Dict[str, Dict[str, int]] = {c: init_res_row() for c in class_names}
    group_rows: Dict[str, Dict[str, int]] = {c: init_res_row() for c in groups_name}
    group_post_rows: Dict[str, Dict[str, int]] = {c: init_res_row() for c in groups_name}
    relaxed_rows: Dict[str, Dict[str, int]] = {c: init_res_row() for c in class_names}

    # ========= AP 계산용 누적기 (예측 + GT per 클래스) =========
    ap_pred_strict: Dict[str, List[Tuple[str, float, np.ndarray]]] = {c: [] for c in class_names}
    ap_pred_group: Dict[str, List[Tuple[str, float, np.ndarray]]] = {c: [] for c in groups_name}
    ap_pred_group_post: Dict[str, List[Tuple[str, float, np.ndarray]]] = {c: [] for c in groups_name}
    gts_by_image: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in class_names}
    group_gts_by_image: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in groups_name}

    # ========= 추론 메인 루프 =========
    tot_ap_result = {}  # 클래스별 AP 결과
    g_tot_ap_result = {}  # 그룹별 AP 결과
    g_post_tot_ap_result = {}  # 그룹별 후처리 AP 결과

    # ========= 후보 생성 =========
    # 그룹별 후보 저장용 딕셔너리 (이미지별로 그룹별 후보 리스트)
    group_cand = defaultdict(lambda: defaultdict(list))
    
    for cname in class_names:
        # 현재 클래스가 속한 그룹 찾기
        group_name = next(group for group, members in args.class_groups_raw.items() if cname in members)
        group_id = args.class_groups_id[group_name]

        print(f"\n[K={args.K}] [Fold={args.fold_idx}] [eval] class={cname}")
        class_id = name_to_idx[cname]
        allowed_strict = [class_id]  # STRICT 모드에서 허용되는 클래스 인덱스

        # 훈련/테스트 이미지 분할 정보 가져오기
        train_imgs, test_imgs = splits[cname]

        # ========= 이미지/라벨 디렉토리 설정 =========
        if args.eval_sample:
            img_dir = SRC_ROOT / "images"  # 샘플 데이터셋의 경우
            lbl_dir = SRC_ROOT / "labels"
        else:
            img_dir = SRC_ROOT / cname / "images"  # 클래스별 디렉토리
            lbl_dir = SRC_ROOT / cname / "labels"

        # 타일링 여부 결정 (특정 클래스는 타일링 제외)
        do_tiles = (cname.lower() not in {s.lower() for s in TILE_EXCEPT})

        # ========= 테스트 이미지 수집 (20% TEST) =========
        unseen = []
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.tif", "*.tiff"):
            if args.eval_sample:
                # 샘플 데이터셋의 경우 모든 이미지 사용
                unseen += [p for p in img_dir.glob(ext)]
            else:
                # 테스트 세트에 속한 이미지만 사용
                unseen += [p for p in img_dir.glob(ext) if str(p) in test_imgs]

        unseen = sorted(unseen)
        
        # ========= 시각화 디렉토리 생성 =========
        (base_viz / "STRICT" / cname).mkdir(parents=True, exist_ok=True)
        (base_viz / "RELAXED" / cname).mkdir(parents=True, exist_ok=True)

        # ========= 각 테스트 이미지에 대한 검출 수행 =========
        for img_path in unseen:
            img_id = img_path.name
            img = load_image(img_path)  # EXIF 정보를 고려한 이미지 로드

            # ========= Ground Truth 로드 =========
            raw_gt = read_gt_yolo(lbl_dir / f"{img_path.stem}.txt")  # YOLO 형식 라벨 읽기

            if args.eval_sample:
                # 샘플 데이터셋의 경우 특정 그룹 ID만 필터링
                gt = raw_gt[raw_gt[:, 0] == group_id]
                if gt.shape[0] == 0:
                    continue  # 해당 그룹의 GT가 없으면 스킵
            else:
                gt = raw_gt  # 전체 GT 사용

            # YOLO 형식 (cx, cy, w, h)을 xyxy 형식으로 변환
            gt_xyxy = boxes_cxcywh_to_xyxy_norm(gt[:, 1:5]) if gt.shape[0] else np.zeros((0, 4), np.float32)

            # GT 정보를 이미지별로 저장 (AP 계산용)
            gts_by_image[cname][img_id] = gt_xyxy
            group_gts_by_image[group_name][img_id] = gt_xyxy

            # ========= 임베딩 추출 (캐시 확인) =========
            npz_file = img_path.with_suffix(".npz")

            if npz_file.exists():
                # 이미 추출된 임베딩이 있으면 로드
                data = np.load(npz_file)
                boxes = data["boxes"]  # 바운딩 박스 [N, 4]
                embs = data["embs"]    # 임베딩 [N, D]
            else:
                # 임베딩이 없으면 새로 추출
                print(f'{npz_file.name} embedding extracting.')
                boxes, embs = run_detector_with_tiling(img, args.min_area, args.device, do_tiles)
                np.savez(npz_file, boxes=boxes, embs=embs)  # 캐시로 저장

            # ========= 임베딩이 없는 경우 처리 =========
            if len(embs) == 0:
                # 여전히 이미지/GT 카운트는 수행
                for row in (strict_rows[cname], relaxed_rows[cname]):
                    row["images"] += 1
                    row["total_gt"] += len(gt_xyxy)

                print(f"[K={args.K}] [{cname}/{img_path.name}] props=0")
                continue

            # ========= 코사인 유사도 계산 =========
            # 임베딩을 L2 정규화 (코사인 유사도 계산을 위해)
            E = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
            
            # 클래스별/그룹별 프로토타입 매트릭스 설정
            Pmat = class_protos[cname]      # 클래스별 프로토타입
            PmatG = group_protos[group_name]  # 그룹별 프로토타입

            # 모든 프로토타입과의 코사인 유사도 계산
            S = cosine_mat(E, Pmat)    # [N,C] - 클래스별 유사도
            S_G = cosine_mat(E, PmatG)  # [N,C] - 그룹별 유사도

            # ========= STRICT 모드 점수 계산 =========
            # 프로토타입이 1차원인 경우 (단일 프로토타입)와 다차원인 경우 처리
            if len(np.shape(S)) == 1:
                s_strict = S           # 단일 프로토타입의 경우 직접 사용
                s_strict_G = S_G
            else:
                s_strict = np.max(S, axis=1)      # 다중 프로토타입의 경우 최대값 사용
                s_strict_G = np.max(S_G, axis=1)

            # ========= 바이너리 분류기를 통한 예측 정제 =========
            if args.binary_classifier:
                # 훈련된 바이너리 분류기 로드
                cnn_dir = args.cnn_dir / cname / args.inter_dir
                classifier, tf_eval, crop_size, ctx = load_cnn_for_class(cnn_dir, cname, args.device)

                # 바운딩 박스를 크롭으로 변환하여 분류기 입력 준비
                xb = crops_to_tensor(img, boxes, tf_eval, context=ctx).to(args.device, non_blocking=True)
                logits = classifier(xb).squeeze(-1)  # 분류기 예측
                prob_owl = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)  # 확률로 변환

                # 임계값을 사용하여 true/false 마스크 생성
                mask_true = prob_owl >= float(args.thr_use)   # 임계값 이상인 박스들
                mask_false = ~mask_true                       # 임계값 미만인 박스들

                # true 마스크를 적용하여 박스와 점수 필터링
                boxes, s_strict = boxes[mask_true], s_strict[mask_true]
                boxes_false, s_strict_false = boxes[mask_false], s_strict[mask_false]

            # ========= 시각화 (STRICT 모드) =========
            if args.visualization:
                visualize_result(args, img_path, cname, img, gt_xyxy, s_strict, boxes)

            # 그룹별 후보 저장 (후처리용)
            group_cand[img_id][group_name].append((boxes, s_strict, gt_xyxy))

            # ========= 단일 지점 메트릭을 위한 후보 예측 평가 (τ + NMS + TOP_K) =========
            # 임계값(τ) 적용
            keep_tau = np.where(s_strict >= args.tau)[0]      # 클래스별 임계값 필터링
            keep_tau_G = np.where(s_strict_G >= args.tau)[0]  # 그룹별 임계값 필터링
            
            # 임계값을 통과한 박스와 점수 추출
            pb, ps = (
                boxes[keep_tau], s_strict[keep_tau]) \
                if keep_tau.size > 0 else (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))

            pb_G, ps_G = (
                boxes[keep_tau_G], s_strict_G[keep_tau_G]) \
                if keep_tau_G.size > 0 else (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))

            # NMS 및 TOP-K 적용
            if len(pb) > 0:
                kept = nms_xyxy(pb, ps, args.nms_iou)  # NMS 적용
                pb, ps = pb[kept], ps[kept]
                if args.top_k and len(ps) > args.top_k:
                    topk = ps.argsort()[::-1][:args.top_k]  # 상위 K개 선택
                    pb, ps = pb[topk], ps[topk]

            # TP, FP, FN, Coverage 계산
            tp, fp, fn, cov = match_and_score(pb, ps, gt_xyxy, args.ap_iou)

            # ========= STRICT 모드 메트릭 누적 =========
            R = strict_rows[cname]
            R["TP"] += tp
            R["FP"] += fp
            R["FN"] += fn
            R["covered"] += cov
            R["total_gt"] += len(gt_xyxy)
            R["images"] += 1
            R["fp_count"] += fp

            # ========= 그룹별 메트릭 계산 =========
            if len(pb_G) > 0:
                kept_G = nms_xyxy(pb_G, ps_G, args.nms_iou)  # 그룹별 NMS
                pb_G, ps_G = pb_G[kept_G], ps_G[kept_G]
                if args.top_k and len(ps_G) > args.top_k:
                    topk_G = ps_G.argsort()[::-1][:args.top_k]  # 그룹별 TOP-K
                    pb_G, ps_G = pb_G[topk_G], ps_G[topk_G]

            tp, fp, fn, cov = match_and_score(pb_G, ps_G, gt_xyxy, args.ap_iou)

            # ========= 그룹별 메트릭 누적 =========
            G = group_rows[group_name]
            G["TP"] += tp
            G["FP"] += fp
            G["FN"] += fn
            G["covered"] += cov
            G["total_gt"] += len(gt_xyxy)
            G["images"] += 1
            G["fp_count"] += fp

            # ========= AP 계산용 예측 저장 (τ 필터링 없이, TOP_K 없이) =========
            if len(boxes) > 0:
                kept_all = nms_xyxy(boxes, s_strict, args.nms_iou) if len(boxes) > 0 else []
                if kept_all:
                    b_all = boxes[kept_all]
                    s_all = s_strict[kept_all]
                    for b_, s_ in zip(b_all, s_all):
                        # AP 계산을 위한 예측 결과 저장 (이미지ID, 점수, 박스)
                        ap_pred_strict[cname].append((img_id, float(s_), b_.astype(np.float32)))
                        ap_pred_group[group_name].append((img_id, float(s_), b_.astype(np.float32)))

    # -------- Evaluate Candidate Predictions after postprocessing for group_post ------------
    for tmp_img_id, group_dict in group_cand.items():
        for tmp_g_name, cand_list in group_dict.items():
            tmp_boxes_filtered = []
            tmp_strict_filtered = []
            for tmp_cand_list in cand_list:
                keep_tau = np.where(tmp_cand_list[1] >= args.tau)[0]
                tmp_pb, tmp_ps = (
                    tmp_cand_list[0][keep_tau], tmp_cand_list[1][keep_tau]) \
                    if keep_tau.size > 0 else (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))

                tmp_boxes_filtered.append(tmp_pb)
                tmp_strict_filtered.append(tmp_ps)

            tmp_boxes_filtered = np.concatenate(tmp_boxes_filtered, axis=0)
            tmp_strict_filtered = np.concatenate(tmp_strict_filtered, axis=0)
            tmp_boxes = np.concatenate([x[0] for x in cand_list], axis=0)
            tmp_strict = np.concatenate([x[1] for x in cand_list], axis=0)
            tmp_gt_xyxx = cand_list[0][2]

            # For single-point metrics (τ + NMS + TOP_K)
            if len(tmp_boxes_filtered) > 0:
                kept = nms_xyxy(tmp_boxes_filtered, tmp_strict_filtered, args.nms_iou)
                tmp_boxes_filtered, tmp_strict_filtered = tmp_boxes_filtered[kept], tmp_strict_filtered[kept]
                if args.top_k and len(tmp_strict_filtered) > args.top_k:
                    topk = tmp_strict_filtered.argsort()[::-1][:args.top_k]
                    tmp_boxes_filtered, tmp_strict_filtered = tmp_boxes_filtered[topk], tmp_strict_filtered[topk]

            tp, fp, fn, cov = match_and_score(tmp_boxes_filtered, tmp_strict_filtered, tmp_gt_xyxx, args.ap_iou)

            G_post = group_post_rows[tmp_g_name]
            G_post["TP"] += tp
            G_post["FP"] += fp
            G_post["FN"] += fn
            G_post["covered"] += cov
            G_post["total_gt"] += len(tmp_gt_xyxx)
            G_post["images"] += 1
            G_post["fp_count"] += fp

            # For AP: use ALL predictions after NMS (no τ filter, no TOP_K)
            if len(tmp_boxes) > 0:
                kept_all = nms_xyxy(tmp_boxes, tmp_strict, args.nms_iou) if len(tmp_boxes) > 0 else []
                if kept_all:
                    b_all = tmp_boxes[kept_all]
                    s_all = tmp_strict[kept_all]
                    for b_, s_ in zip(b_all, s_all):
                        ap_pred_group_post[tmp_g_name].append((tmp_img_id, float(s_), b_.astype(np.float32)))

    # ----- Compute AP@0.5 per class (STRICT/RELAXED) -----
    for g_name_tmp, ap_pred_group_tmp in ap_pred_group.items():
        g_tot_ap_result[g_name_tmp] = compute_ap50(ap_pred_group_tmp, group_gts_by_image[g_name_tmp], args.ap_iou)

    for g_name_tmp, ap_pred_group_post_tmp in ap_pred_group_post.items():
        g_post_tot_ap_result[g_name_tmp] = compute_ap50(ap_pred_group_post_tmp, group_gts_by_image[g_name_tmp], args.ap_iou)

    for c_name_tmp, ap_pred_strict_tmp in ap_pred_strict.items():
        tot_ap_result[c_name_tmp] = compute_ap50(ap_pred_strict_tmp, gts_by_image[c_name_tmp], args.ap_iou)

    write_csv_tot(
        "STRICT",
        strict_rows, tot_ap_result,
        group_rows, g_tot_ap_result,
        group_post_rows, g_post_tot_ap_result,
        class_names, groups_name,
        args)


if __name__ == "__main__":
    # ========= 설정 =========
    OUT_IDX = Path("../index").resolve()  # 임베딩 인덱스 저장 경로
    VIZ_DIR = Path("../results/viz_fixed_5").resolve()  # 시각화 결과 저장 경로 (k<K>별로 그룹화)
    METRICS_DIR = Path("../results/metrics_fixed_5").resolve()  # 클래스별 CSV 결과 저장 경로 (k<K>별로 그룹화)
    CNN_DIR = Path("../cnn_models_roi_on_fly").resolve()  # 바이너리 분류기 모델 저장 경로

    # ========= 명령행 인자 파싱 =========
    parser = argparse.ArgumentParser(description="Zero-shot 객체 검출 평가 스크립트")
    
    # 기본 설정
    parser.add_argument('-seed', "--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument('-did', "--device_id", type=str, default="0", help="사용할 GPU 디바이스 ID")
    
    # 검출 임계값 설정
    parser.add_argument('-t', "--tau", type=float, default=0.95,
                        help="프로토타입과의 코사인 유사도 임계값")
    parser.add_argument("--K", type=int, default=2, help="GT당 사용할 top-K 임베딩 수")
    parser.add_argument("--fold_idx", type=int, default=5, help="5-fold CV에서 사용할 폴드 인덱스")
    
    # 후처리 설정
    parser.add_argument('-tk', "--top_k", type=int, default=15, help="NMS 후 유지할 최대 검출 수")
    parser.add_argument('-ma', "--min_area", type=float, default=0.0, help="최소 박스 면적 필터")
    parser.add_argument('-nt', "--nms_iou", type=float, default=0.10,
                        help="NMS IoU 임계값 (메트릭과 시각화 모두에 적용)")
    parser.add_argument('-ai', "--ap_iou", type=float, default=0.30,
                        help="TP/FN 판정을 위한 IoU 임계값 (원래 coverage@0.5 헤더)")
    
    # 프로토타입 방식 설정
    parser.add_argument('-p', "--proto", type=str, default='topk_avg',
                        help="프로토타입 생성 방식: ['topk_avg', 'topk_raw', 'topk_cluster']")
    parser.add_argument('-nc', "--num_clusters", type=int, default=4, help="클러스터링 방식에서 사용할 클러스터 수")
    
    # 바이너리 분류기 설정
    parser.add_argument('-bc', "--binary_classifier", action='store_true', help="바이너리 분류기 사용 여부")
    parser.add_argument('-bct', "--thr_use", type=float, default=0.50, help="바이너리 분류기 임계값")
    
    # 평가 모드 설정
    parser.add_argument('-sample', "--eval_sample", action='store_true', help="샘플 데이터셋 사용 여부")
    parser.add_argument('-fcv', "--fold_cv", action='store_true', help="5-fold cross validation 수행 여부")
    parser.add_argument('-vis', "--visualization", action='store_true', help="시각화 결과 생성 여부")
    
    args = parser.parse_args()

    # ========= 경로 설정 =========
    args.out_idx = OUT_IDX  # pos_<class>.pt 파일들이 저장된 경로
    args.cnn_dir = CNN_DIR  # 바이너리 분류기 모델 저장 경로
    args.vis_dir = VIZ_DIR  # 시각화 결과 저장 경로
    args.metric_dir = METRICS_DIR  # 메트릭 결과 저장 경로

    # 클래스 그룹 설정
    args.class_groups_raw = CLASS_GROUPS_RAW  # 원시 클래스 그룹 정의
    args.class_groups_id = CLASS_GROUPS_ID    # 그룹 ID 매핑

    # ========= 랜덤 시드 설정 =========
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ========= 디바이스 설정 =========
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id.strip()
    
    # 디바이스 설정 및 검증
    if torch.cuda.is_available():
        assert torch.cuda.device_count() >= 1, "No CUDA device visible after CUDA_VISIBLE_DEVICES."
        torch.cuda.set_device(0)  # index within the visible list
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # ========= 프로토타입 설정 =========
    # 훈련 데이터(80%)에서 GT당 top-K IoU 임베딩을 사용하여 프로토타입 생성
    PROTO_TOP_PER_GT: Union[int, List[int]] = [1, 2, 3, 4, 5]  # 평가할 K 값들
    Ks: Iterable[int] = PROTO_TOP_PER_GT if isinstance(PROTO_TOP_PER_GT, (list, tuple)) else [int(PROTO_TOP_PER_GT)]

    # ========= 임베딩 데이터 로드 =========
    print(f"Load Class Embedding Vectors\n")
    packs = load_pos_packs(args.out_idx)  # 클래스별 임베딩 팩 로드

    # ========= 5-fold cross validation 분할 생성 =========
    splits = create_5fold_splits(packs, args.seed)

    # ========= 평가 실행 =========
    if args.fold_cv:
        # 5-fold cross validation 수행
        print(f"Data split for 5-fold cross validation\n")
        for fold_idx, split in splits.items():
            args.inter_dir = f"{args.seed}_fold_{fold_idx}"  # 중간 결과 저장 디렉토리
            for k in Ks:
                print(f"Evaluation for {args.proto}_K_{k}_fold: {fold_idx}")
                args.K = k
                args.fold_idx = fold_idx
                run(args, split)  # 각 K값과 폴드에 대해 평가 실행

            # 다른 프로토타입 방식들도 테스트할 수 있음 (주석 처리됨)
            # args.proto = "topk_raw"
            # for k in Ks:
            #     run(args, split, k, fold_idx)

            # args.proto = 'topk_cluster'
            # cluster_list = [20]
            # for cluster in cluster_list:
            #     args.num_clusters = cluster
            #     for k in Ks:
            #         run(args, split, k, fold_idx)
    else:
        # 단순 평가 (첫 번째 폴드만 사용)
        print(f"Data split for simple evaluation\n")
        fold_idx = 0
        split = splits[fold_idx]
        args.inter_dir = f"{args.seed}_fold_{fold_idx}"
        for k in Ks:
            print(f"Evaluation for {args.proto}_K_{k}_fold: {fold_idx}")
            args.K = k
            args.fold_idx = fold_idx
            run(args, split)  # 각 K값에 대해 평가 실행

        # 다른 프로토타입 방식들도 테스트할 수 있음 (주석 처리됨)
        # args.proto = "topk_raw"
        # for k in Ks:
        #     run(args, split, k, fold_idx)

        # args.proto = 'topk_cluster'
        # cluster_list = [20]
        # for cluster in cluster_list:
        #     args.num_clusters = cluster
        #     for k in Ks:
        #         run(args, split, k, fold_idx)




