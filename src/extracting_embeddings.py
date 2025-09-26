"""
임베딩 추출 스크립트 - OWLv2 모델을 사용한 객체 검출 및 임베딩 추출

이 스크립트는 다음과 같은 기능을 수행합니다:
1. 클래스별 이미지에서 OWLv2 모델을 사용한 객체 검출 수행
2. 슬라이딩 윈도우를 통한 대형 이미지 처리 (특정 클래스 제외)
3. Ground Truth와의 IoU 계산을 통한 양성 샘플 선별
4. 중복 제거 및 GT당 최대 샘플 수 제한
5. 추출된 임베딩과 메타데이터를 클래스별로 저장

핵심 특징:
- 타일링을 통한 대형 이미지 처리
- IoU 기반 양성 샘플 선별
- 중복 제거를 통한 데이터 품질 향상
- 클래스별 임베딩 인덱스 생성
"""

from pathlib import Path
import os, glob, time
from typing import Dict, Tuple
import torch, numpy as np
from PIL import Image, ImageOps
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# ========= 설정 =========
SRC_ROOT   = Path("/home/dromii_shared/obstacle_subclass").resolve()  # 소스 데이터셋 경로
OUT_IDX    = Path("index").resolve()  # 임베딩 인덱스 저장 경로
OUT_CROPS  = Path("crops").resolve()  # 크롭 이미지 저장 경로
MODEL_NAME = "google/owlv2-large-patch14-ensemble"  # 사용할 OWLv2 모델명
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 디바이스 설정

# ========= 임베딩 추출 파라미터 =========
TAU        = 0.65       # 양성 샘플을 위한 IoU 임계값
MAX_PER_GT = 5          # GT당 최대 샘플 수 제한
DEDUP_IOU  = 0.80       # 중복 제거를 위한 IoU 임계값

# ========= 타일링 파라미터 =========
PATCH, OVERLAP = 1008, 500  # 패치 크기와 겹침 크기
STRIDE, CORE   = PATCH-OVERLAP, 250  # 스트라이드와 코어 영역 크기
# 특정 클래스는 타일링 제외 (대소문자 무관)
TILE_EXCEPT = {"group_of_trees"}


# ========= 유틸리티 함수들 =========
def l2norm(x: torch.Tensor, dim=-1, eps=1e-8):
    """
    L2 정규화 함수 - 벡터를 단위 벡터로 변환
    
    Args:
        x: 정규화할 텐서
        dim: 정규화할 차원
        eps: 0으로 나누기 방지를 위한 작은 값
    
    Returns:
        L2 정규화된 텐서
    """
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def boxes_cxcywh_to_xyxy(a: np.ndarray) -> np.ndarray:
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


def iou_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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


def iou_one(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 개의 바운딩 박스 간의 IoU 계산
    
    Args:
        a: 첫 번째 바운딩 박스 [x1, y1, x2, y2]
        b: 두 번째 바운딩 박스 [x1, y1, x2, y2]
    
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1 = max(a[0], b[0])  # 교집합 x1
    y1 = max(a[1], b[1])  # 교집합 y1
    x2 = min(a[2], b[2])  # 교집합 x2
    y2 = min(a[3], b[3])  # 교집합 y2

    iw = max(x2 - x1, 0.0)  # 교집합 너비
    ih = max(y2 - y1, 0.0)  # 교집합 높이
    inter = iw * ih  # 교집합 면적

    # 합집합 면적 계산
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-8
    return float(inter / ua)


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


# ========= 모델 초기화 =========
proc = Owlv2Processor.from_pretrained(MODEL_NAME)  # OWLv2 전처리기
model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE).eval()  # OWLv2 모델


def infer(img: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
    """
    이미지에서 바운딩 박스와 임베딩을 추출하는 추론 함수
    
    Args:
        img: 입력 이미지 (PIL Image)
    
    Returns:
        boxes: [N, 4] 형태의 바운딩 박스 배열 (xyxy, 정규화됨)
        embs: [N, D] 형태의 임베딩 텐서 (L2 정규화됨)
    """
    # 이미지 전처리
    inp = proc(images=img, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # 이미지 임베딩 추출
        fmap = model.image_embedder(inp.pixel_values)[0]  # [1, h, w, D]
        B, h, w, D = fmap.shape
        feats = fmap.reshape(1, h * w, D)  # [1, N, D] - N = h*w
        
        # 바운딩 박스 예측
        pred = model.box_predictor(feats, feature_map=fmap)[0].cpu().numpy()  # cxcywh norm
        
        # 클래스 임베딩 추출
        _, cls = model.class_predictor(feats)

    # 바운딩 박스를 xyxy 형식으로 변환하고 정규화
    boxes = np.clip(boxes_cxcywh_to_xyxy(pred), 0.0, 1.0)  # xyxy norm
    
    # 임베딩을 L2 정규화
    embs = l2norm(cls[0].float(), dim=-1).cpu().to(torch.float32)  # [N, E]
    
    return boxes, embs


def mine_image(img_p: Path, lbl_p: Path, class_name: str, do_tiles: bool, store: Dict[str, Dict[str, list]]):
    """
    단일 이미지에서 양성 샘플을 마이닝하는 함수
    
    Args:
        img_p: 이미지 파일 경로
        lbl_p: 라벨 파일 경로
        class_name: 클래스 이름
        do_tiles: 타일링 수행 여부
        store: 임베딩과 메타데이터를 저장할 딕셔너리
    """
    # ========= 이미지 로드 (EXIF 정보 고려) =========
    # EXIF 정보를 고려한 안정적인 이미지 로드 (크롭 생성 시 재사용)
    orig = ImageOps.exif_transpose(Image.open(img_p).convert("RGB"))
    W, H = orig.size

    # ========= Ground Truth 로드 =========
    gts = read_gt_yolo(lbl_p)
    if gts.shape[0] == 0:
        return  # GT가 없으면 스킵

    # YOLO 형식을 xyxy 형식으로 변환하고 정규화
    gt_xyxy = np.clip(boxes_cxcywh_to_xyxy(gts[:, 1:5]), 0.0, 1.0)
    per_gt = {i: [] for i in range(gt_xyxy.shape[0])}  # GT별 후보 저장

    # ========= 전체 이미지에서 검출 =========
    boxes, embs = infer(orig)
    if boxes.shape[0]:
        # 검출된 박스와 GT 간의 IoU 계산
        ious = iou_mat(boxes, gt_xyxy)
        max_i = ious.argmax(1)  # 각 검출 박스에 대해 가장 높은 IoU를 가진 GT 인덱스
        max_v = ious.max(1)     # 각 검출 박스에 대해 가장 높은 IoU 값
        
        for pidx in range(boxes.shape[0]):
            gid = int(max_i[pidx])  # 매칭된 GT ID
            iou = float(max_v[pidx])  # IoU 값
            
            if iou < TAU:
                continue  # 임계값 미만이면 스킵

            # 정규화된 좌표를 픽셀 좌표로 변환
            x1, y1, x2, y2 = (boxes[pidx] * np.array([W, H, W, H], np.float32))
            
            # GT별 후보 리스트에 추가
            per_gt[gid].append({
                "iou": iou,
                "box": boxes[pidx].copy(),  # 정규화된 박스 좌표
                "crop": (float(x1), float(y1), float(x2), float(y2)),  # 픽셀 좌표
                "emb": embs[pidx].clone(),  # 임베딩
                "route": "full"  # 전체 이미지에서 검출됨을 표시
            })

    # ========= 타일링 처리 (TILE_EXCEPT에 포함되지 않은 클래스만) =========
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
                # 타일 추출
                tile = orig.crop((x0, y0, x0 + PATCH, y0 + PATCH))
                tboxes, tembs = infer(tile)
                if tboxes.shape[0] == 0:
                    continue
                
                # ========= 타일 좌표를 전체 이미지 좌표로 변환 =========
                px = np.stack([
                    tboxes[:, 0] * PATCH + x0, tboxes[:, 1] * PATCH + y0,
                    tboxes[:, 2] * PATCH + x0, tboxes[:, 3] * PATCH + y0
                ], axis=1)

                # 전체 이미지 기준으로 정규화
                gboxes = px / np.array([W, H, W, H], np.float32)

                # ========= 코어 영역 필터링 =========
                # 타일 경계 근처의 검출을 제거하여 중복 방지
                cx = (tboxes[:, 0] + tboxes[:, 2]) * 0.5 * PATCH  # 타일 내 중심 x
                cy = (tboxes[:, 1] + tboxes[:, 3]) * 0.5 * PATCH  # 타일 내 중심 y
                in_core = (cx >= CORE) & (cx <= PATCH - CORE) & (cy >= CORE) & (cy <= PATCH - CORE)
                if not in_core.any():
                    continue

                # ========= 타일 검출 결과와 GT 매칭 =========
                ious = iou_mat(gboxes, gt_xyxy)
                max_i = ious.argmax(1)
                max_v = ious.max(1)
                
                for pidx in range(gboxes.shape[0]):
                    if not in_core[pidx]:
                        continue  # 코어 영역 밖이면 스킵

                    gid = int(max_i[pidx])  # 매칭된 GT ID
                    iou = float(max_v[pidx])  # IoU 값

                    if iou < TAU:
                        continue  # 임계값 미만이면 스킵

                    x1, y1, x2, y2 = px[pidx]
                    per_gt[gid].append({
                        "iou": iou,
                        "box": gboxes[pidx].copy(),  # 정규화된 박스 좌표
                        "crop": (float(x1), float(y1), float(x2), float(y2)),  # 픽셀 좌표
                        "emb": tembs[pidx].clone(),  # 임베딩
                        "route": "tile"  # 타일에서 검출됨을 표시
                    })

    # ========= 선택 및 저장 (≤ MAX_PER_GT, 중복 제거) =========
    cls_dir = OUT_CROPS / class_name  # 클래스별 크롭 저장 디렉토리
    cls_dir.mkdir(parents=True, exist_ok=True)
    base = img_p.stem  # 이미지 파일명 (확장자 제외)
    pos_embs, pos_meta = store[class_name]["embs"], store[class_name]["meta"]

    # 각 GT에 대해 후보 선택 및 저장
    for gid, cand in per_gt.items():
        if not cand:
            continue  # 후보가 없으면 스킵

        # IoU와 면적을 기준으로 내림차순 정렬 (높은 IoU와 큰 면적 우선)
        cand.sort(key=lambda d: (d["iou"], (d["crop"][2] - d["crop"][0]) * (d["crop"][3] - d["crop"][1])), reverse=True)

        # 중복 제거 및 최대 개수 제한
        picks = []
        taken = []
        for it in cand:
            # 이미 선택된 박스와의 IoU가 임계값 이상이면 중복으로 간주하여 스킵
            if any(iou_one(it["box"], t) >= DEDUP_IOU for t in taken):
                continue

            picks.append(it)
            taken.append(it["box"])
            if len(picks) >= MAX_PER_GT:
                break  # 최대 개수에 도달하면 중단

        # 선택된 후보들을 저장
        for k, it in enumerate(picks):
            x1, y1, x2, y2 = it["crop"]
            cpath = cls_dir / f"{base}_gt{gid}_pos_{k}.jpg"
            
            # EXIF 정보가 수정된 원본 이미지에서 크롭 생성 및 저장
            orig.crop((x1, y1, x2, y2)).save(cpath)
            
            # 임베딩과 메타데이터 저장
            pos_embs.append(it["emb"])
            pos_meta.append({
                "image_path": str(img_p),  # 원본 이미지 경로
                "class": class_name,       # 클래스 이름
                "gt_id": int(gid),         # GT ID
                "role": "pos",             # 역할 (양성 샘플)
                "iou": float(it["iou"]),   # IoU 값
                "box_norm": [float(v) for v in it["box"].tolist()],  # 정규화된 박스 좌표
                "crop_path": str(cpath),   # 크롭 이미지 경로
                "route": it["route"],      # 검출 경로 (full/tile)
            })


def main():
    """
    메인 함수 - 모든 클래스에 대해 임베딩 추출 수행
    """
    # ========= 출력 디렉토리 생성 =========
    OUT_IDX.mkdir(parents=True, exist_ok=True)   # 임베딩 인덱스 저장 디렉토리
    OUT_CROPS.mkdir(parents=True, exist_ok=True) # 크롭 이미지 저장 디렉토리
    store: Dict[str, Dict[str, list]] = {}       # 클래스별 임베딩과 메타데이터 저장

    # ========= 클래스별 저장소 초기화 =========
    # images와 labels 디렉토리가 모두 존재하는 클래스만 처리
    for sub in sorted(SRC_ROOT.iterdir()):
        if sub.is_dir() and (sub / "images").is_dir() and (sub / "labels").is_dir():
            store[sub.name] = {"embs": [], "meta": []}

    # ========= 각 클래스별 처리 =========
    for sub in sorted(SRC_ROOT.iterdir()):
        if not sub.is_dir():
            continue

        if not ((sub / "images").is_dir() and (sub / "labels").is_dir()):
            continue

        cname = sub.name
        # group_of_trees를 제외한 모든 클래스에 대해 타일링 수행
        do_tiles = (cname.lower() not in TILE_EXCEPT)

        # ========= 이미지 파일 수집 =========
        imgs = []
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.tif", "*.tiff"):
            imgs += glob.glob(str((sub / "images") / ext))

        imgs = sorted(map(Path, imgs))
        
        # ========= 각 이미지에 대해 임베딩 추출 =========
        for p in imgs:
            lbl = sub / "labels" / (p.stem + ".txt")
            if lbl.exists():
                mine_image(p, lbl, cname, do_tiles, store)

        # ========= 클래스별 임베딩 팩 저장 =========
        embs = store[cname]["embs"]
        meta = store[cname]["meta"]
        if embs:
            torch.save({
                "embeddings": torch.stack(embs).to(torch.float32),  # 임베딩 스택
                "metadata": meta,                                    # 메타데이터 리스트
                "model_id": MODEL_NAME,                             # 사용된 모델 ID
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),     # 생성 시간
                "class": cname,                                     # 클래스 이름
                "params": {                                         # 사용된 파라미터
                    "iou_thr": TAU, 
                    "max_per_gt": MAX_PER_GT,
                    "tiling_except": list(TILE_EXCEPT), 
                    "patch": PATCH, 
                    "overlap": OVERLAP, 
                    "core": CORE
                }
            }, OUT_IDX / f"pos_{cname}.pt")
        print(f"[{cname}] pos={len(embs)}")

    print("Done.")


main()