"""
텍스트 기반 객체 탐지 시스템
=====================================
이 모듈은 자연어 프롬프트를 사용한 객체 탐지의 초기 구현체입니다.

주요 기능:
- 다중 프롬프트를 사용한 객체 탐지
- EXIF 메타데이터 처리
- NMS를 통한 중복 탐지 제거
- 시각화 및 결과 저장
"""

import os
import json
import pandas as pd
from pathlib import Path

import torch
from torchvision.ops import nms
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw, ImageFont, ExifTags

def load_image_with_correct_orientation(image_path):
    """
    이미지를 올바른 방향으로 로드하는 함수
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        PIL.Image: RGB로 변환된 올바른 방향의 이미지
        
    Note:
        스마트폰이나 카메라로 촬영한 이미지는 EXIF 정보에 방향이 저장되어 있음
        이 정보를 읽어서 이미지를 올바른 방향으로 회전시킴
    """
    image = Image.open(image_path)

    # EXIF Orientation 처리 - 카메라/스마트폰 촬영 이미지의 방향 정보 읽기
    try:
        exif = image._getexif()
        if exif is not None:
            # EXIF 태그에서 Orientation 정보 찾기
            orientation_tag = next(
                k for k, v in ExifTags.TAGS.items() if v == 'Orientation'
            )
            orientation = exif.get(orientation_tag, None)

            # 방향에 따른 이미지 회전
            if orientation == 3:      # 180도 회전
                image = image.rotate(180, expand=True)
            elif orientation == 6:    # 270도 회전 (90도 반시계방향)
                image = image.rotate(270, expand=True)
            elif orientation == 8:    # 90도 회전 (90도 시계방향)
                image = image.rotate(90, expand=True)
    except Exception as e:
        pass  # EXIF 정보가 없거나 읽을 수 없으면 무시

    return image.convert("RGB")  # RGB 채널로 변환

# ==================== 시스템 설정 ====================
IMAGE_DIR = Path("dataset/images")                    # 입력 이미지 디렉토리
OUTPUT_DIR = Path("results/prediction_with_multiple_text")  # 결과 저장 디렉토리
TARGET_SIZE = 1008                                    # OWLv2 최적 입력 크기 (14×72 패치)
TEXT_THRESHOLD = 0.3                                  # 텍스트-이미지 매칭 신뢰도 임계값
IOU_THRESHOLD = 0.5                                   # NMS IoU 임계값

# 디바이스 자동 선택 (MPS > CUDA > CPU 순서)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")                      # Apple Silicon GPU
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # NVIDIA GPU 또는 CPU

# ==================== 프롬프트 전략 ====================
# 각 객체 클래스별로 여러 개의 텍스트 프롬프트를 정의
# 다양한 표현을 사용하여 탐지 성능을 높이려 했으나, 
# 실제로는 프롬프트 품질에 따른 성능 편차가 크다는 한계가 발견됨
QUERY_PROMPTS = {
    "Tomb": [                    # 무덤/고분 클래스
        "Korean burial mound",           # 한국식 고분
        "ancient tomb site",             # 고대 무덤 유적
        "grave mound",                   # 무덤 언덕
        "traditional burial mound",      # 전통적인 매장 언덕
        "burial site covered with grass" # 풀로 덮인 매장지
    ],
    "Tree": [                    # 나무 클래스
        "forest",                        # 숲
        "isolated tree canopy",          # 고립된 나무 수관
        "deciduous tree",                # 활엽수
        "single tall tree",              # 단일 큰 나무
        "mature oak tree"                # 성숙한 참나무
    
        # 실패한 프롬프트들 (주석 처리)
        # "A leafless tree",             # 잎이 없는 나무
        # "green leafed tree",           # 녹색 잎이 있는 나무
        # "plant"                        # 식물 (너무 일반적)
        # "A bare tree",                 # 벌거벗은 나무
        # "A tree with bare branches",   # 벌거벗은 가지가 있는 나무
    ],
    "Greenhouse": [
        "plastic greenhouse",
        "glass greenhouse structure",
        "hoop house greenhouse",
        "abandoned greenhouse frame",
        "greenhouse with plastic cover"
    ],
    "Building": [
        "single story building",
        "flat roof house",
        "residential building structure",
        "industrial shed building",
        "roofed concrete building"
    ],
    "Field": [
        "agricultural field",
        "plowed farmland",
        "cultivated crop field",
        "rice paddy field",
        "open grass field"
    ],
    "Container": [
        "shipping container",
        "cargo container",
        "metal storage container",
        "freight container box",
        "stacked container unit"
    ]
}

# ==================== 시각화 설정 ====================
# 각 클래스별로 고유한 색상 할당 (RGB 값)
COLOR_MAP = {
    "Tomb":       (0,   0, 255), # 파란색 - 무덤
    "Tree":       (0, 255,   0), # 초록색 - 나무
    "Greenhouse": (255, 0,   0), # 빨간색 - 온실
    "Building":   (0, 255, 255), # 청록색 - 건물
    "Field":      (255, 0, 255), # 자홍색 - 밭
    "Container":  (255, 255,   0), # 노란색 - 컨테이너
    "Vehicle":    (0, 0,   0), # 검은색 - 차량
}

# ==================== 모델 초기화 ====================
# OWLv2 모델과 프로세서 로드 (Google의 Open-Vocabulary Object Detection 모델)
print("OWLv2 모델 로딩 중...")
processor = Owlv2Processor.from_pretrained(
    "google/owlv2-large-patch14-ensemble",  # 14×14 패치, 72개 패치 = 1008×1008
    image_size=TARGET_SIZE
)
model = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-large-patch14-ensemble"
).to(DEVICE).eval()  # 평가 모드로 설정 (드롭아웃 등 비활성화)

# ==================== 출력 준비 ====================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 결과 저장 디렉토리 생성
font = ImageFont.load_default()                # 텍스트 표시용 폰트
csv_records = []                               # CSV 통계 데이터 저장용 리스트

# ==================== 메인 처리 루프 ====================
# 각 이미지에 대해 모든 클래스의 프롬프트를 적용하여 객체 탐지 수행
for img_path in IMAGE_DIR.glob("*"):
    # 지원하는 이미지 형식만 처리
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    print(f"\n▶ Processing {img_path.name} …")
    
    # 1. 이미지 로드 및 전처리
    orig = load_image_with_correct_orientation(img_path)  # EXIF 방향 보정
    img_resized = orig.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)  # 1008×1008 리사이징
    w, h = img_resized.size

    # 2. 탐지 결과 저장용 리스트 초기화
    all_boxes, all_scores, all_labels = [], [], []

    # 3. 각 클래스별로 프롬프트 적용
    for cls, prompts in QUERY_PROMPTS.items():
        # 4. 각 프롬프트에 대해 OWLv2 모델 추론 수행
        for prompt in prompts:
            # 텍스트와 이미지를 모델 입력 형태로 변환
            inputs = processor(
                text=[prompt],                    # 텍스트 프롬프트
                images=img_resized,               # 리사이징된 이미지
                return_tensors="pt"               # PyTorch 텐서로 반환
            )
            # GPU/CPU 디바이스로 데이터 이동
            inputs = {k: v.to(DEVICE) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # 5. 모델 추론 (그래디언트 계산 비활성화)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 6. 탐지 결과 후처리 (바운딩 박스, 신뢰도 점수 추출)
            results = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=torch.tensor([[h, w]], device=DEVICE),  # 원본 이미지 크기
                threshold=TEXT_THRESHOLD                              # 신뢰도 임계값 (0.3)
            )

            # 7. 탐지 결과를 전체 리스트에 추가
            for res in results:
                for box, score in zip(res["boxes"], res["scores"]):
                    all_boxes.append(box)      # 바운딩 박스 좌표
                    all_scores.append(score)   # 신뢰도 점수
                    all_labels.append(cls)     # 클래스 라벨

            # 빈 결과 예외 처리
            if not results or len(results[0]["boxes"]) == 0:
                csv_records.append({
                    "image_name": img_path.name,
                    "class": cls,
                    "prompt": prompt,
                    "box_count": 0,
                    "scores": ""
                })
                continue

            boxes = results[0]["boxes"]
            scores = results[0]["scores"]

            keep = nms(boxes, scores, IOU_THRESHOLD)
            if len(keep) == 0:
                csv_records.append({
                    "image_name": img_path.name,
                    "class": cls,
                    "prompt": prompt,
                    "box_count": 0,
                    "scores": ""
                })
                continue

            # 시각화
            draw_img = img_resized.copy()
            draw = ImageDraw.Draw(draw_img)
            color = COLOR_MAP.get(cls, (255, 255, 255))
            score_values = []

            for i in keep:
                box = boxes[i]
                score = float(scores[i])
                x1, y1, x2, y2 = box.tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                txt = f"{cls}: {score:.2f}"
                tx0, ty0, tx1, ty1 = draw.textbbox((x1, y1), txt, font=font)
                draw.rectangle([tx0, ty0, tx1, ty1], fill=color)
                draw.text((x1, y1), txt, fill=(255, 255, 255), font=font)
                score_values.append(f"{score:.4f}")

            # 저장 경로 설정
            safe_prompt = prompt.replace(" ", "_").replace("/", "_")
            subdir = OUTPUT_DIR / img_path.stem / cls
            subdir.mkdir(parents=True, exist_ok=True)
            save_name = f"{img_path.stem}_{cls}_{safe_prompt}.jpg"
            draw_img.save(subdir / save_name)

            # CSV 기록
            csv_records.append({
                "image_name": img_path.name,
                "class": cls,
                "prompt": prompt,
                "box_count": len(keep),
                "scores": ";".join(score_values)
            })

    # 8. 탐지 결과가 없는 경우 건너뛰기
    if not all_boxes:
        print(f"  [WARN] No detections for {img_path.name}")
        continue

    # 9. NMS (Non-Maximum Suppression) 적용 - 중복 탐지 제거
    boxes_t = torch.stack(all_boxes)      # 모든 바운딩 박스를 텐서로 변환
    scores_t = torch.stack(all_scores)    # 모든 신뢰도 점수를 텐서로 변환
    keep_all = nms(boxes_t, scores_t, IOU_THRESHOLD)  # IoU 0.5 임계값으로 NMS 적용

    # 10. 최종 결과 시각화 및 저장
    draw = ImageDraw.Draw(img_resized)
    records = []
    for i in keep_all:
        # 바운딩 박스 좌표 추출 및 경계 조정
        x1, y1, x2, y2 = boxes_t[i].tolist()
        x1, y1 = max(0, x1), max(0, y1)      # 최소값 0으로 제한
        x2, y2 = min(w, x2), min(h, y2)      # 최대값 이미지 크기로 제한
        
        label = all_labels[i]                # 클래스 라벨
        score = float(scores_t[i])           # 신뢰도 점수
        color = COLOR_MAP.get(label, (255, 255, 255))  # 클래스별 색상

        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 텍스트 배경 + 라벨 표시
        txt = f"{label}: {score:.2f}"
        tx0, ty0, tx1, ty1 = draw.textbbox((x1, y1), txt, font=font)
        draw.rectangle([tx0, ty0, tx1, ty1], fill=color)  # 텍스트 배경
        draw.text((x1, y1), txt, fill=(255, 255, 255), font=font)  # 흰색 텍스트

        # JSON 결과용 데이터 저장
        records.append({
            "label": label,
            "score": score,
            "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    # 11. 최종 결과 저장
    out_img = OUTPUT_DIR / img_path.name                    # 시각화된 이미지 저장
    out_json = OUTPUT_DIR / f"{img_path.stem}.json"        # JSON 형태의 탐지 결과 저장
    img_resized.save(out_img)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)

    print(f"  [OK] Saved → {out_img.name}, {out_json.name}")

# ==================== 전체 통계 저장 ====================
# 모든 이미지의 탐지 결과를 CSV 형태로 저장 (프롬프트별 성능 분석용)
df = pd.DataFrame(csv_records)
csv_path = OUTPUT_DIR / "result_prediction_with_multiple_text_stat.csv"
df.to_csv(csv_path, index=False)
print(f"\n✅ CSV saved: {csv_path}")



