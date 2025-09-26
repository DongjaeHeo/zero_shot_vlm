"""
Ground Truth 시각화 도구
========================
YOLO 형식의 라벨 파일을 읽어서 실제 정답 데이터를 시각화하는 도구입니다.
텍스트 기반 탐지 결과와 비교하여 성능을 평가할 때 사용됩니다.

주요 기능:
- YOLO 형식 라벨 파싱
- 클래스별 색상 구분 시각화
- 1008x1008 해상도로 통일
- 통계 정보 출력
"""

import os
from PIL import Image, ImageDraw, ImageFont, ExifTags
from collections import defaultdict

# ==================== 시스템 설정 ====================
IMAGE_DIR = "encumbrance_labels/encumbrance"           # 원본 이미지 디렉토리
LABEL_DIR = "encumbrance_labels/encumbrance_labels"    # YOLO 형식 라벨 디렉토리
OUTPUT_DIR = "encumbrance_labels/resized_GT"           # 시각화 결과 저장 디렉토리
TARGET_SIZE = 1008                                     # OWLv2 최적 입력 크기

# ==================== 클래스 정의 ====================
# YOLO 형식의 클래스 ID와 실제 클래스명 매핑
CLASS_NAMES = {
    0: "Tomb",        # 무덤/고분
    1: "Tree",        # 나무
    2: "Greenhouse",  # 온실
    3: "Building",    # 건물
    4: "Field",       # 밭
    5: "Container"    # 컨테이너
}

# ==================== 시각화 설정 ====================
# 각 클래스별 고유한 색상 할당 (RGB 값)
COLOR_MAP = {
    "Tomb":       (0,   0, 255),  # 파란색 - 무덤
    "Tree":       (0, 255,   0),  # 초록색 - 나무
    "Greenhouse": (255, 0,   0),  # 빨간색 - 온실
    "Building":   (0, 255, 255),  # 청록색 - 건물
    "Field":      (255, 0, 255),  # 자홍색 - 밭
    "Container":  (255, 255,   0), # 노란색 - 컨테이너
}

font = ImageFont.load_default()  # 텍스트 표시용 폰트


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


def process_image(image_path, label_path, output_path):
    """
    단일 이미지의 Ground Truth를 시각화하는 함수
    
    Args:
        image_path: 원본 이미지 파일 경로
        label_path: YOLO 형식 라벨 파일 경로
        output_path: 시각화 결과 저장 경로
        
    Note:
        YOLO 형식: class_id x_center y_center width height (모두 정규화된 값)
        이를 1008x1008 해상도로 변환하여 시각화
    """
    # 1. 이미지 로드 및 전처리
    image = load_image_with_correct_orientation(image_path)  # EXIF 방향 보정
    image_resized = image.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)  # 1008x1008 리사이징
    draw = ImageDraw.Draw(image_resized)  # 그리기 객체 생성

    class_counter = defaultdict(int)  # 클래스별 객체 개수 카운터

    # 2. 라벨 파일 존재 확인
    if not os.path.exists(label_path):
        print(f"⚠️ Label not found: {label_path}")
        return

    # 3. YOLO 형식 라벨 파일 읽기
    with open(label_path, "r") as f:
        lines = f.readlines()

    # 4. 각 라벨 라인 처리
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # YOLO 형식이 아닌 라벨은 무시

        # YOLO 형식 파싱: class_id x_center y_center width height
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # 정규화된 좌표를 픽셀 좌표로 변환
        x_center *= TARGET_SIZE
        y_center *= TARGET_SIZE
        width *= TARGET_SIZE
        height *= TARGET_SIZE

        # 중심점과 크기로부터 바운딩 박스 좌표 계산
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # 클래스 정보 및 색상 설정
        label = CLASS_NAMES.get(class_id, str(class_id))  # 클래스 ID를 이름으로 변환
        color = COLOR_MAP.get(label, (255, 255, 255))     # 클래스별 색상 할당

        # 5. 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 6. 클래스 라벨 텍스트 표시
        txt = label  # Ground Truth에는 신뢰도 점수가 없으므로 클래스명만 표시
        text_bbox = draw.textbbox((x1, y1), txt, font=font)
        draw.rectangle(text_bbox, fill=color)  # 텍스트 배경
        draw.text((x1, y1), txt, fill=(255, 255, 255), font=font)  # 흰색 텍스트

        class_counter[label] += 1  # 클래스별 카운트 증가

    # 7. 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_resized.save(output_path)
    print(f"✅ Saved: {output_path}")

    # 8. 클래스 통계 출력
    base_name = os.path.basename(image_path)
    if class_counter:
        print(f"▶ {base_name}: {dict(class_counter)}")  # 클래스별 객체 개수 출력
    else:
        print(f"▶ {base_name}: No objects found.")      # 객체가 없는 경우


def batch_process_all_images():
    """
    모든 이미지에 대해 Ground Truth 시각화를 일괄 처리하는 함수
    
    Note:
        IMAGE_DIR의 모든 이미지 파일을 처리하여
        각각에 대응하는 라벨 파일을 찾아 시각화 수행
    """
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 이미지 디렉토리의 모든 파일 처리
    for filename in os.listdir(IMAGE_DIR):
        # 지원하는 이미지 형식만 처리
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # 파일 경로 설정
        image_path = os.path.join(IMAGE_DIR, filename)                    # 원본 이미지
        label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")  # 라벨 파일
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + "_resized_GT.jpg")  # 출력 파일

        # 개별 이미지 처리
        process_image(image_path, label_path, output_path)


if __name__ == "__main__":
    print("Ground Truth 시각화 시작...")
    batch_process_all_images()
    print("Ground Truth 시각화 완료!")
