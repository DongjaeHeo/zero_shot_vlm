"""
클래스 그룹 정의 모듈 - Zero-shot 객체 검출 시스템의 클래스 그룹화 설정

이 모듈은 RELAXED 평가 모드를 위한 클래스 그룹 정의를 제공합니다.
STRICT 모드에서는 정확한 클래스 매칭을 요구하지만, RELAXED 모드에서는
같은 그룹 내의 클래스들을 서로 매칭 가능한 것으로 간주합니다.

예를 들어, "house", "factory", "shed"는 모두 "building" 그룹에 속하므로
RELAXED 모드에서는 서로 매칭될 수 있습니다.

핵심 특징:
- 클래스별 세분화된 평가 (STRICT)
- 그룹별 유연한 평가 (RELAXED)
- 도메인별 클래스 그룹화
"""

from typing import Dict, List, Tuple, Any, Iterable, Union


# RELAXED 평가를 위한 클래스 그룹 정의 (pos_*.pt 파일의 'class' 필드와 일치해야 함)
CLASS_GROUPS_RAW: Dict[str, List[str]] = {
    "building":   ["house", "factory", "shed"],  # 건물 관련 클래스들
    "vegetation": ["single_tree", "group_of_trees"],  # 식생 관련 클래스들
    "burial":     ["Mound_with_Headstone", "Headstone_without_Mound"],  # 묘지 관련 클래스들
    "polytunnel": ["polytunnel_with_cover", "polytunnel_no_cover"],  # 비닐하우스 관련 클래스들
    "field":      ["with_crop", "without_crop"],  # 농지 관련 클래스들
}

# 그룹별 ID 매핑 (RELAXED 평가에서 사용)
CLASS_GROUPS_ID: Dict[str, int] = {
    "burial": 0,      # 묘지 그룹
    "vegetation": 1,  # 식생 그룹
    "polytunnel": 2,  # 비닐하우스 그룹
    "building": 3,    # 건물 그룹
    "field": 4,       # 농지 그룹
}