"""텍스트 정규화 유틸.

인용 검증에서 "같은 문장인데 표기만 다른" 경우(전각/반각, 따옴표 종류,
공백 개수, 대소문자)를 흡수하기 위한 전처리 계층.
LLM이 인용문을 옮겨 적을 때 생기는 무해한 변형을 여기서 걸러내고,
그 이후 비교는 전부 결정적(deterministic) 문자열 대조로 수행한다.
"""

from __future__ import annotations

import re
import unicodedata

# 스마트 따옴표·전각 구두점 → 일반 문자로 통일
_QUOTE_MAP = str.maketrans({
    "“": '"', "”": '"',  # “ ”
    "‘": "'", "’": "'",  # ‘ ’
    "–": "-", "—": "-",  # – —
    "·": " ",                 # 가운뎃점(·)은 공백 취급
})

_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")


def normalize(text: str) -> str:
    """비교용 정규형으로 변환한다.

    NFKC 정규화(전각→반각 등) → 특수 따옴표 통일 → 제로폭 문자 제거
    → 소문자화 → 연속 공백을 단일 공백으로 축약 → 양끝 공백 제거.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_MAP)
    text = _ZERO_WIDTH.sub("", text)
    text = text.lower()
    text = _WS.sub(" ", text)
    return text.strip()


def squash(text: str) -> str:
    """정규화 후 공백까지 모두 제거한 형태.

    한국어 검색 질의처럼 띄어쓰기가 불규칙한 텍스트를
    char n-gram 매칭할 때 사용한다.
    """
    return normalize(text).replace(" ", "")
