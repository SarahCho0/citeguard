"""문자열 유사도 — 편집거리(Levenshtein distance) 기반.

수학적 배경
-----------
편집거리 d(a, b)는 문자열 a를 b로 바꾸는 데 필요한 최소 편집 연산
(삽입·삭제·치환) 횟수로, 문자열 집합 위의 **거리함수(metric)** 다:

  1. d(a, b) ≥ 0,  d(a, b) = 0 ⟺ a = b   (비퇴화성)
  2. d(a, b) = d(b, a)                     (대칭성 — 편집 연산은 가역)
  3. d(a, c) ≤ d(a, b) + d(b, c)           (삼각부등식 — a→b, b→c 편집
                                            스크립트를 이어붙이면 a→c가 되므로)

거리공간의 구조가 주는 실용적 결과:
  · 길이 하한:  |len(a) − len(b)| ≤ d(a, b)
    (편집 1회는 길이를 최대 1 바꾸므로) → 길이 차가 이미 허용치를 넘는
    비교는 계산 없이 제외해도 안전하다.
  · DP 행 최솟값의 단조성: Wagner–Fischer 표에서 i행의 최솟값은
    이후 행에서 감소하지 않는다 → 행 최솟값이 상한(cap)을 넘는 순간
    계산을 중단해도 "d > cap"이라는 판정은 정확하다.

이 두 성질 덕분에 아래 cap 기반 조기 중단(branch-and-bound)은
**휴리스틱이 아니라 결과가 보존되는 최적화**다 — 검증 계층의
결정성(determinism)을 깨지 않는다.

알고리즘: Wagner–Fischer 동적계획법.
  d[i][j] = a[:i]와 b[:j] 사이의 편집거리
  d[i][j] = min( d[i-1][j] + 1,            # a[i] 삭제
                 d[i][j-1] + 1,            # b[j] 삽입
                 d[i-1][j-1] + (a[i]≠b[j]) # 치환 또는 일치 )
  공간 O(min(n,m)) — 직전 행만 유지.
"""

from __future__ import annotations


def levenshtein(a: str, b: str, cap: int | None = None) -> int:
    """a, b의 편집거리. cap이 주어지면 d > cap일 때 cap+1을 반환(조기 중단).

    cap+1 반환은 "정확히 얼마인지는 모르지만 cap을 초과한다"는 뜻이며,
    임계값 판정(d ≤ cap 인가?)에는 손실이 없다.
    """
    if a == b:
        return 0

    # 길이 하한 |len(a)-len(b)| ≤ d 로 계산 자체를 건너뛴다
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1

    # 공간 절약: 짧은 쪽을 열(column)로
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        row_min = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr[j] = min(
                prev[j] + 1,      # 삭제
                curr[j - 1] + 1,  # 삽입
                prev[j - 1] + cost,  # 치환/일치
            )
            if curr[j] < row_min:
                row_min = curr[j]
        # 행 최솟값은 이후 행에서 감소하지 않으므로(단조성) 안전한 중단점
        if cap is not None and row_min > cap:
            return cap + 1
        prev = curr
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """정규화 유사도 = 1 − d(a,b) / max(|a|,|b|) ∈ [0, 1].

    max 길이로 나누므로 d의 상한(긴 쪽 길이)에 대한 비율이 되어
    항상 [0,1] 구간에 들어온다. 1.0 ⟺ 완전 일치.
    """
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))
