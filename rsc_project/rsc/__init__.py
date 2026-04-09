"""
RSC (Recursive Self-Consistency) 핵심 모듈 패키지

이 패키지는 RSC 알고리즘의 핵심 구성 요소를 포함합니다:
- generator: reasoning path 생성 (Stage 1 및 Stage t≥2의 refinement 프롬프트 생성 포함)
- aggregator: majority vote를 통한 답변 집계
- rsc_core: RSC 전체 오케스트레이션
"""

from .rsc_core import RSCEngine

__all__ = ['RSCEngine']

