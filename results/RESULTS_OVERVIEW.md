# 결과 요약

## 주요 결과 표

| Domain | Base | 최고 Domain F1 Run | Domain F1 | Gain | Weighted General Acc | Weighted Forgetting |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| keboola_docs | keboola_base | all_r16_seed42 | 0.5208 | 0.1814 | 0.7498 | -0.0008 |
| techqa | techqa_base | techqa_lower_r8_seed42 | 0.3050 | 0.1043 | 0.7664 | -0.0174 |

## 해석

- 핵심 해석은 raw metric 기준으로 합니다.
- composite score는 후보를 빠르게 정렬하기 위한 보조 지표입니다.
- Keboola 파일럿에서는 `all_r16_seed42`가 최고였고, TechQA 확장에서는 `techqa_lower_r8_seed42`가 최고였습니다.
