# 요약

- summary 기본 설정에서 smoke/mini run 포함 여부: 제외
- 집계 대상 run 수: 20
- domain 수: 2
- composite score는 보조 지표이며, raw metric과 domain별 base 대비 변화량을 우선 해석합니다.

## Domain Base

- keboola_docs / keboola_base: domain F1 0.3394, weighted general accuracy 0.7490, WikiText PPL 10.8035
- techqa / techqa_base: domain F1 0.2007, weighted general accuracy 0.7490, WikiText PPL 10.8035

## Domain별 핵심 관찰

- keboola_docs 최고 domain F1: all_r16_seed42 (F1 0.5208, gain 0.1814)
- keboola_docs 최고 composite: all_r16_seed42 (score 0.8998, F1 0.5208, forgetting -0.0008)
- keboola_docs 최소 weighted forgetting: upper_r4_seed42 (forgetting -0.0114, weighted general accuracy 0.7604)
- techqa 최고 domain F1: techqa_lower_r8_seed42 (F1 0.3050, gain 0.1043)
- techqa 최고 composite: techqa_lower_r8_seed42 (score 0.8494, F1 0.3050, forgetting -0.0174)
- techqa 최소 weighted forgetting: techqa_all_r8_seed42 (forgetting -0.0270, weighted general accuracy 0.7760)

## Adapter Run

- keboola_docs / all_r16_seed42: placement all, rank 16, domain F1 0.5208, weighted general accuracy 0.7498, WikiText PPL 10.6009, gain 0.1814, forgetting -0.0008, composite 0.8998
- keboola_docs / all_r8_seed42: placement all, rank 8, domain F1 0.5200, weighted general accuracy 0.7291, WikiText PPL 10.5769, gain 0.1806, forgetting 0.0200, composite 0.7298
- keboola_docs / all_r4_seed42: placement all, rank 4, domain F1 0.5167, weighted general accuracy 0.7379, WikiText PPL 10.5999, gain 0.1773, forgetting 0.0112, composite 0.7199
- keboola_docs / upper_r8_seed42: placement upper, rank 8, domain F1 0.5005, weighted general accuracy 0.7533, WikiText PPL 10.6280, gain 0.1611, forgetting -0.0043, composite 0.4954
- keboola_docs / lower_r4_seed42: placement lower, rank 4, domain F1 0.5054, weighted general accuracy 0.7469, WikiText PPL 10.7309, gain 0.1660, forgetting 0.0022, composite 0.4821
- keboola_docs / lower_r8_seed42: placement lower, rank 8, domain F1 0.5026, weighted general accuracy 0.7451, WikiText PPL 10.7431, gain 0.1632, forgetting 0.0039, composite 0.4025
- keboola_docs / lower_r16_seed42: placement lower, rank 16, domain F1 0.5104, weighted general accuracy 0.7234, WikiText PPL 10.7331, gain 0.1710, forgetting 0.0256, composite 0.3918
- keboola_docs / upper_r16_seed42: placement upper, rank 16, domain F1 0.4985, weighted general accuracy 0.7443, WikiText PPL 10.6349, gain 0.1591, forgetting 0.0047, composite 0.3763
- keboola_docs / upper_r4_seed42: placement upper, rank 4, domain F1 0.4916, weighted general accuracy 0.7604, WikiText PPL 10.6379, gain 0.1522, forgetting -0.0114, composite 0.3633
- techqa / techqa_lower_r8_seed42: placement lower, rank 8, domain F1 0.3050, weighted general accuracy 0.7664, WikiText PPL 10.7553, gain 0.1043, forgetting -0.0174, composite 0.8494
- techqa / techqa_all_r8_seed42: placement all, rank 8, domain F1 0.2707, weighted general accuracy 0.7760, WikiText PPL 10.6081, gain 0.0700, forgetting -0.0270, composite 0.7847
- techqa / techqa_all_r4_seed42: placement all, rank 4, domain F1 0.2523, weighted general accuracy 0.7682, WikiText PPL 10.6556, gain 0.0516, forgetting -0.0192, composite 0.6257
- techqa / techqa_all_r16_seed42: placement all, rank 16, domain F1 0.2256, weighted general accuracy 0.7565, WikiText PPL 10.4953, gain 0.0248, forgetting -0.0074, composite 0.4727
- techqa / techqa_lower_r16_seed42: placement lower, rank 16, domain F1 0.2096, weighted general accuracy 0.7711, WikiText PPL 10.7480, gain 0.0088, forgetting -0.0221, composite 0.3922
- techqa / techqa_upper_r8_seed42: placement upper, rank 8, domain F1 0.2111, weighted general accuracy 0.7406, WikiText PPL 10.6368, gain 0.0103, forgetting 0.0084, composite 0.2511
- techqa / techqa_lower_r4_seed42: placement lower, rank 4, domain F1 0.1878, weighted general accuracy 0.7637, WikiText PPL 10.7794, gain -0.0129, forgetting -0.0147, composite 0.2241
- techqa / techqa_upper_r4_seed42: placement upper, rank 4, domain F1 0.1959, weighted general accuracy 0.7416, WikiText PPL 10.6711, gain -0.0048, forgetting 0.0074, composite 0.1673
- techqa / techqa_upper_r16_seed42: placement upper, rank 16, domain F1 0.1909, weighted general accuracy 0.7273, WikiText PPL 10.5900, gain -0.0099, forgetting 0.0217, composite 0.0822
