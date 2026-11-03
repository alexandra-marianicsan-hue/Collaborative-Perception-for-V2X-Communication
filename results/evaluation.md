# Evaluation

Detector (ONNX, val set): mAP50 = 0.290, mAP50-95 = 0.153

Images: 4 | IoU>=0.5 | overlap between views: 20%

| setup | precision | recall | F1 |
|---|---|---|---|
| full image | 0.188 | 0.115 | 0.143 |
| vehicle A only | 0.462 | 0.231 | 0.308 |
| vehicle B only | 0.200 | 0.077 | 0.111 |
| fused A+B | 0.318 | 0.269 | 0.292 |
