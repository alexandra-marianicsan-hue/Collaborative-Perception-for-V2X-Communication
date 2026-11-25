
Association on 195 pairs, pose noise 0.0 m / 0.0 deg, detections=model
| method | P | R | F1 | recall on cut objects | count error | ms/pair |
|---|---|---|---|---|---|---|
| no association | 0.00 | 0.00 | 0.00 | 0.00 (n=87) | 90 | 0.0 |
| geo | 0.54 | 0.90 | 0.68 | 0.90 (n=87) | 60 | 5.2 |
| app | 0.10 | 0.42 | 0.16 | 0.41 (n=87) | 315 | 35.0 |
| kp | 0.13 | 0.59 | 0.21 | 0.57 (n=87) | 333 | 127.1 |
| geo+app | 0.53 | 0.90 | 0.67 | 0.90 (n=87) | 63 | 39.7 |
| geo+kp | 0.52 | 0.91 | 0.66 | 0.91 (n=87) | 67 | 125.8 |
| geo+app+kp | 0.50 | 0.91 | 0.65 | 0.91 (n=87) | 73 | 160.4 |

Association on 195 pairs, pose noise 0.0 m / 0.0 deg, detections=model
| method | P (verified) | R | F1 | wrong links | links to unannotated boxes | recall on cut objects | count error | ms/pair |
|---|---|---|---|---|---|---|---|---|
| no association | 0.00 | 0.00 | 0.00 | 0 | 0 | 0.00 (n=87) | 90 | 0.0 |
| geo | 0.84 | 0.90 | 0.87 | 16 | 53 | 0.90 (n=87) | 17 | 5.2 |
| app | 0.14 | 0.42 | 0.21 | 235 | 118 | 0.41 (n=87) | 199 | 35.1 |
| kp | 0.17 | 0.59 | 0.27 | 256 | 114 | 0.57 (n=87) | 221 | 125.0 |
| geo+app | 0.84 | 0.90 | 0.87 | 16 | 56 | 0.90 (n=87) | 19 | 39.8 |
| geo+kp | 0.79 | 0.91 | 0.85 | 22 | 53 | 0.91 (n=87) | 20 | 126.5 |
| geo+app+kp | 0.77 | 0.91 | 0.83 | 25 | 56 | 0.91 (n=87) | 25 | 161.3 |
(P verified = correct / (correct + wrong), links to boxes without a ground-truth match are not judged; count error on GT-matched boxes only)
