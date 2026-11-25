"""Draw the projected 2D labels on a few images -> results/label_check/ (LOOK at them!)."""
import random
from pathlib import Path

import cv2

out = Path("results/label_check"); out.mkdir(parents=True, exist_ok=True)
imgs = sorted(Path("data/opv2v/images/train").glob("*.jpg")) + sorted(Path("data/opv2v/images/val").glob("*.jpg"))
random.Random(0).shuffle(imgs)
for p in imgs[:12]:
    img = cv2.imread(str(p)); H, W = img.shape[:2]
    lp = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
    for line in (lp.read_text().split("\n") if lp.exists() else []):
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        cv2.rectangle(img, (int((cx - w / 2) * W), int((cy - h / 2) * H)),
                      (int((cx + w / 2) * W), int((cy + h / 2) * H)), (0, 255, 0), 2)
    cv2.imwrite(str(out / p.name), img)
print("Saved", out, "- boxes must sit on the cars; if shifted, tell me.")
