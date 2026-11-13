"""[3일차] 한국 연예인 데이터셋 정리: 사람 단위 중복 제거 + FairFace와 같은 방식으로 얼굴 크롭 + soft label 변환.

입력 폴더 구조 (구글 드라이브에 업로드):
    <src>/dog/001_이름.jpg, <src>/cat/002_이름.jpg, ...   (폴더 이름 = 동물상)
    - 파일 이름의 '번호_이름' 에서 이름이 같으면 같은 사람으로 본다.
    - 이름이 없는 파일(예: horse_001.jpg)은 각각 다른 사람으로 본다.

처리:
  1) 같은 사람의 사진이 여러 장이면 1장만 남긴다 (얼굴이 가장 크고 정면에 가까운 사진).
  2) 한 사람이 여러 동물상 폴더에 있으면, 해당 동물상들에 같은 비율을 준다 (soft label).
     예) dog 와 rabbit 폴더에 모두 있으면 -> 강아지 0.5, 토끼 0.5
  3) FairFace 와 똑같은 MediaPipe 정렬·크롭(facecrop.py)으로 224x224 저장.
  4) 사람 이름은 어디에도 저장하지 않는다. 이미지 이름은 kc_00001.jpg 형식의 ID만 사용.

실행 예:
    python training/prepare_celeb.py --data-dir $DATA --src $DATA/celeb_raw

출력:
    <data-dir>/images/kc_00001.jpg ...            (FairFace 사진과 같은 폴더)
    <data-dir>/celeb_soft_labels.csv              (soft_labels.csv 와 같은 형식)
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import CLASSES, DEFAULT_DATA_DIR
from facecrop import FaceCropper, crop_params, warp_face

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FOLDER_ALIAS = {"dinosaur": "dino"}   # 폴더 이름이 코드의 클래스 이름과 다를 때


def person_key(path: Path) -> str:
    m = re.match(r"^\d+_(.+)$", path.stem)
    return ("name:" + m.group(1).strip()) if m else ("file:" + str(path))


def face_quality(lm: np.ndarray) -> float:
    """얼굴이 클수록, 정면에 가까울수록 높은 점수"""
    width = lm[:, 0].max() - lm[:, 0].min()
    le, re_ = lm[[33, 133]].mean(axis=0), lm[[362, 263]].mean(axis=0)
    e = re_ - le
    yaw = abs(np.dot(lm[1] - (le + re_) / 2, e) / (np.dot(e, e) + 1e-6))
    return width * max(0.0, 1.0 - 2 * yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--src", default=None, help="기본: <data-dir>/celeb_raw")
    ap.add_argument("--landmarker", default="face_landmarker.task")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    src = Path(args.src) if args.src else data_dir / "celeb_raw"
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1) 사람별로 사진과 동물상 모으기 (이름은 메모리에서만 사용)
    photos = defaultdict(list)     # person -> [path]
    classes = defaultdict(set)     # person -> {class}
    skipped = 0
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        cls = FOLDER_ALIAS.get(d.name.lower(), d.name.lower())
        if cls not in CLASSES:
            print(f"  건너뜀: '{d.name}' 폴더는 클래스 목록에 없음")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            k = person_key(f)
            photos[k].append(f)
            classes[k].add(cls)
    n_files = sum(len(v) for v in photos.values())
    print(f"사진 {n_files}장, 사람 {len(photos)}명 "
          f"(여러 동물상에 속한 사람 {sum(len(c) > 1 for c in classes.values())}명)")

    # 2) 사람마다 가장 좋은 사진 1장 선택 + 크롭
    cropper = FaceCropper(args.landmarker)
    results = []   # (face, soft vector, n_classes)
    no_face = 0
    for k in sorted(photos, key=lambda x: str(photos[x][0])):
        best = None
        for f in photos[k]:
            try:
                img = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
            except Exception:
                skipped += 1
                continue
            lm = cropper.landmarks(img)
            if lm is None:
                continue
            q = face_quality(lm)
            if best is None or q > best[0]:
                best = (q, img, lm)
        if best is None:
            no_face += 1
            continue
        affine, _ = crop_params(best[2])
        face = warp_face(best[1], affine)
        cs = classes[k]
        p = [1.0 / len(cs) if c in cs else 0.0 for c in CLASSES]
        results.append((face, p, len(cs)))

    # 3) 저장 (ID만 사용, 이름 저장 안 함)
    out = data_dir / "celeb_soft_labels.csv"
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["image"] + [f"p_{c}" for c in CLASSES] + ["n_votes", "hard"])
        for i, (face, p, n) in enumerate(results, start=1):
            name = f"kc_{i:05d}.jpg"
            Image.fromarray(face).save(img_dir / name, quality=95)
            w.writerow([name] + [f"{x:.4f}" for x in p] + [n, CLASSES[int(np.argmax(p))]])

    dist = defaultdict(float)
    for _, p, _ in results:
        for c, x in zip(CLASSES, p):
            dist[c] += x
    print(f"완료: {len(results)}명(1인 1장) 저장 -> {img_dir}  (얼굴 검출 실패 {no_face}명, 읽기 실패 {skipped}장)")
    print("동물상별 합계(soft):", {c: round(dist[c], 1) for c in CLASSES})
    print(f"라벨 저장: {out}")


if __name__ == "__main__":
    main()
