"""[3일차] 중복/유사 이미지 검사 후 Train/Validation/Test 분리 (70/15/15).

- dHash(64bit)로 해밍 거리 <= --dup-thresh 인 이미지끼리 같은 그룹으로 묶고,
  같은 그룹은 반드시 같은 분할에 들어가게 한다 (계획서 4-2).
- 최다 득표 클래스(hard) 기준으로 층화(stratify)해서 분할별 클래스 비율을 맞춘다.
- <data-dir>/celeb_soft_labels.csv (prepare_celeb.py 결과)가 있으면 함께 합쳐서 하나의 데이터셋으로 나눈다.

실행 예:  python training/split_dataset.py --data-dir data
출력: <data-dir>/dataset.csv  (soft_labels.csv + split 열)
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from common import DEFAULT_DATA_DIR


def dhash(path, size=8):
    g = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--dup-thresh", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-labels", nargs="*", default=None,
                    help="함께 합칠 라벨 파일 (기본: celeb_soft_labels.csv 가 있으면 자동 포함)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    with open(data_dir / "soft_labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"FairFace 라벨: {len(rows)}장")
    extras = args.extra_labels
    if extras is None:
        auto = data_dir / "celeb_soft_labels.csv"
        extras = [str(auto)] if auto.exists() else []
    for path in extras:
        with open(path, encoding="utf-8") as f:
            more = list(csv.DictReader(f))
        print(f"추가 라벨: {Path(path).name} {len(more)}장")
        rows += more
    hashes = [dhash(data_dir / "images" / r["image"]) for r in rows]

    uf = UnionFind(len(rows))
    dup_pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= args.dup_thresh:
                uf.union(i, j)
                dup_pairs += 1
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    print(f"유사 이미지 쌍 {dup_pairs}개 -> 그룹 {len(groups)}개 (이미지 {len(rows)}장)")

    # 클래스별로 그룹을 섞어 비율대로 배정
    by_class = defaultdict(list)
    for g in groups.values():
        by_class[rows[g[0]]["hard"]].append(g)
    rng = random.Random(args.seed)
    split_of = {}
    names = ["train", "val", "test"]
    for cls, gs in by_class.items():
        rng.shuffle(gs)
        total = sum(len(g) for g in gs)
        bounds = [args.ratios[0] * total, (args.ratios[0] + args.ratios[1]) * total]
        acc = 0
        for g in gs:
            s = 0 if acc < bounds[0] else (1 if acc < bounds[1] else 2)
            for i in g:
                split_of[i] = names[s]
            acc += len(g)

    out = data_dir / "dataset.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["split"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "split": split_of[i]})

    print("분할 결과 (클래스별 장수):")
    for s in names:
        cnt = defaultdict(int)
        for i, r in enumerate(rows):
            if split_of[i] == s:
                cnt[r["hard"]] += 1
        print(f"  {s:5s} {sum(cnt.values()):5d}  {dict(sorted(cnt.items()))}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
