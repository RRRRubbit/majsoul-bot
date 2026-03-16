"""
真实截图批量评估工具（无标注版本）。

功能：
1) 遍历截图目录，按手牌区域切牌并识别
2) 统计每张截图的置信度分布、>=阈值占比、unknown 占比
3) 输出总体汇总，辅助识别优化回归
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from majsoul_bot.vision.tile_recognizer import TileRecognizer


def _eval_once(recognizer: TileRecognizer, img, has_drawn_tile: bool) -> Dict:
    recognizer.recognize_hand(img, hand_count=13, has_drawn_tile=has_drawn_tile)
    details = recognizer.last_recognition_details or []

    if not details:
        return {
            "has_drawn_tile": has_drawn_tile,
            "tiles": 0,
            "accepted": 0,
            "unknown": 0,
            "scores": [],
            "recognized": [],
        }

    scores = [float(d.get("best_score", 0.0)) for d in details]
    recognized = [str(d.get("recognized_name", "")) for d in details]
    accepted = sum(1 for d in details if bool(d.get("accepted", False)))
    unknown = sum(1 for r in recognized if r.startswith("unknown_"))

    return {
        "has_drawn_tile": has_drawn_tile,
        "tiles": len(details),
        "accepted": accepted,
        "unknown": unknown,
        "scores": scores,
        "recognized": recognized,
    }


def _pick_better(a: Dict, b: Dict) -> Dict:
    # 优先 accepted 数，再看平均分
    if a["accepted"] != b["accepted"]:
        return a if a["accepted"] > b["accepted"] else b
    ma = mean(a["scores"]) if a["scores"] else 0.0
    mb = mean(b["scores"]) if b["scores"] else 0.0
    return a if ma >= mb else b


def _summary(scores: List[float], threshold: float) -> Dict[str, float]:
    if not scores:
        return {
            "avg": 0.0,
            "p50": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ge_threshold_ratio": 0.0,
        }
    ge = sum(1 for s in scores if s >= threshold)
    return {
        "avg": float(mean(scores)),
        "p50": float(median(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
        "ge_threshold_ratio": float(ge / len(scores)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="真实截图批量评估（置信度/unknown 统计）")
    p.add_argument("--input-dir", default="logs", help="截图目录")
    p.add_argument("--glob", default="*.png", help="截图匹配模式")
    p.add_argument("--templates-dir", default="templates/tiles", help="模板目录")
    p.add_argument("--threshold", type=float, default=0.75, help="识别阈值")
    p.add_argument("--nn-enabled", action="store_true", help="启用 NN 融合")
    p.add_argument("--nn-model-path", default="models/tile_ann.xml", help="NN 模型路径")
    p.add_argument("--nn-labels-path", default=None, help="NN 标签路径")
    p.add_argument("--output-json", default="logs/recognition_eval_report.json", help="输出报告路径")
    p.add_argument("--min-width", type=int, default=500, help="参与评估的最小截图宽度")
    p.add_argument("--min-height", type=int, default=300, help="参与评估的最小截图高度")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return 1

    image_candidates = sorted(input_dir.rglob(args.glob))
    image_candidates = [p for p in image_candidates if p.is_file()]
    if not image_candidates:
        print(f"[ERROR] 未找到截图: {input_dir} ({args.glob})")
        return 1

    images = []
    skipped_small = 0
    for pth in image_candidates:
        img = cv2.imread(str(pth))
        if img is None or img.size == 0:
            continue
        h, w = img.shape[:2]
        if w < int(args.min_width) or h < int(args.min_height):
            skipped_small += 1
            continue
        images.append(pth)

    if not images:
        print(
            f"[ERROR] 未找到满足尺寸条件的截图：{input_dir} ({args.glob}) | "
            f"min={args.min_width}x{args.min_height} | 候选={len(image_candidates)}"
        )
        return 1

    if skipped_small:
        print(
            f"[INFO] 已过滤小图 {skipped_small} 张（通常是单牌切图），"
            f"纳入评估 {len(images)} 张完整截图"
        )

    recognizer = TileRecognizer(
        templates_dir=args.templates_dir,
        threshold=float(args.threshold),
        nn_enabled=bool(args.nn_enabled),
        nn_model_path=args.nn_model_path,
        nn_labels_path=args.nn_labels_path,
    )

    per_image = []
    all_scores: List[float] = []
    total_tiles = 0
    total_unknown = 0

    for path in images:
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            continue

        res14 = _eval_once(recognizer, img, has_drawn_tile=True)
        res13 = _eval_once(recognizer, img, has_drawn_tile=False)
        best = _pick_better(res14, res13)

        scores = best["scores"]
        summ = _summary(scores, float(args.threshold))
        total_tiles += best["tiles"]
        total_unknown += best["unknown"]
        all_scores.extend(scores)

        row = {
            "file": str(path.as_posix()),
            "has_drawn_tile": bool(best["has_drawn_tile"]),
            "tiles": int(best["tiles"]),
            "accepted": int(best["accepted"]),
            "unknown": int(best["unknown"]),
            **summ,
        }
        per_image.append(row)
        print(
            f"[EVAL] {path.name} | tiles={row['tiles']} | accepted={row['accepted']} | "
            f"unknown={row['unknown']} | ge@{args.threshold:.2f}={row['ge_threshold_ratio']:.2%}"
        )

    overall = _summary(all_scores, float(args.threshold))
    overall.update(
        {
            "images": len(per_image),
            "tiles": int(total_tiles),
            "unknown": int(total_unknown),
            "unknown_ratio": float(total_unknown / total_tiles) if total_tiles else 0.0,
        }
    )

    report = {
        "config": {
            "input_dir": str(input_dir.as_posix()),
            "glob": args.glob,
            "min_width": int(args.min_width),
            "min_height": int(args.min_height),
            "templates_dir": args.templates_dir,
            "threshold": float(args.threshold),
            "nn_enabled": bool(args.nn_enabled),
            "nn_model_path": args.nn_model_path,
            "nn_labels_path": args.nn_labels_path,
        },
        "overall": overall,
        "per_image": per_image,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("真实截图批量评估完成")
    print(f"图片数: {overall['images']} | 牌数: {overall['tiles']}")
    print(f"平均置信度: {overall['avg']:.4f} | P50: {overall['p50']:.4f}")
    print(f">= {args.threshold:.2f} 占比: {overall['ge_threshold_ratio']:.2%}")
    print(f"unknown 占比: {overall['unknown_ratio']:.2%}")
    print(f"报告: {out.resolve()}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
