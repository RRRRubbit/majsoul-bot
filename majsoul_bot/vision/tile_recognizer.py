"""
麻将牌识别模块
支持多种识别方式：YOLO检测、模板匹配、神经网络分类
"""
import cv2
import numpy as np
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from loguru import logger

from ..game_logic.tile import Tile, TileType
from .regions import ScreenRegions, DEFAULT_REGIONS
from .tile_nn_classifier import TileNNClassifier
from .yolo_tile_detector import YOLOTileDetector


# 所有牌型标准命名（含赤宝 0m/0p/0s）
ALL_TILE_NAMES: List[str] = (
    ["0m"] + [f"{i}m" for i in range(1, 10)]   # 万子 0m(赤五)~9m
    + ["0p"] + [f"{i}p" for i in range(1, 10)] # 筒子 0p(赤五)~9p
    + ["0s"] + [f"{i}s" for i in range(1, 10)] # 索子 0s(赤五)~9s
    + [f"{i}z" for i in range(1, 8)]  # 字牌 1z~7z（东南西北白发中）
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_label(raw: str) -> Optional[str]:
    name = raw.strip().lower()
    if name in ALL_TILE_NAMES:
        return name
    return None


def _infer_label_from_path(path: Path) -> Optional[str]:
    parent_label = _normalize_label(path.parent.name)
    if parent_label:
        return parent_label

    stem = path.stem.lower()
    tokens = [tok for tok in re.split(r"[^0-9a-z]+", stem) if tok]
    for tok in tokens:
        label = _normalize_label(tok)
        if label:
            return label
        if len(tok) >= 2:
            label = _normalize_label(tok[:2])
            if label:
                return label

    m = re.search(r"(0[psm]|[1-9][psm]|[1-7]z)", stem)
    if m:
        return _normalize_label(m.group(1))
    return None


def _infer_label_from_stem(stem: str) -> Optional[str]:
    s = stem.lower()
    tokens = [tok for tok in re.split(r"[^0-9a-z]+", s) if tok]
    for tok in tokens:
        label = _normalize_label(tok)
        if label:
            return label
        if len(tok) >= 2:
            label = _normalize_label(tok[:2])
            if label:
                return label
    m = re.search(r"(0[psm]|[1-9][psm]|[1-7]z)", s)
    if m:
        return _normalize_label(m.group(1))
    return None


class TileRecognizer:
    """
    麻将牌识别器

    工作流程：
    1. 从 templates/tiles/ 加载各牌型的模板图像
    2. 截取手牌区域并按位置切割出单张牌图像
    3. 对每张牌使用 cv2.matchTemplate 与所有模板比对
    4. 返回识别结果（牌名 + 截图内像素坐标）

    若模板目录为空，则只返回手牌位置，供调试或手动配置使用。
    """

    def __init__(
        self,
        templates_dir: str = "templates/tiles",
        threshold: float = 0.75,
        regions: Optional[ScreenRegions] = None,
        nn_enabled: bool = True,
        nn_model_path: str = "models/tile_ann.xml",
        nn_labels_path: Optional[str] = None,
        nn_fusion_weight: float = 0.65,
        nn_min_confidence: float = 0.58,
        nn_top_k: int = 5,
        nn_priority: bool = False,
        enforce_mpsz_order: bool = True,
        yolo_enabled: bool = True,
        yolo_model_path: Optional[str] = None,
        yolo_conf_threshold: float = 0.5,
        yolo_priority: bool = True,
    ):
        """
        Args:
            templates_dir: 模板图片目录路径
            threshold: 模板匹配最低得分（0~1，越高越严格）
            regions: 屏幕区域配置，默认使用全局 DEFAULT_REGIONS
            yolo_enabled: 是否启用YOLO检测
            yolo_model_path: YOLO模型路径，None则使用默认路径
            yolo_conf_threshold: YOLO置信度阈值
            yolo_priority: YOLO优先模式（True时优先使用YOLO结果）
        """
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.regions = regions or DEFAULT_REGIONS
        # {tile_name: template_bgr}（兼容旧逻辑：保留每类第一张模板）
        self.templates: Dict[str, np.ndarray] = {}
        # {tile_name: [template_bgr, ...]}（新逻辑：支持每类多样本模板）
        self.template_samples: Dict[str, List[np.ndarray]] = {}
        # {tile_name: [sample_hash, ...]}
        self.template_sample_hashes: Dict[str, List[str]] = {}
        # {tile_name: prototype_gray_float32}（标准尺寸灰度原型）
        self.template_prototypes: Dict[str, np.ndarray] = {}
        # 出现在多个类别中的样本哈希（判定为“无区分度样本”）
        self.ambiguous_sample_hashes: set[str] = set()
        # 上一次 recognize_hand 的详细结果（用于调试日志）
        self.last_recognition_details: List[Dict[str, Any]] = []

        # NN 融合参数
        self.nn_enabled = bool(nn_enabled)
        self.nn_model_path = nn_model_path
        self.nn_labels_path = nn_labels_path
        self.nn_fusion_weight = float(nn_fusion_weight)
        self.nn_min_confidence = float(nn_min_confidence)
        self.nn_top_k = max(1, int(nn_top_k))
        self.nn_priority = bool(nn_priority)
        self.enforce_mpsz_order = bool(enforce_mpsz_order)
        self.nn_classifier: Optional[TileNNClassifier] = None

        # YOLO 检测器参数
        self.yolo_enabled = bool(yolo_enabled)
        self.yolo_model_path = yolo_model_path
        self.yolo_conf_threshold = float(yolo_conf_threshold)
        self.yolo_priority = bool(yolo_priority)
        self.yolo_detector: Optional[YOLOTileDetector] = None

        # recognize_tile_with_candidates() 的最近一次调试信息
        self._last_single_recognition_meta: Dict[str, Any] = {}

        # 扫描式识别的辅助状态（供 vision_main 读取）
        self.last_scan_boxes: List[Tuple[int, int, int, int]] = []   # (x1,y1,x2,y2) 已识别牌的裁剪框
        self.last_scan_hand_count: int = 0    # 本次扫描得到的手牌数（不含摸牌）
        self.last_scan_has_drawn: bool = False  # 本次扫描是否检测到摸牌

        self._load_templates()

        if self.nn_enabled:
            self.nn_classifier = TileNNClassifier(
                model_path=self.nn_model_path,
                labels_path=self.nn_labels_path,
            )

        if self.yolo_enabled:
            try:
                if self.yolo_model_path:
                    self.yolo_detector = YOLOTileDetector(
                        model_path=self.yolo_model_path,
                        conf_threshold=self.yolo_conf_threshold,
                    )
                else:
                    # 使用默认路径
                    project_root = Path(__file__).parent.parent.parent
                    default_model_path = project_root / "yolo_dataset" / "runs" / "tiles_yolov52" / "weights" / "best.pt"
                    if default_model_path.exists():
                        self.yolo_detector = YOLOTileDetector(
                            model_path=str(default_model_path),
                            conf_threshold=self.yolo_conf_threshold,
                        )
                        logger.info(f"✅ YOLO检测器已启用: {default_model_path}")
                    else:
                        logger.warning(f"YOLO模型未找到: {default_model_path}，YOLO检测已禁用")
                        self.yolo_enabled = False
            except Exception as e:
                logger.warning(f"YOLO检测器初始化失败: {e}，YOLO检测已禁用")
                self.yolo_enabled = False
                self.yolo_detector = None

    def has_nn_model(self) -> bool:
        """是否已加载可用神经网络模型"""
        return self.nn_classifier is not None and self.nn_classifier.available()

    def has_yolo_model(self) -> bool:
        """是否已加载可用YOLO模型"""
        return self.yolo_detector is not None

    def has_recognition_backend(self) -> bool:
        """是否具备任一识别能力（YOLO、模板或 NN）"""
        return self.has_yolo_model() or self.has_templates() or self.has_nn_model()

    # ------------------------------------------------------------------
    # 模板加载
    # ------------------------------------------------------------------

    def _load_templates(self):
        """从模板目录加载所有牌型图片"""
        if not self.templates_dir.exists():
            logger.warning(
                f"模板目录不存在: {self.templates_dir}\n"
                "  → 请运行 tools/capture_templates.py 来生成模板图片"
            )
            return

        loaded = 0
        multi_sample_classes = 0
        total_sample_files = 0

        supplemental_roots = [
            PROJECT_ROOT / "tools" / "templates" / "tiles",
            PROJECT_ROOT / "majsoul_bot" / "tools" / "templates" / "tiles",
        ]

        def _collect_candidate_paths(base_dir: Path, name: str) -> List[Path]:
            collected: List[Path] = []

            class_dir = base_dir / name
            if class_dir.is_dir():
                collected.extend(
                    sorted(
                        p for p in class_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                    )
                )

            if not collected:
                for ext in (".png", ".jpg", ".jpeg"):
                    flat_path = base_dir / f"{name}{ext}"
                    if flat_path.exists():
                        collected.append(flat_path)
                        break

            return collected

        hash_to_classes: Dict[str, set[str]] = {}

        for tile_name in ALL_TILE_NAMES:
            candidate_paths: List[Path] = []
            candidate_paths.extend(_collect_candidate_paths(self.templates_dir, tile_name))

            # 主模板样本不足时，自动补充 tools 下的样本库，提升区分能力
            if len(candidate_paths) <= 1:
                for extra_root in supplemental_roots:
                    if not extra_root.exists() or extra_root.resolve() == self.templates_dir.resolve():
                        continue
                    extra_candidates = _collect_candidate_paths(extra_root, tile_name)
                    for p in extra_candidates:
                        if p not in candidate_paths:
                            candidate_paths.append(p)

            total_sample_files += len(candidate_paths)
            if len(candidate_paths) > 1:
                multi_sample_classes += 1

            class_samples: List[np.ndarray] = []
            class_hashes: List[str] = []
            seen_signatures: set[str] = set()
            for path in candidate_paths:
                parent_label = _normalize_label(path.parent.name)
                stem_label = _infer_label_from_stem(path.stem)

                # 若目录标签与文件名标签冲突，视为脏数据，跳过
                if parent_label and stem_label and parent_label != stem_label:
                    continue

                inferred = _infer_label_from_path(path)
                if inferred is not None and inferred != tile_name:
                    continue

                img = cv2.imread(str(path))
                if img is not None:
                    h, w = img.shape[:2]
                    # 过滤异常大样本（通常是误裁剪的大图，会干扰匹配）
                    if h * w > 10_000:
                        continue

                    # 去重：避免同一像素模板重复参与评分
                    sig = str(hash(img.tobytes()))
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)
                    class_samples.append(img)
                    class_hashes.append(sig)
                    hash_to_classes.setdefault(sig, set()).add(tile_name)

            if class_samples:
                # 样本限流：避免样本过多导致匹配耗时过高
                if len(class_samples) > 6:
                    def _sample_priority(img: np.ndarray) -> float:
                        if img.ndim == 3:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = img
                        h, w = gray.shape[:2]
                        lap = cv2.Laplacian(gray, cv2.CV_32F)
                        detail = float(np.var(lap))
                        # 优先较高清晰样本，同时兼顾分辨率
                        return detail + 0.001 * float(h * w)

                    ranked_idx = sorted(
                        range(len(class_samples)),
                        key=lambda i: _sample_priority(class_samples[i]),
                        reverse=True,
                    )[:6]
                    class_samples = [class_samples[i] for i in ranked_idx]
                    class_hashes = [class_hashes[i] for i in ranked_idx]

                self.template_samples[tile_name] = class_samples
                self.template_sample_hashes[tile_name] = class_hashes

                proto_stack: List[np.ndarray] = []
                for sample in class_samples:
                    if sample.ndim == 3:
                        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = sample
                    proto_stack.append(cv2.resize(gray, (32, 48), interpolation=cv2.INTER_AREA).astype(np.float32))
                if proto_stack:
                    self.template_prototypes[tile_name] = np.mean(np.stack(proto_stack, axis=0), axis=0)

                # 兼容旧逻辑：选一张更稳定的代表模板（优先更高分辨率，次看纹理信息）
                def _representative_score(img: np.ndarray) -> float:
                    if img.ndim == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img
                    h, w = gray.shape[:2]
                    lap = cv2.Laplacian(gray, cv2.CV_32F)
                    detail = float(np.var(lap))
                    return float(h * w) + 0.01 * detail

                self.templates[tile_name] = max(class_samples, key=_representative_score)
                loaded += 1

        self.ambiguous_sample_hashes = {
            h for h, cls_set in hash_to_classes.items() if len(cls_set) > 1
        }

        # 在知道“跨类重复样本”之后，重建每类原型与代表模板，尽量使用非歧义样本
        for tile_name, class_samples in self.template_samples.items():
            hashes = self.template_sample_hashes.get(tile_name, [])

            if hashes and len(hashes) == len(class_samples):
                unique_samples = [
                    img for img, h in zip(class_samples, hashes) if h not in self.ambiguous_sample_hashes
                ]
                effective_samples = unique_samples if unique_samples else class_samples
            else:
                effective_samples = class_samples

            proto_stack: List[np.ndarray] = []
            for sample in effective_samples:
                if sample.ndim == 3:
                    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                else:
                    gray = sample
                proto_stack.append(cv2.resize(gray, (32, 48), interpolation=cv2.INTER_AREA).astype(np.float32))
            if proto_stack:
                self.template_prototypes[tile_name] = np.mean(np.stack(proto_stack, axis=0), axis=0)

            def _representative_score(img: np.ndarray) -> float:
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                h, w = gray.shape[:2]
                lap = cv2.Laplacian(gray, cv2.CV_32F)
                detail = float(np.var(lap))
                return float(h * w) + 0.01 * detail

            self.templates[tile_name] = max(effective_samples, key=_representative_score)

        if loaded == 0:
            logger.warning(f"模板目录 {self.templates_dir} 中未找到任何模板图片")
        else:
            logger.info(
                f"已加载 {loaded}/{len(ALL_TILE_NAMES)} 个牌型模板"
                f"（检测到样本文件 {total_sample_files} 张，"
                f"其中 {multi_sample_classes} 类含多样本）"
            )

    def has_templates(self) -> bool:
        """是否已加载模板"""
        return len(self.template_samples) > 0 or len(self.templates) > 0

    @staticmethod
    def _safe_match_score(target: np.ndarray, template: np.ndarray) -> float:
        """
        执行一次模板匹配并返回 [0,1] 分数。

        采用多种匹配准则融合，降低单一 CCOEFF 在相似牌面上的“并列高分”问题。
        """
        if target.size == 0 or template.size == 0:
            return 0.0
        if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
            return 0.0

        result_ccoeff = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        _, ccoeff_max, _, _ = cv2.minMaxLoc(result_ccoeff)

        result_ccorr = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED)
        _, ccorr_max, _, _ = cv2.minMaxLoc(result_ccorr)

        result_sqdiff = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
        sqdiff_min, _, _, _ = cv2.minMaxLoc(result_sqdiff)
        sqdiff_score = 1.0 - float(sqdiff_min)

        score = 0.55 * float(ccoeff_max) + 0.25 * float(ccorr_max) + 0.20 * sqdiff_score
        return max(0.0, min(1.0, score))

    def _score_template_samples(
        self,
        tile_gray: np.ndarray,
        tile_edge: np.ndarray,
        samples: Sequence[np.ndarray],
        scales: Sequence[float],
        sample_hashes: Optional[Sequence[str]] = None,
        ambiguous_hashes: Optional[set[str]] = None,
    ) -> float:
        """
        对单类多个模板样本评分，返回该类最终模板分。

        评分策略：
        - 每个样本在多尺度上匹配
        - 灰度分 + 边缘分融合（抗光照/色偏）
        - 同类样本取“最佳样本 + 次佳样本均值”增强稳定性
        """
        sample_scores: List[float] = []

        ambiguous_hashes = ambiguous_hashes or set()
        enabled_indices: List[int] = []
        if sample_hashes and len(sample_hashes) == len(samples):
            enabled_indices = [i for i, h in enumerate(sample_hashes) if h not in ambiguous_hashes]
            if not enabled_indices:
                # 若全部样本都冲突，退化为使用全量样本
                enabled_indices = list(range(len(samples)))
        else:
            enabled_indices = list(range(len(samples)))

        for i in enabled_indices:
            sample = samples[i]
            if sample.ndim == 3:
                sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            else:
                sample_gray = sample
            sample_edge = cv2.Canny(sample_gray, 60, 160)

            best_sample_score = 0.0
            for scale in scales:
                new_w = max(8, int(tile_gray.shape[1] * scale))
                new_h = max(8, int(tile_gray.shape[0] * scale))

                resized_gray = cv2.resize(sample_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                gray_score = self._safe_match_score(tile_gray, resized_gray)

                edge_score = 0.0
                if np.any(sample_edge) and np.any(tile_edge):
                    resized_edge = cv2.resize(sample_edge, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    edge_score = self._safe_match_score(tile_edge, resized_edge)

                # 灰度为主，边缘辅助
                combined = 0.75 * gray_score + 0.25 * edge_score
                if combined > best_sample_score:
                    best_sample_score = combined

            sample_scores.append(best_sample_score)

        if not sample_scores:
            return 0.0

        ranked = sorted(sample_scores, reverse=True)
        top1 = ranked[0]
        if len(ranked) == 1:
            return float(top1)

        topn = ranked[: min(3, len(ranked))]
        return float(0.8 * top1 + 0.2 * float(np.mean(topn)))

    @staticmethod
    def _prototype_similarity(tile_gray: np.ndarray, proto_gray: np.ndarray) -> float:
        """计算 tile 与类别原型的相似度（0~1，越高越相似）。"""
        if tile_gray.size == 0 or proto_gray.size == 0:
            return 0.0
        resized = cv2.resize(tile_gray, (proto_gray.shape[1], proto_gray.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        mae = float(np.mean(np.abs(resized - proto_gray)))
        sim = 1.0 - (mae / 255.0)
        return max(0.0, min(1.0, sim))

    # ------------------------------------------------------------------
    # 单张牌识别
    # ------------------------------------------------------------------

    def recognize_tile_with_candidates(
        self,
        tile_img: np.ndarray,
        top_k: int = 3,
    ) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        """
        识别单张牌并返回候选列表。

        Returns:
            (best_name, best_score, candidates)
            - best_name: 得分最高的牌名（低于阈值时仍返回最高候选）
            - best_score: 最高候选分数
            - candidates: Top-K 候选 [(tile_name, score), ...]
        """
        if tile_img.size == 0:
            return None, 0.0, []

        template_score_map: Dict[str, float] = {}
        nn_score_map: Dict[str, float] = {}
        template_best_name: Optional[str] = None
        template_best_score: float = 0.0
        nn_best_name: Optional[str] = None
        nn_best_prob: float = 0.0

        if tile_img.ndim == 3:
            tile_gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
        else:
            tile_gray = tile_img
        tile_edge = cv2.Canny(tile_gray, 60, 160)

        # 以当前牌尺寸为基准做多尺度匹配（兼顾校准误差与截图缩放）
        scales = (0.74, 0.86, 1.0)

        # 优先使用多样本模板；若旧代码直接写入 self.templates 也可兼容
        if self.template_samples:
            class_to_samples = self.template_samples
        else:
            class_to_samples = {k: [v] for k, v in self.templates.items()}

        # 先用原型分做粗筛，降低全量细匹配开销
        proto_score_map: Dict[str, float] = {}
        for tile_name in class_to_samples.keys():
            proto_score_map[tile_name] = self._prototype_similarity(
                tile_gray,
                self.template_prototypes.get(tile_name, np.array([])),
            )

        shortlist_names = {
            name for name, _ in sorted(
                proto_score_map.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:8]
        }
        if not shortlist_names:
            shortlist_names = set(class_to_samples.keys())

        for tile_name, samples in class_to_samples.items():
            sample_hashes = self.template_sample_hashes.get(tile_name)
            proto_score = float(proto_score_map.get(tile_name, 0.0))
            if tile_name in shortlist_names:
                sample_score = self._score_template_samples(
                    tile_gray=tile_gray,
                    tile_edge=tile_edge,
                    samples=samples,
                    scales=scales,
                    sample_hashes=sample_hashes,
                    ambiguous_hashes=self.ambiguous_sample_hashes,
                )
                score = 0.85 * sample_score + 0.15 * proto_score
            else:
                # 非候选类仅用原型分（加速）
                score = 0.90 * proto_score
            template_score_map[tile_name] = score

            if score > template_best_score:
                template_best_score = score
                template_best_name = tile_name

        if self.has_nn_model():
            nn_best_name, nn_best_prob, _nn_top_candidates, nn_score_map = self.nn_classifier.predict(
                tile_img,
                top_k=max(top_k, self.nn_top_k),
            )

        candidate_names = set(template_score_map.keys()) | set(nn_score_map.keys())
        if not candidate_names:
            self._last_single_recognition_meta = {
                "template_best_name": template_best_name,
                "template_best_score": float(template_best_score),
                "nn_best_name": nn_best_name,
                "nn_best_prob": float(nn_best_prob),
                "best_name": None,
                "best_score": 0.0,
            }
            return None, 0.0, []

        nn_weight = self.nn_fusion_weight if self.has_nn_model() else 0.0
        if self.nn_priority and self.has_nn_model():
            nn_weight = max(0.90, nn_weight)
        fused_score_map: Dict[str, float] = {}
        for name in candidate_names:
            tmpl_score = max(0.0, float(template_score_map.get(name, 0.0)))
            nn_score = max(0.0, float(nn_score_map.get(name, 0.0)))
            fused_score = (1.0 - nn_weight) * tmpl_score + nn_weight * nn_score
            fused_score_map[name] = float(fused_score)

        class_rank = {name: idx for idx, name in enumerate(ALL_TILE_NAMES)}
        if self.nn_priority and self.has_nn_model():
            scored_candidates = sorted(
                fused_score_map.items(),
                key=lambda item: (
                    float(nn_score_map.get(item[0], 0.0)),
                    item[1],
                    float(template_score_map.get(item[0], 0.0)),
                    -int(class_rank.get(item[0], 10_000)),
                ),
                reverse=True,
            )
        else:
            scored_candidates = sorted(
                fused_score_map.items(),
                key=lambda item: (
                    item[1],
                    float(template_score_map.get(item[0], 0.0)),
                    float(nn_score_map.get(item[0], 0.0)),
                    -int(class_rank.get(item[0], 10_000)),
                ),
                reverse=True,
            )
        if top_k > 0:
            scored_candidates = scored_candidates[:top_k]

        best_name, best_score = scored_candidates[0]
        if self.nn_priority and self.has_nn_model() and nn_best_name:
            best_name = nn_best_name
            best_score = max(
                float(nn_best_prob),
                float(fused_score_map.get(nn_best_name, 0.0)),
            )
        second_score = float(scored_candidates[1][1]) if len(scored_candidates) > 1 else 0.0
        margin = max(0.0, float(best_score) - second_score)
        agree_bonus = 0.0
        if best_name == template_best_name:
            agree_bonus += 0.02
        if best_name == nn_best_name and self.has_nn_model():
            agree_bonus += 0.02
        calibrated_best_score = min(1.0, float(best_score) + min(0.05, 0.5 * margin) + agree_bonus)

        self._last_single_recognition_meta = {
            "template_best_name": template_best_name,
            "template_best_score": float(template_best_score),
            "nn_best_name": nn_best_name,
            "nn_best_prob": float(nn_best_prob),
            "best_name": best_name,
            "best_score": float(calibrated_best_score),
            "raw_best_score": float(best_score),
            "second_score": float(second_score),
            "margin": float(margin),
            "nn_priority": bool(self.nn_priority),
        }

        return best_name, float(calibrated_best_score), scored_candidates

    def recognize_tile(self, tile_img: np.ndarray) -> Tuple[Optional[str], float]:
        """识别单张牌，返回阈值过滤后的结果（兼容旧接口）。"""
        best_name, best_score, _ = self.recognize_tile_with_candidates(tile_img, top_k=1)

        meta = self._last_single_recognition_meta or {}
        nn_best_name = meta.get("nn_best_name")
        nn_best_prob = float(meta.get("nn_best_prob", 0.0))

        if self.nn_priority and self.has_nn_model() and nn_best_name:
            return nn_best_name, nn_best_prob

        if best_score >= self.threshold:
            return best_name, best_score

        if nn_best_name and nn_best_prob >= self.nn_min_confidence:
            return nn_best_name, nn_best_prob

        return None, best_score

    # ------------------------------------------------------------------
    # 手牌整体识别
    # ------------------------------------------------------------------

    def recognize_hand(
        self,
        screenshot: np.ndarray,
        hand_count: int = 13,
        has_drawn_tile: bool = True,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        识别手牌区域内的所有牌

        Args:
            screenshot: 完整游戏截图（BGR）
            hand_count: 手牌数量（不含摸牌），通常为 13
            has_drawn_tile: 是否存在摸牌（第 14 张）

        Returns:
            List[(tile_name, (center_x, center_y))]:
                - tile_name: 牌名，如 "3m"、"7z"；无法识别时为 "unknown_N"
                - (center_x, center_y): 该牌中心在 **截图** 中的像素坐标
        """
        img_h, img_w = screenshot.shape[:2]
        reg = self.regions.hand
        results: List[Tuple[str, Tuple[int, int]]] = []
        details: List[Dict[str, Any]] = []

        total = hand_count + (1 if has_drawn_tile else 0)

        for i in range(total):
            is_drawn = i >= hand_count
            x_start_rel, y_start_rel, tw_rel, th_rel = self.regions.get_tile_rect(
                i, hand_count, is_drawn
            )

            # 转换为像素坐标
            x_start = int(x_start_rel * img_w)
            y_start = int(y_start_rel * img_h)
            tile_w = max(1, int(tw_rel * img_w))
            tile_h = max(1, int(th_rel * img_h))

            # 边界保护
            x_end = min(x_start + tile_w, img_w)
            y_end = min(y_start + tile_h, img_h)

            if x_start >= img_w or y_start >= img_h:
                continue

            def _clip_rect(xs: int, ys: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
                xe = min(xs + w, img_w)
                ye = min(ys + h, img_h)
                xs = max(0, xs)
                ys = max(0, ys)
                if xe <= xs or ye <= ys:
                    return None
                return xs, ys, xe, ye

            base_rect = _clip_rect(x_start, y_start, tile_w, tile_h)
            if base_rect is None:
                continue

            # 允许在小范围内微调截取框，缓解“摸牌位偏移/牌被抬起”导致的低分
            y_step = max(1, int(tile_h * 0.08))
            x_step = max(1, int(tile_w * 0.10))

            offset_candidates: List[Tuple[int, int]] = [(0, 0), (0, -y_step), (0, y_step)]
            if is_drawn:
                offset_candidates.extend([
                    (-x_step, 0),
                    (x_step, 0),
                    (-x_step, -y_step),
                    (x_step, -y_step),
                ])

            checked_offsets: set[Tuple[int, int]] = set()
            crop_variants: List[Tuple[int, int, int, int, Tuple[int, int]]] = []
            for dx, dy in offset_candidates:
                if (dx, dy) in checked_offsets:
                    continue
                checked_offsets.add((dx, dy))
                rect = _clip_rect(x_start + dx, y_start + dy, tile_w, tile_h)
                if rect is None:
                    continue
                xs, ys, xe, ye = rect
                crop_variants.append((xs, ys, xe, ye, (dx, dy)))

            if not crop_variants:
                continue

            meta: Dict[str, Any] = {}

            # 识别（模板 / NN / 融合）
            if self.has_recognition_backend():
                best_name: Optional[str] = None
                best_score = 0.0
                candidates: List[Tuple[str, float]] = []
                chosen_rect = crop_variants[0]
                chosen_meta: Dict[str, Any] = {}

                for xs, ys, xe, ye, offset in crop_variants:
                    tile_img = screenshot[ys:ye, xs:xe]
                    if tile_img.size == 0:
                        continue

                    cur_best_name, cur_best_score, cur_candidates = self.recognize_tile_with_candidates(
                        tile_img,
                        top_k=3,
                    )
                    cur_meta = dict(self._last_single_recognition_meta or {})

                    if cur_best_score > best_score:
                        best_name = cur_best_name
                        best_score = float(cur_best_score)
                        candidates = cur_candidates
                        chosen_rect = (xs, ys, xe, ye, offset)
                        chosen_meta = cur_meta

                meta = chosen_meta
                nn_best_name = meta.get("nn_best_name")
                nn_best_prob = float(meta.get("nn_best_prob", 0.0))

                accept_reason = "below_threshold"
                accepted = False

                if self.nn_priority and nn_best_name:
                    best_name = nn_best_name
                    best_score = nn_best_prob
                    accepted = True
                    accept_reason = "nn_priority"
                elif best_name is not None and best_score >= self.threshold:
                    accepted = True
                    accept_reason = "fusion"
                elif nn_best_name and nn_best_prob >= self.nn_min_confidence:
                    # NN 兜底：融合分未过阈值时，允许高置信 NN 结果通过
                    best_name = nn_best_name
                    best_score = nn_best_prob
                    accepted = True
                    accept_reason = "nn_confidence"

                tile_name = best_name if accepted else f"unknown_{i}"
            else:
                # 无识别后端：仅记录位置
                tile_name = f"pos_{i}"
                best_name = None
                best_score = 0.0
                candidates = []
                accepted = False
                accept_reason = "no_backend"
                chosen_rect = crop_variants[0]

            chosen_xs, chosen_ys, chosen_xe, chosen_ye, chosen_offset = chosen_rect
            center_x = chosen_xs + (chosen_xe - chosen_xs) // 2
            center_y = chosen_ys + (chosen_ye - chosen_ys) // 2
            results.append((tile_name, (center_x, center_y)))

            details.append(
                {
                    "index": i,
                    "is_drawn": is_drawn,
                    "recognized_name": tile_name,
                    "best_name": best_name,
                    "best_score": float(best_score),
                    "accepted": accepted,
                    "accept_reason": accept_reason,
                    "template_best_name": meta.get("template_best_name"),
                    "template_best_score": float(meta.get("template_best_score", 0.0)),
                    "nn_best_name": meta.get("nn_best_name"),
                    "nn_best_score": float(meta.get("nn_best_prob", 0.0)),
                    "candidates": candidates,
                    "center": (center_x, center_y),
                    "crop_offset": chosen_offset,
                }
            )

        if self.enforce_mpsz_order and details:
            self._apply_mpsz_order_constraint(details, hand_count=hand_count)
            results = [
                (str(d.get("recognized_name", "")), tuple(d.get("center", (0, 0))))
                for d in details
            ]

        self.last_recognition_details = details

        return results

    @staticmethod
    def _tile_suit_rank(tile_name: Optional[str]) -> Optional[int]:
        if not tile_name or len(tile_name) < 2:
            return None
        suit = str(tile_name)[-1].lower()
        return {"m": 0, "p": 1, "s": 2, "z": 3}.get(suit)

    def _apply_mpsz_order_constraint(self, details: List[Dict[str, Any]], hand_count: int = 13) -> None:
        """
        利用雀魂手牌自动排序特性，对手牌（不含摸牌）应用 m→p→s→z 顺序约束。

        策略：
        - 作用于所有可识别为牌名的手牌位（0~hand_count-1）
        - 使用每个牌位的 candidates 做动态规划，求 suit 非递减路径
        - 在满足非递减前提下，增加“换花色代价”，抑制同花色区块中的跨花色错插
        - 仅在当前存在逆序且新路径可行时应用
        """
        suit_positions: List[int] = []
        options_per_pos: List[List[Tuple[str, float, int]]] = []

        for i, d in enumerate(details):
            if i >= hand_count or bool(d.get("is_drawn", False)):
                continue

            options_raw = d.get("candidates", []) or []
            merged: Dict[str, float] = {}
            for name, score in options_raw:
                rank = self._tile_suit_rank(name)
                if rank is None:
                    continue
                merged[str(name)] = max(float(score), float(merged.get(str(name), 0.0)))

            best_name = str(d.get("best_name") or "")
            best_score = float(d.get("best_score", 0.0))
            if best_name:
                rank = self._tile_suit_rank(best_name)
                if rank is not None:
                    merged[best_name] = max(best_score, float(merged.get(best_name, 0.0)))

            opts: List[Tuple[str, float, int]] = []
            for name, score in merged.items():
                rank = self._tile_suit_rank(name)
                if rank is None:
                    continue
                opts.append((name, float(score), int(rank)))

            if not opts:
                continue

            opts.sort(key=lambda x: x[1], reverse=True)
            options_per_pos.append(opts[:5])
            suit_positions.append(i)

        if len(options_per_pos) < 2:
            return

        current_ranks = []
        for pos in suit_positions:
            rk = self._tile_suit_rank(str(details[pos].get("best_name") or ""))
            if rk is None:
                return
            current_ranks.append(rk)
        current_violations = sum(
            1 for i in range(1, len(current_ranks)) if current_ranks[i] < current_ranks[i - 1]
        )
        if current_violations == 0:
            return

        switch_penalty = 0.03

        # dp[i][rank] = (objective, raw_score_sum)
        # objective = raw_score_sum - switch_penalty * switches
        dp: List[Dict[int, Tuple[float, float]]] = [{} for _ in range(len(options_per_pos))]
        back: List[Dict[int, Tuple[int, str, float]]] = [{} for _ in range(len(options_per_pos))]

        for name, score, rank in options_per_pos[0]:
            dp[0][rank] = (float(score), float(score))
            back[0][rank] = (-1, name, float(score))

        for i in range(1, len(options_per_pos)):
            for name, score, rank in options_per_pos[i]:
                best_prev_rank = None
                best_prev_obj = -1e9
                best_prev_raw = -1e9
                for prev_rank, (prev_obj, prev_raw) in dp[i - 1].items():
                    if prev_rank > rank:
                        continue
                    switch_cost = switch_penalty if prev_rank != rank else 0.0
                    cand_obj = prev_obj + float(score) - switch_cost
                    cand_raw = prev_raw + float(score)
                    if (
                        cand_obj > best_prev_obj
                        or (abs(cand_obj - best_prev_obj) <= 1e-9 and cand_raw > best_prev_raw)
                    ):
                        best_prev_obj = cand_obj
                        best_prev_raw = cand_raw
                        best_prev_rank = prev_rank
                if best_prev_rank is None:
                    continue
                cur_obj = best_prev_obj
                cur_raw = best_prev_raw
                old = dp[i].get(rank)
                if old is None or cur_obj > old[0] or (abs(cur_obj - old[0]) <= 1e-9 and cur_raw > old[1]):
                    dp[i][rank] = (cur_obj, cur_raw)
                    back[i][rank] = (best_prev_rank, name, float(score))

        if not dp[-1]:
            return

        end_rank = max(dp[-1].keys(), key=lambda r: (dp[-1][r][0], dp[-1][r][1]))
        selected: List[Tuple[str, float]] = [("", 0.0)] * len(options_per_pos)
        cur_rank = end_rank
        for i in range(len(options_per_pos) - 1, -1, -1):
            prev_rank, name, score = back[i][cur_rank]
            selected[i] = (name, score)
            cur_rank = prev_rank
            if cur_rank < 0:
                break

        for local_i, pos in enumerate(suit_positions):
            new_name, new_score = selected[local_i]
            if not new_name:
                continue
            old_name = str(details[pos].get("best_name") or "")
            if new_name == old_name:
                continue
            details[pos]["best_name"] = new_name
            details[pos]["recognized_name"] = new_name
            details[pos]["best_score"] = float(new_score)
            details[pos]["accepted"] = True
            details[pos]["accept_reason"] = "mpsz_order_constraint"

    # ------------------------------------------------------------------
    # 扫描式手牌识别（不依赖预设张数）
    # ------------------------------------------------------------------

    def scan_hand_row(
        self,
        screenshot: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """
        在手牌行扫描并返回所有检测到的牌的边界框。

        算法：
        1. 裁剪手牌行区域（以 regions.hand 参数为基准，稍加上下边距）
        2. 对每列计算亮度均值，用 Otsu 自适应阈值分离牌列 vs 背景列
        3. 形态学闭运算填充牌内部的细小暗纹
        4. 识别连续亮列段（= 一张或多张相邻牌）
        5. 按参考牌宽将过宽的段均匀分割为单牌

        Returns:
            [(x1, y1, x2, y2), ...] 截图绝对像素坐标，按 x 排序
        """
        img_h, img_w = screenshot.shape[:2]
        h_reg = self.regions.hand

        ref_tw = max(8, int(h_reg.tile_width * img_w))
        ref_th = max(8, int(h_reg.tile_height * img_h))

        # 手牌行裁剪范围（y方向留 1% 边距，x 从手牌起始到屏幕右边）
        scan_y1 = max(0, int((h_reg.y_start - 0.01) * img_h))
        scan_y2 = min(img_h, int((h_reg.y_start + h_reg.tile_height + 0.01) * img_h))
        scan_x1 = max(0, int((h_reg.x_start - 0.05) * img_w))
        scan_x2 = img_w

        strip = screenshot[scan_y1:scan_y2, scan_x1:scan_x2]
        if strip.size == 0:
            return []

        sw = strip.shape[1]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

        # 列亮度均值
        col_mean = np.mean(gray.astype(np.float32), axis=0)  # shape (sw,)

        # Otsu 阈值：分离"牌列"(亮) vs "背景列"(暗/间隙)
        col_uint8 = col_mean.clip(0, 255).astype(np.uint8).reshape(1, -1)
        otsu_thresh, col_binary = cv2.threshold(
            col_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        is_tile_col = col_binary.flatten().astype(np.uint8).reshape(1, -1)

        # 闭运算：填充牌内部细小暗纹间隙（最大 ref_tw//5 像素）
        gap_fill = max(1, ref_tw // 5)
        kernel_h = np.ones((1, gap_fill * 2 + 1), dtype=np.uint8)
        closed = cv2.morphologyEx(is_tile_col, cv2.MORPH_CLOSE, kernel_h).flatten().astype(bool)

        # 找连续亮段
        boxes: List[Tuple[int, int, int, int]] = []
        in_seg = False
        seg_s = 0
        min_seg_w = ref_tw // 2

        for cx in range(sw):
            if closed[cx] and not in_seg:
                in_seg = True
                seg_s = cx
            elif (not closed[cx] or cx == sw - 1) and in_seg:
                seg_e = cx if not closed[cx] else cx + 1
                seg_w = seg_e - seg_s
                if seg_w >= min_seg_w:
                    # 按参考牌宽均匀分割段
                    n = max(1, round(seg_w / ref_tw))
                    sub_w = seg_w / n
                    for t in range(n):
                        tx1 = scan_x1 + int(seg_s + t * sub_w)
                        tx2 = scan_x1 + int(seg_s + (t + 1) * sub_w)
                        boxes.append((tx1, scan_y1, tx2, scan_y2))
                in_seg = False

        return sorted(boxes, key=lambda b: b[0])

    def _classify_scan_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        img_w: int,
        has_drawn_tile: bool,
    ) -> Tuple[
        List[Tuple[int, int, int, int]],   # hand_boxes
        List[Tuple[int, int, int, int]],   # meld_boxes
        Optional[Tuple[int, int, int, int]],  # drawn_box
    ]:
        """
        按坐标将扫描到的牌框分类为手牌、摸牌、副露牌。

        分类逻辑：
        - x_rel > hand_x_max → 属于副露区（吃/碰/杠），不计入手牌
        - 剩余牌按 x 从左到右排列，找最大间隙
          - 若间隙 > drawn_x_gap_ratio × ref_tile_w：
              间隙前 = 手牌，间隙后第一张 = 摸牌（若 has_drawn_tile）
          - 若无大间隙：全部视为手牌（尚未摸牌或摸牌已合并）
        """
        if not boxes:
            return [], [], None

        ref_tw = max(8, int(self.regions.hand.tile_width * img_w))
        gap_thresh = ref_tw * self.regions.drawn_x_gap_ratio
        hand_x_max_px = int(self.regions.hand_x_max * img_w)

        # 按 x 边界初步分流
        in_hand_zone = sorted(
            [b for b in boxes if b[0] < hand_x_max_px],
            key=lambda b: b[0],
        )
        meld_boxes = [b for b in boxes if b[0] >= hand_x_max_px]

        if not in_hand_zone:
            return [], meld_boxes, None

        # 找手牌区内的最大相邻间隙
        gaps = []
        for i in range(1, len(in_hand_zone)):
            gap = in_hand_zone[i][0] - in_hand_zone[i - 1][2]  # x1_next - x2_prev
            gaps.append(gap)

        drawn_box: Optional[Tuple[int, int, int, int]] = None
        hand_boxes: List[Tuple[int, int, int, int]] = []

        if gaps and has_drawn_tile:
            max_gap = max(gaps)
            if max_gap > gap_thresh:
                split_idx = gaps.index(max_gap) + 1  # 第一个大间隙后的索引
                hand_boxes = in_hand_zone[:split_idx]
                after_gap = in_hand_zone[split_idx:]
                if after_gap:
                    drawn_box = after_gap[0]
                    # 若间隙后有多张（罕见情况），其余加入手牌
                    hand_boxes.extend(after_gap[1:])
            else:
                hand_boxes = in_hand_zone
        else:
            hand_boxes = in_hand_zone

        return hand_boxes, meld_boxes, drawn_box

    def recognize_hand_by_yolo(
        self,
        screenshot: np.ndarray,
        has_drawn_tile: bool = True,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        使用YOLO直接检测手牌区域的所有牌。

        相比扫描式识别,YOLO可以直接返回牌名和位置,无需预设张数。

        Args:
            screenshot:     完整游戏截图(BGR)
            has_drawn_tile: 是否已摸牌标志(用于结果分类)

        Returns:
            [(tile_name, (center_x, center_y)), ...],按 x 从左到右排序
        """
        if not self.has_yolo_model():
            logger.warning("YOLO模型未加载,fallback到扫描式识别")
            return self.recognize_hand_by_scan(screenshot, has_drawn_tile)

        img_h, img_w = screenshot.shape[:2]
        h_reg = self.regions.hand

        # 裁剪手牌区域(包含摸牌位置)
        crop_y1 = max(0, int((h_reg.y_start - 0.02) * img_h))
        crop_y2 = min(img_h, int((h_reg.y_start + h_reg.tile_height + 0.02) * img_h))
        crop_x1 = max(0, int((h_reg.x_start - 0.05) * img_w))
        # 包含摸牌区域
        crop_x2 = min(img_w, int(self.regions.hand_x_max * img_w))

        hand_region = screenshot[crop_y1:crop_y2, crop_x1:crop_x2]

        if hand_region.size == 0:
            logger.warning("手牌区域为空")
            return []

        # YOLO检测
        detections = self.yolo_detector.detect_tiles(hand_region)

        if not detections:
            logger.debug("YOLO未检测到任何牌")
            return []

        # 转换为绝对坐标并计算中心点
        results: List[Tuple[str, Tuple[int, int], Tuple[int, int, int, int]]] = []
        for tile_name, confidence, (x1, y1, x2, y2) in detections:
            abs_x1 = crop_x1 + x1
            abs_y1 = crop_y1 + y1
            abs_x2 = crop_x1 + x2
            abs_y2 = crop_y1 + y2
            center_x = (abs_x1 + abs_x2) // 2
            center_y = (abs_y1 + abs_y2) // 2
            results.append((tile_name, (center_x, center_y), (abs_x1, abs_y1, abs_x2, abs_y2)))

        # 按x坐标排序
        results.sort(key=lambda r: r[2][0])

        # 分类手牌和摸牌
        ref_tw = max(8, int(h_reg.tile_width * img_w))
        gap_thresh = ref_tw * self.regions.drawn_x_gap_ratio

        hand_count = 0
        drawn_idx = -1

        # 查找最大间隙
        if len(results) > 1 and has_drawn_tile:
            max_gap = 0
            max_gap_idx = -1
            for i in range(1, len(results)):
                gap = results[i][2][0] - results[i-1][2][2]  # x1_next - x2_prev
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i

            if max_gap > gap_thresh and max_gap_idx > 0:
                hand_count = max_gap_idx
                drawn_idx = max_gap_idx
            else:
                hand_count = len(results)
        else:
            hand_count = len(results)

        # 记录状态
        self.last_scan_hand_count = hand_count
        self.last_scan_has_drawn = drawn_idx >= 0
        self.last_scan_boxes = [r[2] for r in results]

        # 构建识别详情
        details: List[Dict[str, Any]] = []
        for i, (tile_name, center, bbox) in enumerate(results):
            is_drawn = i == drawn_idx
            details.append({
                "index": i,
                "is_drawn": is_drawn,
                "recognized_name": tile_name,
                "best_name": tile_name,
                "best_score": 0.99,  # YOLO置信度已在检测时过滤
                "accepted": True,
                "accept_reason": "yolo",
                "template_best_name": None,
                "template_best_score": 0.0,
                "nn_best_name": None,
                "nn_best_score": 0.0,
                "candidates": [(tile_name, 0.99)],
                "center": center,
            })

        # 应用花色顺序约束
        if self.enforce_mpsz_order and details:
            self._apply_mpsz_order_constraint(details, hand_count=hand_count)

        self.last_recognition_details = details

        # 返回最终结果
        return [(d["recognized_name"], d["center"]) for d in details]

    def recognize_hand_by_scan(
        self,
        screenshot: np.ndarray,
        has_drawn_tile: bool = True,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        扫描式手牌识别:不需要预设手牌张数。

        通过扫描手牌行找到所有牌的位置,按坐标分类后对手牌+摸牌进行
        模板/NN 识别,返回格式与 recognize_hand() 完全兼容。

        Args:
            screenshot:     完整游戏截图(BGR)
            has_drawn_tile: 来自 game_state_detector 的"是否已摸牌"标志

        Returns:
            [(tile_name, (center_x, center_y)), ...],按 x 从左到右排序,
            摸牌排在末尾(若存在)
        """
        # 如果启用YOLO且优先使用,则使用YOLO识别
        if self.yolo_priority and self.has_yolo_model():
            return self.recognize_hand_by_yolo(screenshot, has_drawn_tile)

        img_h, img_w = screenshot.shape[:2]

        # ── 1. 检测牌框 ──
        all_boxes = self.scan_hand_row(screenshot)

        if not all_boxes:
            logger.debug("recognize_hand_by_scan: 手牌行未检测到任何牌")
            self.last_scan_boxes = []
            self.last_scan_hand_count = 0
            self.last_scan_has_drawn = False
            self.last_recognition_details = []
            return []

        # ── 2. 分类 ──
        hand_boxes, meld_boxes, drawn_box = self._classify_scan_boxes(
            all_boxes, img_w, has_drawn_tile
        )
        hand_count = len(hand_boxes)
        self.last_scan_hand_count = hand_count
        self.last_scan_has_drawn = drawn_box is not None

        logger.debug(
            f"recognize_hand_by_scan: 检测 {len(all_boxes)} 牌 "
            f"→ 手牌={hand_count}, 副露={len(meld_boxes)}, "
            f"摸牌={'是' if drawn_box else '否'}"
        )

        # ── 3. 组装待识别列表（手牌在前，摸牌在末） ──
        to_recognize: List[Tuple[int, int, int, int]] = list(hand_boxes)
        if drawn_box:
            to_recognize.append(drawn_box)
        self.last_scan_boxes = list(to_recognize)

        # ── 4. 逐牌识别 ──
        results: List[Tuple[str, Tuple[int, int]]] = []
        details: List[Dict[str, Any]] = []

        for i, (bx1, by1, bx2, by2) in enumerate(to_recognize):
            is_drawn_slot = i >= hand_count

            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(img_w, bx2); by2 = min(img_h, by2)
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2

            if bx2 <= bx1 or by2 <= by1:
                name = f"pos_{i}"
                results.append((name, (center_x, center_y)))
                details.append({
                    "index": i, "is_drawn": is_drawn_slot,
                    "recognized_name": name, "best_name": None,
                    "best_score": 0.0, "accepted": False, "accept_reason": "empty_crop",
                    "candidates": [], "center": (center_x, center_y),
                    "template_best_name": None, "template_best_score": 0.0,
                    "nn_best_name": None, "nn_best_score": 0.0,
                })
                continue

            tile_img = screenshot[by1:by2, bx1:bx2]

            if tile_img.size == 0 or not self.has_recognition_backend():
                name = f"pos_{i}"
                results.append((name, (center_x, center_y)))
                details.append({
                    "index": i, "is_drawn": is_drawn_slot,
                    "recognized_name": name, "best_name": None,
                    "best_score": 0.0, "accepted": False, "accept_reason": "no_backend",
                    "candidates": [], "center": (center_x, center_y),
                    "template_best_name": None, "template_best_score": 0.0,
                    "nn_best_name": None, "nn_best_score": 0.0,
                })
                continue

            best_name, best_score, candidates = self.recognize_tile_with_candidates(
                tile_img, top_k=self.nn_top_k
            )
            meta = dict(self._last_single_recognition_meta or {})
            nn_best_name = meta.get("nn_best_name")
            nn_best_prob = float(meta.get("nn_best_prob", 0.0))

            accepted = False
            accept_reason = "below_threshold"

            if self.nn_priority and nn_best_name:
                best_name = nn_best_name
                best_score = nn_best_prob
                accepted = True
                accept_reason = "nn_priority"
            elif best_name is not None and best_score >= self.threshold:
                accepted = True
                accept_reason = "fusion"
            elif nn_best_name and nn_best_prob >= self.nn_min_confidence:
                best_name = nn_best_name
                best_score = nn_best_prob
                accepted = True
                accept_reason = "nn_confidence"

            tile_name = best_name if accepted else f"unknown_{i}"
            results.append((tile_name, (center_x, center_y)))
            details.append({
                "index": i,
                "is_drawn": is_drawn_slot,
                "recognized_name": tile_name,
                "best_name": best_name,
                "best_score": float(best_score),
                "accepted": accepted,
                "accept_reason": accept_reason,
                "template_best_name": meta.get("template_best_name"),
                "template_best_score": float(meta.get("template_best_score", 0.0)),
                "nn_best_name": nn_best_name,
                "nn_best_score": float(nn_best_prob),
                "candidates": candidates,
                "center": (center_x, center_y),
            })

        # ── 5. 花色顺序约束（仅作用于手牌部分）──
        if self.enforce_mpsz_order and details:
            self._apply_mpsz_order_constraint(details, hand_count=hand_count)
            results = [
                (str(d.get("recognized_name", "")), tuple(d.get("center", (0, 0))))
                for d in details
            ]

        self.last_recognition_details = details
        return results

    def draw_scan_results(
        self,
        screenshot: np.ndarray,
        results: Optional[List[Tuple[str, Tuple[int, int]]]] = None,
    ) -> np.ndarray:
        """
        在截图上绘制扫描识别结果的边界框（调试用）。

        使用 last_scan_boxes / last_scan_hand_count 中记录的最近一次扫描状态。
        """
        debug = screenshot.copy()
        boxes = self.last_scan_boxes
        hand_count = self.last_scan_hand_count

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            is_drawn = i >= hand_count
            color = (0, 200, 255) if is_drawn else (0, 255, 100)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

            label = ""
            if results and i < len(results):
                label = results[i][0]
            cv2.putText(debug, label, (x1 + 2, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        return debug

    # ------------------------------------------------------------------
    # 视觉聚类 / 智能选牌
    # ------------------------------------------------------------------

    def compute_isolation_scores(
        self,
        tile_imgs: List[np.ndarray],
        std_w: int = 32,
        std_h: int = 32,
    ) -> List[float]:
        """
        计算每张牌与其他牌的"孤立度"得分（0~1，越高越孤立）

        原理：
          对每张牌，找出它与手牌中其他牌的最大视觉相似度，
          孤立度 = 1 - max_similarity（与最相似的牌越不像，孤立度越高）

        适用场景：
          在没有命名模板、无法识别具体牌型时，优先出打最孤立（
          最难凑成搭子）的牌，作为简单位置模式下的 discard 策略。

        Returns:
            List[float]: 每张牌的孤立度，与 tile_imgs 等长
        """
        n = len(tile_imgs)
        if n == 0:
            return []

        # 统一缩放到 std_w × std_h，便于快速比较
        normed: List[np.ndarray] = []
        for img in tile_imgs:
            if img.size == 0:
                normed.append(np.zeros((std_h, std_w, 3), dtype=np.uint8))
            else:
                normed.append(cv2.resize(img, (std_w, std_h)))

        isolation: List[float] = []
        for i in range(n):
            max_sim = 0.0
            for j in range(n):
                if i == j:
                    continue
                res = cv2.matchTemplate(normed[i], normed[j], cv2.TM_CCOEFF_NORMED)
                sim = max(0.0, float(res[0, 0]))
                if sim > max_sim:
                    max_sim = sim
            isolation.append(1.0 - max_sim)

        return isolation

    def find_best_discard_index(
        self,
        tile_imgs: List[np.ndarray],
        has_drawn_tile: bool,
    ) -> int:
        """
        基于视觉孤立度选出最佳出牌位置

        策略：
          1. 计算所有牌的孤立度
          2. 孤立度最高（与其他牌最不像）的牌优先出
          3. 如有摸牌（最后一张），且孤立度最高的牌是摸牌→摸切；
             否则出手牌中孤立度最高的一张

        Returns:
            int: 建议出打的牌在 tile_imgs 中的索引
        """
        if not tile_imgs:
            return 0

        scores = self.compute_isolation_scores(tile_imgs)
        if not scores:
            return len(tile_imgs) - 1 if has_drawn_tile else len(tile_imgs) // 2

        n = len(tile_imgs)
        hand_count = n - 1 if has_drawn_tile else n

        # 找全局孤立度最高的牌
        best_idx = int(max(range(n), key=lambda i: scores[i]))

        logger.debug(
            f"视觉孤立度: {[f'{s:.2f}' for s in scores]}  "
            f"→ 选择位置 {best_idx}"
        )
        return best_idx

    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------

    def extract_tile_images(
        self,
        screenshot: np.ndarray,
        hand_count: int = 13,
        has_drawn_tile: bool = True,
    ) -> List[np.ndarray]:
        """
        提取手牌区域内所有牌的图像（用于模板捕获工具）

        Returns:
            List[np.ndarray]: 每张牌的 BGR 图像
        """
        img_h, img_w = screenshot.shape[:2]
        images: List[np.ndarray] = []
        total = hand_count + (1 if has_drawn_tile else 0)

        for i in range(total):
            is_drawn = i >= hand_count
            x_rel, y_rel, tw_rel, th_rel = self.regions.get_tile_rect(
                i, hand_count, is_drawn
            )

            x_start = int(x_rel * img_w)
            y_start = int(y_rel * img_h)
            tile_w = max(1, int(tw_rel * img_w))
            tile_h = max(1, int(th_rel * img_h))

            tile_img = screenshot[
                y_start : y_start + tile_h, x_start : x_start + tile_w
            ]
            if tile_img.size > 0:
                images.append(tile_img.copy())
            else:
                images.append(np.zeros((90, 64, 3), dtype=np.uint8))

        return images

    def draw_hand_regions(
        self, screenshot: np.ndarray, hand_count: int = 13, has_drawn: bool = True
    ) -> np.ndarray:
        """
        在截图上绘制手牌区域边框（用于校准调试）

        Returns:
            np.ndarray: 带标注的截图副本
        """
        debug = screenshot.copy()
        img_h, img_w = debug.shape[:2]
        total = hand_count + (1 if has_drawn else 0)

        for i in range(total):
            is_drawn = i >= hand_count
            x_rel, y_rel, tw_rel, th_rel = self.regions.get_tile_rect(
                i, hand_count, is_drawn
            )
            x1 = int(x_rel * img_w)
            y1 = int(y_rel * img_h)
            x2 = x1 + max(1, int(tw_rel * img_w))
            y2 = y1 + max(1, int(th_rel * img_h))

            color = (0, 200, 255) if is_drawn else (0, 255, 100)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                debug, str(i), (x1 + 2, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )

        return debug
