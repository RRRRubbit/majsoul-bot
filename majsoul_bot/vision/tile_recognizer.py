"""
麻将牌识别模块
使用 OpenCV 模板匹配识别游戏画面中的麻将牌
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..game_logic.tile import Tile, TileType
from .regions import ScreenRegions, DEFAULT_REGIONS


# 所有 34 种牌的标准命名（与模板文件名对应）
ALL_TILE_NAMES: List[str] = (
    [f"{i}m" for i in range(1, 10)]   # 万子 1m~9m
    + [f"{i}p" for i in range(1, 10)] # 筒子 1p~9p
    + [f"{i}s" for i in range(1, 10)] # 索子 1s~9s
    + [f"{i}z" for i in range(1, 8)]  # 字牌 1z~7z（东南西北白发中）
)


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
    ):
        """
        Args:
            templates_dir: 模板图片目录路径
            threshold: 模板匹配最低得分（0~1，越高越严格）
            regions: 屏幕区域配置，默认使用全局 DEFAULT_REGIONS
        """
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.regions = regions or DEFAULT_REGIONS
        # {tile_name: template_bgr}
        self.templates: Dict[str, np.ndarray] = {}
        self._load_templates()

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
        for tile_name in ALL_TILE_NAMES:
            for ext in (".png", ".jpg", ".jpeg"):
                path = self.templates_dir / f"{tile_name}{ext}"
                if path.exists():
                    img = cv2.imread(str(path))
                    if img is not None:
                        self.templates[tile_name] = img
                        loaded += 1
                        break

        if loaded == 0:
            logger.warning(f"模板目录 {self.templates_dir} 中未找到任何模板图片")
        else:
            logger.info(f"已加载 {loaded}/{len(ALL_TILE_NAMES)} 个牌型模板")

    def has_templates(self) -> bool:
        """是否已加载模板"""
        return len(self.templates) > 0

    # ------------------------------------------------------------------
    # 单张牌识别
    # ------------------------------------------------------------------

    def recognize_tile(self, tile_img: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.templates or tile_img.size == 0:
            return None, 0.0

        best_name: Optional[str] = None
        best_score: float = -1.0

        target_h, target_w = tile_img.shape[:2]

        for tile_name, template in self.templates.items():
            t_h, t_w = template.shape[:2]

            # ===== 修复：让模板略小于目标，保留滑动空间 =====
            # 模板缩放到目标的 ~80% 大小，留出滑动匹配余量
            scale = 0.8
            new_w = max(8, int(target_w * scale))
            new_h = max(8, int(target_h * scale))

            if (t_h, t_w) != (new_h, new_w):
                resized = cv2.resize(template, (new_w, new_h))
            else:
                resized = template

            # 确保模板不大于目标
            if resized.shape[0] > target_h or resized.shape[1] > target_w:
                continue

            result = cv2.matchTemplate(tile_img, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_name = tile_name

        if best_score >= self.threshold:
            return best_name, best_score

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

            tile_img = screenshot[y_start:y_end, x_start:x_end]
            if tile_img.size == 0:
                continue

            # 识别
            if self.has_templates():
                tile_name, _ = self.recognize_tile(tile_img)
                if tile_name is None:
                    tile_name = f"unknown_{i}"
            else:
                # 无模板：仅记录位置
                tile_name = f"pos_{i}"

            center_x = x_start + (x_end - x_start) // 2
            center_y = y_start + (y_end - y_start) // 2
            results.append((tile_name, (center_x, center_y)))

        return results

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
