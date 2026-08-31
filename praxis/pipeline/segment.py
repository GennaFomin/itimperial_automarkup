"""Разбиение ролика на шаги динамическим программированием.

Задача формулируется так: найти такое разбиение на k отрезков, при котором кадры внутри
каждого отрезка максимально похожи друг на друга, а разрезы попадают в моменты, где
движение затихает. Второе слагаемое и есть «физика»: у манипуляционных действий граница
шага почти всегда совпадает с паузой — рука сменила захват, поднесла деталь, отпустила.

Число отрезков не подбирается эвристиками, а выбирается точно: перебираются все k от 1 до
предела, и каждое лишнее разбиение оплачивается штрафом. Этот штраф — единственная ручка
гранулярности, и именно он защищает от пересегментации, которая убивает step-F1.
"""

from __future__ import annotations

import numpy as np

# Ниже этого значения отрезок считается дрожанием границы, а не действием.
MIN_SEGMENT_SEC = 0.8


def reduce_dimensions(features: np.ndarray, keep: int) -> np.ndarray:
    """Сжатие признаков главными компонентами по самому ролику.

    У эмбеддингов визуального энкодера больше тысячи измерений, и почти все они описывают
    сцену целиком, а не то, что меняется: соседние кадры похожи на 0.997. В сумме квадратов
    такие измерения — чистый шум, который забивает полезный сигнал. Оставляем несколько
    направлений наибольшей изменчивости внутри ролика; так же поступают в литературе по
    temporal action segmentation с признаками I3D и DINOv2.
    """
    if keep <= 0 or features.shape[1] <= keep:
        return features
    centred = features - features.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    return centred @ components[:keep].T


def normalise(features: np.ndarray) -> np.ndarray:
    """Приводим признаки к нулевому среднему и единичной дисперсии по каждому измерению.

    Без этого стоимость разбиения зависит от яркости конкретного ролика, и штраф пришлось
    бы подбирать под каждое видео отдельно.
    """
    centred = features - features.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return centred / scale


def cost_matrix(features: np.ndarray) -> np.ndarray:
    """cost[i, j] — разброс кадров внутри отрезка [i, j), в долях от разброса всего ролика.

    Нормировка обязательна: без неё стоимость растёт с длиной ролика и числом признаков,
    и штраф за отрезок пришлось бы подбирать под каждое видео заново. После нормировки
    стоимость одного отрезка на весь ролик равна примерно единице, а штраф читается как
    «какую долю разброса обязан объяснить лишний разрез».
    """
    length, dimensions = features.shape
    zeros = np.zeros((1, dimensions))
    sum1 = np.vstack([zeros, np.cumsum(features, axis=0)])
    sum2 = np.vstack([zeros, np.cumsum(features**2, axis=0)])
    total = float(length * dimensions) or 1.0

    cost = np.full((length + 1, length + 1), np.inf)
    for start in range(length):
        ends = np.arange(start + 1, length + 1)
        counts = (ends - start)[:, None]
        block1 = sum1[start + 1 :] - sum1[start]
        block2 = sum2[start + 1 :] - sum2[start]
        cost[start, start + 1 :] = np.sum(block2 - block1**2 / counts, axis=1) / total
    return cost


def boundary_scores(motion: np.ndarray, radius: int = 3) -> np.ndarray:
    """Насколько удачно резать в каждом кадре: 1 — движение затихло, 0 — самый пик.

    Настоящие провалы получают надбавку, но только настоящие: на ровном сигнале движения
    провалов нет вообще, и тогда подсказок сегментатору не будет — резать придётся по
    одному лишь виду кадров.
    """
    if len(motion) < 2 * radius + 1:
        return np.zeros_like(motion, dtype=float)

    # Дополняем краевыми значениями, а не нулями: иначе начало и конец ролика выглядят
    # самыми тихими местами, и сегментатор норовит резать по краям.
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    smooth = np.convolve(np.pad(motion, radius, mode="edge"), kernel, mode="valid")
    floor, peak = float(smooth.min()), float(smooth.max())
    span = peak - floor
    if span < 1e-9:
        return np.zeros_like(motion, dtype=float)

    scores = 1.0 - (smooth - floor) / span
    for index in range(radius, len(smooth) - radius):
        window = smooth[index - radius : index + radius + 1]
        deep_enough = float(window.max()) - smooth[index] > 0.1 * span
        if smooth[index] <= float(window.min()) + 1e-9 and deep_enough:
            scores[index] = min(1.0, scores[index] + 0.3)
    return scores


def segment(
    features: np.ndarray,
    motion: np.ndarray,
    *,
    fps: float,
    penalty: float = 0.03,
    boundary_weight: float = 0.02,
    max_segments: int = 8,
    min_segment_sec: float = MIN_SEGMENT_SEC,
    min_gain: float = 0.0,
    components: int = 0,
) -> list[tuple[int, int]]:
    """Оптимальное разбиение на отрезки. Возвращает пары индексов кадров [начало, конец).

    Перебор по числу отрезков точный: dp[k][j] — минимальная стоимость покрытия первых j
    кадров ровно k отрезками. Итоговое k выбирается по сумме стоимости и штрафа за число
    отрезков, поэтому лишний разрез появляется только если он реально что-то объясняет.
    """
    length = len(features)
    if length < 2:
        return [(0, max(length, 1))]

    minimum = max(2, int(round(min_segment_sec * fps)))
    max_segments = max(1, min(max_segments, length // minimum))
    if max_segments <= 1:
        return [(0, length)]

    cost = cost_matrix(normalise(reduce_dimensions(features, components)))
    scores = boundary_scores(motion)

    dp = np.full((max_segments + 1, length + 1), np.inf)
    back = np.zeros((max_segments + 1, length + 1), dtype=int)
    dp[0, 0] = 0.0

    for count in range(1, max_segments + 1):
        for end in range(count * minimum, length + 1):
            # Разрез в точке start оплачивается стоимостью отрезка и скидкой за то,
            # что в этом кадре движение затихло.
            starts = np.arange((count - 1) * minimum, end - minimum + 1)
            if not len(starts):
                continue
            bonus = np.where(
                starts > 0, boundary_weight * scores[np.minimum(starts, length - 1)], 0.0
            )
            totals = dp[count - 1, starts] + cost[starts, end] - bonus
            best = int(np.argmin(totals))
            if totals[best] < dp[count, end]:
                dp[count, end] = totals[best]
                back[count, end] = starts[best]

    totals = dp[:, length] + penalty * np.arange(max_segments + 1)
    totals[0] = np.inf
    best_count = int(np.argmin(totals))

    bounds: list[tuple[int, int]] = []
    end = length
    for count in range(best_count, 0, -1):
        start = int(back[count, end])
        bounds.append((start, end))
        end = start
    return prune_boundaries(list(reversed(bounds)), cost, min_gain)


def prune_boundaries(
    bounds: list[tuple[int, int]], cost: np.ndarray, min_gain: float
) -> list[tuple[int, int]]:
    """Убирает границы, которые ничего не объясняют.

    Глобального штрафа мало: он одинаков для всех роликов, а «лишний» разрез бывает
    дешёвым в одном ролике и осмысленным в другом. Здесь каждая граница проверяется
    локально — насколько разрез двух соседних отрезков лучше их объединения. Если выигрыш
    меньше порога, соседи сливаются, и так до тех пор, пока слабые границы не кончатся.
    """
    if min_gain <= 0:
        return bounds

    working = list(bounds)
    while len(working) > 1:
        gains = []
        for index in range(len(working) - 1):
            left, right = working[index], working[index + 1]
            merged = cost[left[0], right[1]]
            separate = cost[left[0], left[1]] + cost[right[0], right[1]]
            gains.append((merged - separate) / merged if merged > 1e-12 else 0.0)

        weakest = int(np.argmin(gains))
        if gains[weakest] >= min_gain:
            break
        left, right = working[weakest], working[weakest + 1]
        working[weakest : weakest + 2] = [(left[0], right[1])]
    return working


def pick_keyframe(features: np.ndarray, motion: np.ndarray, start: int, end: int) -> int:
    """Кадр, наиболее типичный для отрезка: ближайший к среднему, но не в момент размазанного движения."""
    block = features[start:end]
    if len(block) == 0:
        return start
    centre = block.mean(axis=0)
    distance = np.linalg.norm(block - centre, axis=1)
    distance = distance / (distance.max() or 1.0)

    blur = motion[start:end]
    blur = blur / (blur.max() or 1.0)

    return start + int(np.argmin(distance + 0.5 * blur))
