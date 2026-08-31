#!/usr/bin/env python3
"""Валидационный набор из Charades — единственный публичный, совпадающий с профилем кейса.

Зачем ещё один набор. Assembly101 снят статичной камерой от третьего лица, но его
таксономия — почти одинаковые детали конструктора, и назвать предмет там нельзя даже
человеку. EPIC-KITCHENS даёт различимые бытовые предметы, но снят налобной камерой,
которая движется в каждом кадре, — а кейс прямо требует статичную. Charades закрывает
обе дыры сразу: съёмка на закреплённый телефон в жилой комнате, один человек, бытовые
действия, метки вида «глагол + предмет» с временными границами и компактная таксономия
из 33 глаголов и 38 предметов — примерно такого размера словарь и дадут организаторы.

Видео лежат в одном архиве на 13 ГБ, качать его целиком незачем: S3 отдаёт куски по
range-запросам, поэтому архив открывается как файл, а из него достаются только нужные
ролики. Так же мы уже брали окна из многогигабайтных записей Assembly101.

    python scripts/make_charades_devset.py --out data/devset_charades --clips 60
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from praxis.schema import Annotation, Provenance, Step, VideoMeta

VIDEO_ZIP = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"


class RemoteFile(io.RawIOBase):
    """Файловый объект поверх HTTP-range: zipfile сам разберёт оглавление и распакует."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.length = int(response.headers["Content-Length"])

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.length}[whence]
        self.position = max(0, min(self.length, base + offset))
        return self.position

    def tell(self) -> int:
        return self.position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        size = min(len(buffer), self.length - self.position)
        if size <= 0:
            return 0
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{self.position + size - 1}"}
        )
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def content_crop(path: Path) -> str | None:
    """Рамка полезного кадра.

    Часть Charades снята телефоном вертикально и уложена в горизонтальный контейнер:
    половина кадра — чёрные поля. И видеоэнкодер, и языковая модель тратят на них
    разрешение впустую, поэтому поля срезаются. Для обычной съёмки cropdetect вернёт
    весь кадр, и обрезка ничего не изменит.
    """
    probe = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-vf", "cropdetect=24:2:0",
         "-frames:v", "60", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    found = [line for line in probe.stderr.splitlines() if "crop=" in line]
    return found[-1].split("crop=")[-1].strip() if found else None


def taxonomy(root: Path) -> tuple[dict[str, tuple[str, str]], list[str], list[str]]:
    """Класс действия → (глагол, предмет), плюс полные списки для словаря."""
    verbs = {
        line.split(" ", 1)[0]: line.split(" ", 1)[1].strip()
        for line in (root / "Charades_v1_verbclasses.txt").read_text().splitlines()
        if line.strip()
    }
    objects = {
        line.split(" ", 1)[0]: line.split(" ", 1)[1].strip()
        for line in (root / "Charades_v1_objectclasses.txt").read_text().splitlines()
        if line.strip()
    }
    mapping = {}
    for line in (root / "Charades_v1_mapping.txt").read_text().splitlines():
        if not line.strip():
            continue
        action, obj, verb = line.split()
        mapping[action] = (verbs[verb], objects[obj])
    return mapping, sorted(set(verbs.values())), sorted(set(objects.values()) - {"None"})


def parse_actions(field: str, mapping: dict) -> list[tuple[float, float, str, str | None]]:
    """Строка «c092 11.90 21.20;…» в список сегментов, отсортированных по времени."""
    result = []
    for chunk in field.split(";"):
        parts = chunk.split()
        if len(parts) != 3 or parts[0] not in mapping:
            continue
        verb, obj = mapping[parts[0]]
        result.append((float(parts[1]), float(parts[2]), verb, None if obj == "None" else obj))
    return sorted(result, key=lambda item: (item[0], item[1]))


def label_sets(actions: list, kept: list, share: float = 0.5) -> list[list[list[str]]]:
    """Все метки, покрывающие шаг, а не только оставленная.

    Charades размечен многометочно: один интервал бывает подписан и «hold bag», и
    «put clothes», и «tidy floor» — это три верных ответа, а не два неверных. Жадный
    отбор непересекающихся шагов оставляет одну метку и выбрасывает остальные, и модель,
    ответившая выброшенной, получала бы ошибку ни за что. Поэтому рядом с эталоном
    сохраняется полный набор допустимых меток на каждый шаг.
    """
    result = []
    for start, end, verb, obj in kept:
        span = max(end - start, 1e-6)
        allowed = {(verb, obj or "")}
        for other_start, other_end, other_verb, other_obj in actions:
            overlap = min(end, other_end) - max(start, other_start)
            if overlap > share * span:
                allowed.add((other_verb, other_obj or ""))
        result.append(sorted([verb, obj] for verb, obj in allowed))
    return result


def drop_overlaps(actions: list, min_len: float) -> list:
    """Кейс требует непересекающихся шагов, а в Charades действия наложены друг на друга.

    Берём жадно по времени начала и пропускаем всё, что залезает на уже взятое: остаётся
    последовательность, какую и размечал бы человек по инструкции кейса.
    """
    kept: list = []
    for start, end, verb, obj in actions:
        if end - start < min_len:
            continue
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, verb, obj))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("data/charades"))
    parser.add_argument("--clips", type=int, default=60)
    parser.add_argument("--min-dur", type=float, default=6.0)
    parser.add_argument("--max-dur", type=float, default=30.0)
    parser.add_argument("--min-step", type=float, default=1.5)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    mapping, all_verbs, all_objects = taxonomy(args.annotations)

    rows = list(csv.DictReader((args.annotations / "Charades_v1_train.csv").open()))
    chosen = []
    for row in rows:
        try:
            length = float(row["length"])
        except (TypeError, ValueError):
            continue
        if not (args.min_dur <= length <= args.max_dur):
            continue
        # Ролики, размеченные без проверки или помеченные как мутные, нам только шумят.
        if row.get("verified") != "Yes" or int(row.get("quality") or 0) < 5:
            continue
        actions = parse_actions(row["actions"], mapping)
        steps = drop_overlaps(actions, args.min_step)
        if not (args.min_steps <= len(steps) <= args.max_steps):
            continue
        chosen.append((row["id"], length, steps, label_sets(actions, steps)))
        if len(chosen) >= args.clips:
            break

    clips_dir, gt_dir = args.out / "clips", args.out / "gt"
    clips_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"отобрано {len(chosen)} роликов, открываю архив…", flush=True)
    archive = zipfile.ZipFile(io.BufferedReader(RemoteFile(VIDEO_ZIP), buffer_size=1 << 20))
    inside = {Path(name).stem: name for name in archive.namelist() if name.endswith(".mp4")}

    written = 0
    allowed: dict[str, list] = {}
    for video_id, length, steps, sets in chosen:
        member = inside.get(video_id)
        if member is None:
            continue
        target = clips_dir / f"{video_id}.mp4"
        try:
            with archive.open(member) as source, tempfile.NamedTemporaryFile(suffix=".mp4") as raw:
                raw.write(source.read())
                raw.flush()
                crop = content_crop(Path(raw.name))
                chain = f"crop={crop}," if crop else ""
                # Кейс требует не меньше 720p, а Charades роздан в 480 — поднимаем по высоте.
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", raw.name,
                     "-vf", f"{chain}scale=-2:720", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "20", "-an", str(target)],
                    check=True,
                )
        except (KeyError, OSError, subprocess.CalledProcessError) as error:
            print(f"  {video_id}: пропуск ({error})", flush=True)
            continue

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate",
             "-show_entries", "format=duration", "-of", "json", str(target)],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(probe.stdout)
        stream, fmt = info["streams"][0], info["format"]
        numerator, _, denominator = stream["avg_frame_rate"].partition("/")
        duration = float(fmt["duration"])

        annotation = Annotation(
            video=VideoMeta(
                id=video_id,
                filename=target.name,
                duration_sec=duration,
                fps=float(numerator) / float(denominator or 1),
                width=int(stream["width"]),
                height=int(stream["height"]),
            ),
            steps=[
                Step(
                    id=index,
                    start_sec=round(min(start, duration), 3),
                    end_sec=round(min(end, duration), 3),
                    action=verb,
                    object=obj,
                    confidence=1.0,
                )
                for index, (start, end, verb, obj) in enumerate(steps)
                if start < duration
            ],
            provenance=Provenance(
                app_version="devset", pipeline="charades", vocabulary="charades",
                models={}, backend="charades", processing_sec=0.0,
            ),
        )
        (gt_dir / f"{video_id}.json").write_text(
            annotation.model_dump_json(indent=1), encoding="utf-8"
        )
        allowed[video_id] = [sets[step.id] for step in annotation.steps]
        written += 1
        print(f"  {written}/{len(chosen)} {video_id}: {len(annotation.steps)} шагов", flush=True)

    # Допустимые пары, а не декартово произведение: в Charades осмысленных сочетаний 157
    # из 1254 возможных. Закрытый список пар — ровно то, что выдаёт заказчик разметки,
    # и он же не даёт модели предложить «помыть диван».
    grouped: dict[str, list[str]] = {}
    for verb, obj in mapping.values():
        grouped.setdefault(verb, []).append(obj if obj != "None" else "")
    lines = [
        "name: charades",
        "description: человек занят бытовыми делами дома",
        "actions:",
        *(f"  - {verb}" for verb in all_verbs),
        "objects:",
        *(f"  - {obj}" for obj in all_objects),
        "pairs:",
    ]
    for verb in sorted(grouped):
        nouns = sorted({noun for noun in grouped[verb] if noun})
        if nouns:
            lines.append(f"  {verb}:")
            lines.extend(f"    - {noun}" for noun in nouns)
    (args.out / "vocab_charades.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "allowed.json").write_text(
        json.dumps(allowed, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nготово: {written} роликов в {args.out}")


if __name__ == "__main__":
    main()
