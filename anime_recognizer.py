r"""二次元角色识别（WD14 Tagger 本地推理）。
- 模型: SmilingWolf/wd-swinv2-tagger-v3（onnx, 467MB）
- 能识别角色名（如 hatsune miku / rem_(re:zero)）+ 全部标签
- CPU 推理约 0.3 秒/张
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# 模型目录：优先环境变量 WD14_MODEL_DIR，否则用插件目录下的 models/wd14-tagger
MODEL_DIR = Path(
    os.environ.get(
        "WD14_MODEL_DIR",
        str(Path(__file__).resolve().parent / "models" / "wd14-tagger"),
    )
)
MODEL_PATH = MODEL_DIR / "model.onnx"
TAGS_PATH = MODEL_DIR / "selected_tags.csv"

# 图片输入尺寸（WD tagger 固定 448）
INPUT_SIZE = 448

_session = None
_tag_names = None
_tag_categories = None


def _load_tags():
    global _tag_names, _tag_categories
    if _tag_names is not None:
        return
    names, cats = [], []
    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"])
            cats.append(int(row["category"]))
    _tag_names = names
    _tag_categories = cats


def get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
    return _session


def _preprocess(image_path):
    """WD tagger 预处理：pad 到正方形 → resize 448 → 保持 BGR（模型训练用的就是 BGR）。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    # pad 成正方形（填白）
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = img

    img = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32)  # 保持 0-255
    return arr[None, ...]  # [1, 448, 448, 3]


def predict(image_path, threshold=0.35):
    """返回 (角色列表, 其他标签列表)。"""
    _load_tags()
    sess = get_session()
    inp = _preprocess(image_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    probs = sess.run([output_name], {input_name: inp})[0][0]

    characters = []
    general = []
    for i, p in enumerate(probs):
        if p < threshold:
            continue
        name = _tag_names[i]
        score = float(p)
        # category: 0=general, 4=character
        if _tag_categories[i] == 4:
            # 角色名转成可读形式: rem_(re:zero) → rem (re:zero)
            readable = name.replace("_", " ").replace("(", " (").replace(")", ")")
            characters.append({"name": readable, "raw": name, "score": round(score, 3)})
        else:
            general.append({"name": name.replace("_", " "), "score": round(score, 3)})

    characters.sort(key=lambda x: -x["score"])
    general.sort(key=lambda x: -x["score"])
    return characters, general


def realism_scores(image_path):
    """单独算写实度和二次元信号（不受阈值过滤，用于真假判断）。"""
    _load_tags()
    sess = get_session()
    inp = _preprocess(image_path)
    probs = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: inp})[0][0]

    want = {"realistic", "photorealistic", "1girl", "1boy", "anime_style", "photo"}
    out = {}
    for i, p in enumerate(probs):
        n = _tag_names[i]
        if n in want:
            out[n] = float(p)
    return out


def identify(image_path, char_threshold=0.35, tag_threshold=0.35, top_tags=15):
    """完整识别：返回角色 + 前 N 个标签 + 写实度（用于判断二次元/三次元）。"""
    chars, tags = predict(image_path, threshold=min(char_threshold, tag_threshold))
    chars = [c for c in chars if c["score"] >= char_threshold]
    tags = [t for t in tags if t["score"] >= tag_threshold]

    # 用未过滤的原始分数判断真假
    raw = realism_scores(image_path)
    realistic = raw.get("realistic", 0.0)
    photorealistic = raw.get("photorealistic", 0.0)
    anime_signal = max(raw.get("1girl", 0.0), raw.get("1boy", 0.0))
    real_score = max(realistic, photorealistic)

    return {
        "characters": chars,
        "top_char": chars[0]["name"] if chars else None,
        "top_char_score": chars[0]["score"] if chars else 0,
        "tags": [t["name"] for t in tags[:top_tags]],
        "tag_scores": {t["name"]: t["score"] for t in tags[:top_tags]},
        "realistic_score": round(realistic, 3),
        "photorealistic_score": round(photorealistic, 3),
        "real_score": round(real_score, 3),
        "anime_signal": round(anime_signal, 3),
        "is_real_photo": bool(real_score > 0.25 or (real_score > 0.12 and anime_signal < 0.3)),
    }




if __name__ == "__main__":
    import io
    import json
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python anime_recognizer.py <图片路径>")
        sys.exit(0)

    print(json.dumps(identify(sys.argv[1]), ensure_ascii=False, indent=2))
