r"""二次元图片统一分析：WD14 标签 + trace.moe 动画出处。
策略：
1. WD14 打标签（快，0.3s）—— 永远都做，标签对所有动漫图有效
2. 角色分数够高（>0.5）→ 直接用 WD14 的角色识别
3. 否则试 trace.moe —— 只有真实动画截图才会高分（>0.85）
4. 都不行 → 只返回标签，让 LLM 自己描述
"""
import io
import json
import sys
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import anime_recognizer

TRACE_MOE_URL = "https://api.trace.moe/search?anilistInfo=true"
TM_CHAR_THRESHOLD = 0.5   # WD14 角色分超过这个，直接信它
TM_ANIME_THRESHOLD = 0.85  # trace.moe 相似度超过这个，认为是真实截图


def trace_moe_upload(image_path, timeout=45):
    """上传到 trace.moe，返回最优结果或 None。"""
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
    except Exception:
        return None

    boundary = uuid.uuid4().hex
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.{ext}\"\r\nContent-Type: {mime}\r\n\r\n".encode())
    parts.append(img_data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(TRACE_MOE_URL, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("User-Agent", "DaFeiYu/1.0")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if data.get("error"):
        return {"error": data["error"]}

    results = data.get("result") or []
    if not results:
        return None

    best = results[0]
    anilist = best.get("anilist") or {}
    titles = anilist.get("title") or {}
    chars = anilist.get("characters") or []

    return {
        "similarity": round(float(best.get("similarity", 0)), 3),
        "anime_romaji": titles.get("romaji"),
        "anime_native": titles.get("native"),
        "anime_english": titles.get("english"),
        "episode": best.get("episode"),
        "from_seconds": round(float(best.get("from", 0)), 1),
        "characters": chars[:5] if isinstance(chars, list) else [],
    }


def analyze(image_path):
    """完整分析：返回统一结构。

    第一步判断二次元/三次元：
      - 三次元（真实照片）→ 丢掉 WD14 标签（有噪音），只标记 real_photo
      - 二次元            → 走 WD14 角色识别 + trace.moe 查出处
    """
    result = {
        "wd14": None,
        "trace_moe": None,
        "character": None,
        "anime": None,
        "tags": [],
        "source": "none",
        "summary": "",
        "is_real_photo": False,
    }

    # 1) WD14
    try:
        wd = anime_recognizer.identify(image_path)
        result["wd14"] = wd
        result["tags"] = wd.get("tags", [])
        result["is_real_photo"] = bool(wd.get("is_real_photo"))
    except Exception as e:
        result["wd14_error"] = str(e)
        return result

    # 1.5) 三次元分流：不注入标签，让 Qwen3-VL 的自然描述作为唯一依据
    if result["is_real_photo"]:
        result["source"] = "real_photo"
        result["tags"] = []  # 实拍图上 WD14 标签不可信，全部丢弃
        result["character"] = None
        result["anime"] = None
        result["summary"] = "这是真实照片"
        return result

    top_char = wd.get("top_char")
    top_score = wd.get("top_char_score", 0)

    if top_char and top_score >= TM_CHAR_THRESHOLD:
        result["character"] = top_char
        result["source"] = "wd14"
        result["summary"] = f"角色：{top_char}（置信度 {top_score:.2f}）"
        return result

    # 2) trace.moe（只有 WD14 不确定时才调用，省一次网络请求）
    tm = trace_moe_upload(image_path)
    result["trace_moe"] = tm

    if tm and not tm.get("error") and tm.get("similarity", 0) >= TM_ANIME_THRESHOLD:
        result["anime"] = tm.get("anime_native") or tm.get("anime_romaji")
        result["source"] = "trace_moe"
        result["summary"] = f"出自《{result['anime']}》第 {tm.get('episode')} 集，相似度 {tm['similarity']:.0%}"
        return result

    # 3) 都不行，只给标签
    result["source"] = "tags_only"

    tag_str = "、".join(result["tags"][:12])
    result["summary"] = f"无法确定出处，特征：{tag_str}"
    if top_char:
        result["low_confidence_character"] = f"{top_char}（{top_score:.3f}，不可信）"
    return result


if __name__ == "__main__":
    import sys as _s

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if len(_s.argv) < 2:
        print("用法: python anime_analyzer.py <图片路径>")
        _s.exit(0)
    print(json.dumps(analyze(_s.argv[1]), ensure_ascii=False, indent=2))
