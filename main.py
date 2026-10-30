import asyncio
import os
import re
import sys

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.star.register.star_handler import register_on_llm_request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anime_analyzer  # noqa: E402

TEMP_DIR = r"C:\deepseek\astrbot-deploy\tmp-anime"
os.makedirs(TEMP_DIR, exist_ok=True)

# AstrBot 会把图片描述包成 <image_caption>...</image_caption> 给非视觉模型，
# DeepSeek 不认识这个标签，会把标签名当正文念出来。这里统一转成自然语言。
_CAPTION_TAG_RE = re.compile(r"<image_caption>(.*?)</image_caption>", re.DOTALL)
_PLACEHOLDERS = {
    "[Image Captioning Failed]": "（这张图没能解析出来）",
    "[Image]": "（附带一张图片）",
}


@register(
    "astrbot_plugin_anime_vision",
    "大肥鱼",
    "二次元图片自动分析：WD14 标签 + trace.moe 出处，并修正 AstrBot 的图片描述标签",
    "1.2.0",
    "",
)
class AnimeVisionPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        logger.info("[anime_vision] 插件已加载（自动注入 + 标签修正）")

    async def _get_image_urls(self, event: AstrMessageEvent):
        urls = []
        try:
            for comp in event.message_obj.message:
                if isinstance(comp, Comp.Image):
                    u = getattr(comp, "url", None) or getattr(comp, "file", None)
                    if u:
                        urls.append(u)
        except Exception:
            pass
        return urls

    async def _download(self, url, idx=0):
        import shutil
        from urllib.parse import unquote

        import aiohttp

        raw = unquote(url)
        is_http = raw.lower().startswith("http://") or raw.lower().startswith("https://")

        ext = ".jpg"
        for e in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            if raw.lower().split("?")[0].endswith(e):
                ext = e
                break
        out = os.path.join(TEMP_DIR, f"auto_{os.getpid()}_{idx}{ext}")

        if not is_http:
            try:
                if os.path.isfile(raw):
                    shutil.copy2(raw, out)
                    return out
                logger.warning(f"[anime_vision] 本地文件不存在: {raw}")
                return None
            except Exception as e:
                logger.warning(f"[anime_vision] 复制失败: {e}")
                return None

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as s:
                async with s.get(raw) as r:
                    if r.status != 200:
                        return None
                    with open(out, "wb") as f:
                        f.write(await r.read())
            return out
        except Exception as e:
            logger.warning(f"[anime_vision] 下载失败: {e}")
            return None

    def _fix_caption_tags(self, req: ProviderRequest) -> int:
        """把 <image_caption>...</image_caption> 和占位符转成自然语言。返回修改条数。"""
        changed = 0

        def _repl(m):
            inner = (m.group(1) or "").strip()
            return f"（这张图的内容：{inner}）"

        parts = getattr(req, "extra_user_content_parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if not isinstance(text, str) or not text:
                continue
            new_text = text
            new_text = _CAPTION_TAG_RE.sub(_repl, new_text)
            for old, new in _PLACEHOLDERS.items():
                if old in new_text:
                    new_text = new_text.replace(old, new)

            if new_text != text:
                try:
                    part.text = new_text
                    changed += 1
                except Exception:
                    pass

        p = getattr(req, "prompt", None)
        if isinstance(p, str) and "<image_caption>" in p:
            req.prompt = _CAPTION_TAG_RE.sub(_repl, p)
            changed += 1

        if changed:
            logger.info(f"[anime_vision] 已修正 {changed} 处图片描述标签")
        return changed

    @register_on_llm_request()
    async def inject_anime_context(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """LLM 请求前：修正描述标签 + 自动跑 WD14/trace.moe 注入结果。"""

        # 第一步：无条件修正标签格式
        try:
            self._fix_caption_tags(req)
        except Exception as e:
            logger.warning(f"[anime_vision] 标签修正失败: {e}")

        # 第二步：拿图片
        urls = list(getattr(req, "image_urls", None) or [])
        if not urls:
            urls = await self._get_image_urls(event)
        if not urls:
            return

        local = await self._download(urls[0], 0)
        if not local:
            return

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(anime_analyzer.analyze, local), timeout=90
            )
        except asyncio.TimeoutError:
            logger.warning("[anime_vision] 分析超时")
            return
        except Exception as e:
            logger.error(f"[anime_vision] 分析失败: {e}")
            return
        finally:
            try:
                os.remove(local)
            except Exception:
                pass

        # 三次元（真实照片）：不注入 WD14 标签，让视觉模型自己的描述说话
        if result.get("is_real_photo"):
            logger.info("[anime_vision] 判定为真实照片，跳过 WD14 标签注入")
            return

        lines = ["【二次元识图结果】"]
        if result.get("character"):
            score = (result.get("wd14") or {}).get("top_char_score", 0)
            lines.append(f"- 角色：{result['character']}（{score:.0%}）")
        if result.get("anime"):
            tm = result.get("trace_moe") or {}
            ep = tm.get("episode")
            sim = tm.get("similarity", 0)
            lines.append(
                f"- 出自动画：《{result['anime']}》"
                + (f" 第 {ep} 集" if ep else "")
                + f"（{sim:.0%}）"
            )
        if result.get("tags"):
            lines.append("- 画面特征：" + "、".join(result["tags"][:15]))
        if len(lines) == 1:
            return

        text = "\n".join(lines)
        try:
            if req.prompt:
                req.prompt = str(req.prompt) + "\n\n" + text
            else:
                req.prompt = text
        except Exception as e:
            logger.warning(f"[anime_vision] 注入失败: {e}")
            return

        logger.info(
            f"[anime_vision] 已注入：source={result.get('source')} "
            f"char={result.get('character')} anime={result.get('anime')}"
        )

    async def terminate(self):
        logger.info("[anime_vision] 插件已卸载")
