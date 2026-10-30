# astrbot_plugin_anime_vision

给 [AstrBot](https://github.com/AstrBotDevs/AstrBot) 用的**二次元识图插件**：收到图片自动跑本地 WD14 打标签 + trace.moe 查动画出处，结果注入到 LLM 上下文里。

**不用模型主动调用工具**，图一发，上下文里就有结果了。

---

## 它解决什么问题

AstrBot 默认把图片丢给「图片描述模型」（多模态 LLM）生成一段文字。对二次元图来说，这种做法有两个毛病：

1. **贵且慢** —— 每张图都要过一遍大模型
2. **认不出角色** —— 通用 VLM 经常把动漫角色认成别人

这个插件换了条路：

```
图片进来
   ↓
WD14 Tagger 本地打标签（0.3 秒，CPU）
   ↓
是不是真实照片？──是──→ 什么都不注入（WD14 对实拍图会瞎打标签）
   │
   否（二次元）
   ↓
角色分数 > 0.5？──是──→ 直接给角色名
   │
   否
   ↓
trace.moe 查动画出处（相似度 > 0.85 才信）
   │
   都失败
   ↓
只给标签，让 LLM 自己描述
```

---

## 功能

- ✅ **本地 WD14 打标签**：`1girl` / `blue hair` / `school uniform` 这类描述，0.3 秒出结果，纯 CPU
- ✅ **角色识别**：`hatsune miku` / `rem_(re:zero)` / `mari_(blue_archive)` 等
- ✅ **动画出处**：trace.moe 反查「这是哪部动画第几集」
- ✅ **二次元/三次元自动分流**：实拍照片不注入 WD14 标签（避免「标签对不上」）
- ✅ **修复 `<image_caption>` 标签泄漏**：AstrBot 会把图片描述包成 XML 标签塞给纯文本模型，DeepSeek 这类模型不认识，会把标签名当正文念出来 —— 插件会把它转成中文自然语言

---

## 安装

### 1. 装插件

```bash
cd AstrBot/data/plugins
git clone https://github.com/libird259/astrbot_plugin_anime_vision.git
```

### 2. 装依赖

```bash
pip install onnxruntime opencv-python numpy aiohttp
```

或者用 AstrBot 自带的环境：

```bash
AstrBot/venv/bin/pip install -r requirements.txt
```

### 3. 下载 WD14 模型

插件需要两个文件（合计约 450MB）：

```
models/wd14-tagger/
├── model.onnx          # 465MB
└── selected_tags.csv   # 标签表
```

**从 HuggingFace 下载**（`SmilingWolf/wd-swinv2-tagger-v3`）：

```bash
# model.onnx
curl -L -o models/wd14-tagger/model.onnx \
  https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx

# selected_tags.csv
curl -L -o models/wd14-tagger/selected_tags.csv \
  https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv
```

**国内下载慢的话**用镜像（把 `huggingface.co` 换成 `hf-mirror.com`，并**去掉 `/models/` 前缀**）：

```bash
curl -L -o models/wd14-tagger/model.onnx \
  "https://hf-mirror.com/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx?download=true"
```

**模型放别处也行**，设环境变量：

```bash
export WD14_MODEL_DIR=/path/to/wd14-tagger
```

### 4. 重启 AstrBot

插件会自动加载，日志里能看到：

```
[anime_vision] 插件已加载（自动注入 + 标签修正）
```

---

## 用法

**不需要任何命令**。直接发图就行。

发一张动漫图，日志会打印：

```
[anime_vision] 已注入：source=wd14 char=mari (blue archive) anime=None
```

然后 LLM 的回复里就会带上角色信息。

发实拍照片：

```
[anime_vision] 判定为真实照片，跳过 WD14 标签注入
```

---

## 配置

目前通过环境变量配置：

| 变量 | 说明 | 默认 |
|---|---|---|
| `WD14_MODEL_DIR` | WD14 模型目录 | `插件目录/models/wd14-tagger` |

---

## 注意事项（踩过的坑）

### ⚠️ 1. WD14 预处理必须保持 BGR

模型是用 OpenCV 的 BGR 顺序训练的。如果你在预处理时做了 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`，**颜色会全错** —— 金发会被认成 blue hair。

代码里已经踩过这个坑，`_preprocess()` 故意不做颜色转换。**别"修"它。**

### ⚠️ 2. WD14 是标签模型，不是角色模型

它对**角色名**的置信度普遍比标签低很多（0.3~0.6 是正常水平，不是 bug）。所以代码里角色阈值设成 0.5，低于它就走 trace.moe 或者只给标签。

### ⚠️ 3. trace.moe 对 AI 生成图无效

trace.moe 是靠**画面帧匹配**找动画出处的。AI 同人图（哪怕画得很像）匹配度通常只有 60-75%，达不到 0.85 阈值。

**这是正确行为**，不要为了"让它认出来"把阈值调低 —— 会开始乱认。

### ⚠️ 4. 真实照片必须跳过

WD14 是拿 Danbooru 动漫图训练的，对实拍照片会输出一堆离谱标签（把真人打成 `1girl, solo, anime style`）。

插件用 `realistic + photorealistic` 两个标签的分数判断：

```python
if real_score > 0.25 or (real_score > 0.12 and anime_signal < 0.3):
    is_real_photo = True   # 丢掉全部 WD14 结果
```

实测在自建数据集上区分度很好（实拍 0.50 / 动漫 0.00）。

### ⚠️ 5. AstrBot 会在 prompt 里塞 `<image_caption>` 标签

`astr_main_agent.py` 会把图片描述包成这样：

```python
TextPart(text=f"<image_caption>{caption}</image_caption>")
```

纯文本模型（DeepSeek 等）不认识这个 HTML 标签，**会把标签名当正文念出来** —— 回复变成「image_caption 那张脸我见过了」。

插件在 `@register_on_llm_request()` 里拦截并转换：

```
<image_caption>一个蓝发少女</image_caption>
  ↓
（这张图的内容：一个蓝发少女）
```

### ⚠️ 6. 注入位置很关键

一开始我把识别结果注入到 `req.system_prompt` —— **几乎没效果**，模型会忽略。

改成注入 `req.prompt`（用户消息末尾）后才生效。**这是权重差异，不是玄学。**

### ⚠️ 7. 图片 URL 可能是本地路径

AstrBot 传过来的不一定是 `http://`，可能是：

```
c:%5Cdeepseek%5C...png      # URL 编码过的本地路径
C:\deepseek\...\xxx.jpg
```

代码里做了 `unquote()` + 判断是不是本地文件 + `shutil.copy2()`。

---

## 性能

| 环节 | 耗时 |
|---|---|
| WD14 打标签（CPU） | ~0.3 秒 |
| trace.moe 查询（仅在 WD14 角色不确定时） | ~2 秒 |
| 总开销（二次元图） | 0.3~2.5 秒 |

**不需要 GPU**，onnxruntime CPU 版就够。

---

## 依赖

```
onnxruntime>=1.16
opencv-python>=4.8
numpy>=1.24
aiohttp>=3.9
```

`onnxruntime-gpu` 也可以，但没必要 —— 0.3 秒的推理用 CPU 够了。

---

## 目录结构

```
astrbot_plugin_anime_vision/
├── main.py                # 插件入口（AstrBot 生命周期 + 标签修正）
├── anime_analyzer.py      # 统一分析入口（分流逻辑）
├── anime_recognizer.py    # WD14 推理
├── metadata.yaml          # AstrBot 插件元信息
├── requirements.txt
├── README.md
└── models/
    └── wd14-tagger/       # 自己下载，不进 git
        ├── model.onnx
        └── selected_tags.csv
```

---

## 已知限制

- **AI 同人图认不出角色** —— 模型没见过，trace.moe 也匹配不上，这是数据层面的限制
- **角色识别率取决于模型覆盖** —— WD14 的 danbooru 角色表有限，冷门角色不在里面
- **只处理第一张图** —— 一次发多张图只分析第一张

---

## License

MIT

---

## 致谢

- [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) —— WD14 模型
- [trace.moe](https://trace.moe) —— 动画出处查询 API
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) —— 机器人框架
