# -*- coding: utf-8 -*-
"""
卫星遥感图像鲸鱼检测 · 蓝白对话式界面
------------------------------------------------
视觉：白底 + 深蓝点缀（#1e5a99），18px 圆角气泡 + 淡投影
交互：拖图进去就自动识别，不用再敲指令

人性化细节：
  1. 上传即识别 —— 不再需要输入「开始识别」
  2. 每条消息带时间，知道这次是什么时候做的
  3. 每次识别自动存档到 records/，关掉页面也能翻回来
  4. 首屏三条上手提示
  5. 检测中报「第几块 / 共几块」
  6. 结果给出每个目标的置信度条形图
  7. 没检测到时给具体原因和下一步建议
  8. 单条删除 + 一键清空
  9. 侧边栏一键恢复默认参数
 10. 窄屏（手机）气泡自动收窄

原有能力全部保留：切片推理、NMS 去重、结果图下载、参数可调。
"""

import io
import os
import re
import json
import glob
import time
import html
import base64
import random
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="卫星遥感鲸鱼检测",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 识别记录存放目录（和 app.py 同级）
RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records")
os.makedirs(RECORDS_DIR, exist_ok=True)


# ============================================================
# 一、蓝白主题 CSS
# ============================================================
CSS = """
<style>
/* ---------------- 全局：白底 + 深蓝点缀 ---------------- */
.stApp { background: #ffffff !important; }
body, .stMarkdown, p, span, label, .stCaption { color: #1f2937 !important; }
[data-testid="stHeader"] { background: #ffffff !important; }

/* ---------------- 侧边栏：极浅的蓝灰，和主区分开 ---------------- */
[data-testid="stSidebar"] {
    background: #f5f7fa !important;
    border-right: 1px solid #e3e8ef;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #4b5563 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid #dbe3ee !important;
    border-radius: 12px !important;
}

/* ---------------- 底部输入框 ---------------- */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: #ffffff !important; }
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid #dbe3ee !important;
    border-radius: 18px !important;
    box-shadow: 0 2px 10px rgba(30, 90, 153, .06) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #1e5a99 !important;
    box-shadow: 0 0 0 3px rgba(30, 90, 153, .10) !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #1f2937 !important;
    caret-color: #1e5a99 !important;
}
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9ca3af !important; }

/* ---------------- 按钮 ---------------- */
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"],
.stDownloadButton button {
    border-radius: 16px !important;
    border: none !important;
    background: #1e5a99 !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-secondary"]:hover,
.stDownloadButton button:hover {
    background: #17497d !important;
}
/* 删除这类小按钮：平时低调，悬停才提醒是危险操作 */
[data-testid="stBaseButton-tertiary"] {
    background: transparent !important;
    color: #c9d1dc !important;
    border: none !important;
    padding: 2px 6px !important;
    min-height: 26px !important;
    border-radius: 9px !important;
    font-size: 14px !important;
    line-height: 1 !important;
    box-shadow: none !important;
}
[data-testid="stBaseButton-tertiary"]:hover {
    background: #fef2f2 !important;
    color: #e11d48 !important;
}
.stProgress > div > div > div > div { background: #1e5a99 !important; }

/* ---------------- 对话气泡 ---------------- */
.wb-row { display: flex; gap: 10px; margin: 16px 0; align-items: flex-start; }
.wb-row.user { flex-direction: row-reverse; }

.wb-avatar {
    width: 34px; height: 34px; border-radius: 12px;
    flex: 0 0 34px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; letter-spacing: .5px;
    color: #ffffff;
}
.wb-avatar.user { background: #1e5a99; }
.wb-avatar.ai   { background: #ffffff; color: #1e5a99; border: 1px solid #dbe3ee; }

.wb-bubble {
    max-width: 74%;
    padding: 13px 17px;
    font-size: 15px;
    line-height: 1.65;
    word-break: break-word;
    border-radius: 18px;
}
/* 用户：深蓝底白字，右下角收一点 */
.wb-bubble.user {
    background: #1e5a99;
    color: #ffffff;
    border-radius: 18px 18px 5px 18px;
    box-shadow: 0 4px 16px rgba(30, 90, 153, .16);
}
/* AI：白卡片，淡蓝边，左下方的角收一点 */
.wb-bubble.ai {
    background: #ffffff;
    color: #1f2937;
    border: 1px solid #e0e8f3;
    border-radius: 18px 18px 18px 5px;
    box-shadow: 0 6px 20px rgba(30, 90, 153, .06);
}

/* 气泡内图片 */
.wb-bubble img { width: 100%; display: block; margin-top: 9px; border-radius: 14px; }
/* 气泡内小字 */
.wb-meta { font-size: 12px; color: #6b7280; margin-top: 8px; line-height: 1.6; }
.wb-bubble.user .wb-meta { color: rgba(255, 255, 255, .78); }

/* 置信度条形图 */
.wb-conf { margin-top: 10px; display: flex; flex-direction: column; gap: 7px; }
.wb-conf-row { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.wb-conf-name { flex: 0 0 46px; color: #6b7280; }
.wb-conf-track {
    flex: 1; height: 6px; background: #eef2f7;
    border-radius: 3px; overflow: hidden;
}
.wb-conf-fill { height: 100%; border-radius: 3px; }
.wb-conf-val { flex: 0 0 34px; text-align: right; color: #1f2937; font-variant-numeric: tabular-nums; }

/* 上手提示条 */
.wb-tips {
    background: #f5f8fc;
    border: 1px solid #e0e8f3;
    border-radius: 18px;
    padding: 14px 18px;
    margin: 10px 0 18px;
    font-size: 13px;
    color: #4b5563;
    line-height: 1.9;
}
.wb-tips b { color: #1e5a99; }

/* 时间戳 */
.wb-time { font-size: 11px; color: #9ca3af; margin-top: 6px; }

/* 历史记录卡片里的元信息 */
.hist-meta { font-size: 12px; color: #6b7280; }

/* 窄屏：气泡放宽到 92%，避免被挤成一条缝 */
@media (max-width: 768px) {
    .wb-bubble { max-width: 92%; font-size: 14px; }
    .wb-avatar { width: 28px; height: 28px; flex: 0 0 28px; font-size: 10px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# 二、识别记录：存盘 / 读取 / 删除
# ============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_record(p, boxes, elapsed, n_tiles, result, w, h, conf_thres, iou_thres):
    """把一次识别写进 records/：一张带框图 + 一张缩略图 + 一个 json。
       关掉页面、重启电脑都还在。"""
    ts = datetime.now()
    rid = ts.strftime("%Y%m%d_%H%M%S")
    base, i = rid, 1
    while os.path.exists(os.path.join(RECORDS_DIR, f"{base}.json")):
        base = f"{rid}_{i}"
        i += 1

    result_file = f"{base}_result.jpg"
    cv2.imwrite(os.path.join(RECORDS_DIR, result_file), result)

    thumb_file = f"{base}_thumb.jpg"
    thumb = result.copy()
    tw = 640
    if thumb.shape[1] > tw:
        thumb = cv2.resize(thumb, (tw, int(thumb.shape[0] * tw / thumb.shape[1])))
    cv2.imwrite(os.path.join(RECORDS_DIR, thumb_file), thumb,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82])

    rec = {
        "id": base,
        "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "name": p["name"],
        "w": w, "h": h,
        "count": len(boxes),
        "confs": [round(float(b[4]), 3) for b in sorted(boxes, key=lambda t: -t[4])],
        "elapsed": round(elapsed, 1),
        "tiles": n_tiles,
        "conf_thres": round(float(conf_thres), 2),
        "iou_thres": round(float(iou_thres), 2),
        "result_file": result_file,
        "thumb_file": thumb_file,
    }
    with open(os.path.join(RECORDS_DIR, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec


def load_records(limit=None):
    """按时间倒序读回所有记录。"""
    recs = []
    for fp in glob.glob(os.path.join(RECORDS_DIR, "*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                recs.append(json.load(f))
        except Exception:
            continue
    recs.sort(key=lambda r: r.get("time", ""), reverse=True)
    return recs[:limit] if limit else recs


def delete_record(rec):
    """删 json、结果图、缩略图三件套。"""
    for key in ("result_file", "thumb_file"):
        fp = os.path.join(RECORDS_DIR, rec.get(key, "") or "")
        if os.path.isfile(fp):
            os.remove(fp)
    jp = os.path.join(RECORDS_DIR, f"{rec['id']}.json")
    if os.path.isfile(jp):
        os.remove(jp)


# ============================================================
# 三、侧边栏
# ============================================================
with st.sidebar:
    st.subheader("检测参数")
    model_path = st.text_input("模型权重路径", value="best.pt")

    conf_thres = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05, key="conf_t")
    iou_thres = st.slider("NMS 去重 IoU", 0.30, 0.90, 0.45, 0.05, key="iou_t")
    overlap = st.slider("切片重叠像素", 0, 128, 64, 16, key="ov_t")
    TILE = 512
    st.caption(f"切片 {TILE}×{TILE}　步长 {TILE - overlap}")

    # Streamlit 规定：已实例化的 slider 不能事后改值，
    # 所以这几个按钮必须走 on_click 回调，回调在下一轮渲染前执行。
    def _reset_params():
        st.session_state.conf_t = 0.25
        st.session_state.iou_t = 0.45
        st.session_state.ov_t = 64

    def _set_conf(v):
        st.session_state.conf_t = v

    st.button("恢复默认参数", use_container_width=True, on_click=_reset_params)

    st.caption("一键切模式")
    c1, c2 = st.columns(2)
    with c1:
        st.button("稳妥", use_container_width=True, on_click=_set_conf, args=(0.40,),
                  help="阈值 0.40：只留很有把握的，适合出报告")
    with c2:
        st.button("灵敏", use_container_width=True, on_click=_set_conf, args=(0.15,),
                  help="阈值 0.15：宁可多报，适合先找线索")

    # ---- 可选：接一个大模型来做闲聊/答疑 ----
    with st.expander("接入大模型（可选）"):
        st.text_input("API Key", value="", type="password", key="api_key",
                      help="不填也能用，会走内置的离线问答")
        st.text_input("接口地址", value="https://api.deepseek.com/v1", key="api_url",
                      help="OpenAI 兼容接口，豆包/DeepSeek/Kimi 都行")
        st.text_input("模型名", value="deepseek-chat", key="api_model")
        st.caption("填了就走真模型；不填用内置问答，完全离线，不消耗额度。")

    st.checkbox("联网搜索", value=True, key="web_on",
                help="内置知识答不上来时，自动去 360 搜索找结果")

    st.divider()

    n_rec = len(load_records())
    st.caption(f"已存档 {n_rec} 次识别，存在 records 文件夹里。")

    if st.button("清空当前对话", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_img = None
        st.rerun()
    st.caption("只清界面上的对话，已存档的记录不受影响。")


# ============================================================
# 四、模型
# ============================================================
if not os.path.exists(model_path):
    st.error(f"找不到权重文件：{model_path}　请检查路径是否写对")
    st.stop()


@st.cache_resource(show_spinner="正在加载模型…")
def load_model(path):
    return YOLO(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

# 会话状态提前初始化：下面侧边栏的"选图即识别"要用到 history
for _k, _v in (("history", []), ("last_img", None), ("quick_sig", "")):
    st.session_state.setdefault(_k, _v)


# ============================================================
# 五、工具函数
# ============================================================
def tile_starts(length, tile, step):
    """切片起始坐标，保证图像边缘被完整覆盖。"""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def img_to_b64(img, max_w=900):
    """图像转 base64，直接内嵌到气泡里显示。"""
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def bubble(role, inner_html):
    cls = "user" if role == "user" else "ai"
    label = "我" if role == "user" else "AI"
    st.markdown(
        f'<div class="wb-row {cls}">'
        f'  <div class="wb-avatar {cls}">{label}</div>'
        f'  <div class="wb-bubble {cls}">{inner_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def conf_bars(boxes, limit=6):
    """置信度条形图，把握大小一眼可见。>=0.8 绿、>=0.5 蓝、<0.5 淡蓝。"""
    rows = []
    for i, b in enumerate(sorted(boxes, key=lambda t: -t[4])[:limit], 1):
        c = b[4]
        color = "#16a34a" if c >= 0.8 else ("#1e5a99" if c >= 0.5 else "#7aa8d0")
        rows.append(
            f'<div class="wb-conf-row">'
            f'  <span class="wb-conf-name">目标 {i}</span>'
            f'  <div class="wb-conf-track">'
            f'    <div class="wb-conf-fill" style="width:{min(c * 100, 100):.0f}%;background:{color}"></div>'
            f'  </div>'
            f'  <span class="wb-conf-val">{c:.2f}</span>'
            f'</div>'
        )
    if len(boxes) > limit:
        rows.append(f'<div class="wb-meta">另有 {len(boxes) - limit} 个目标未展开</div>')
    return '<div class="wb-conf">' + "".join(rows) + "</div>"


def process_large_image(img, model, tile=512, overlap=64,
                        conf=0.25, iou=0.45, on_stage=None, on_tile=None):
    """大图切片推理 + NMS 去重。"""
    h, w = img.shape[:2]
    step = tile - overlap
    xs = tile_starts(w, tile, step)
    ys = tile_starts(h, tile, step)
    total = len(xs) * len(ys)

    raw, idx = [], 0
    if on_stage:
        on_stage("正在准备切片")

    for y in ys:
        for x in xs:
            idx += 1
            crop = img[y:y + tile, x:x + tile]
            if crop.size == 0:
                continue
            if on_tile:
                on_tile(idx, total)

            # imgsz=320 必须和训练一致
            res = model.predict(crop, conf=conf, imgsz=320, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                raw.append((float(x1) + x, float(y1) + y,
                            float(x2) + x, float(y2) + y,
                            float(b.conf), int(b.cls)))

    if on_stage:
        on_stage(f"正在合并重叠框（候选 {len(raw)} 个）")

    final = []
    if raw:
        xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _, _ in raw]
        scores = [c for _, _, _, _, c, _ in raw]
        keep = cv2.dnn.NMSBoxes(xywh, scores, conf, iou)
        if len(keep) > 0:
            for i in np.array(keep).flatten().astype(int):
                final.append(raw[i])
    return final


def draw_boxes(img, boxes, names):
    """按置信度分档上色：高绿、中蓝、低淡蓝。"""
    out = img.copy()
    for x1, y1, x2, y2, c, cls in boxes:
        if c > 0.8:
            color = (0, 140, 60)
        elif c > 0.5:
            color = (153, 90, 30)
        else:
            color = (208, 168, 122)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, f"{names.get(cls, cls)} {c:.2f}",
                    (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ------------------------------------------------------------
# 闲聊：填了 API Key 走真模型，没填走内置离线问答
# ------------------------------------------------------------
CHAT_SYS_PROMPT = (
    "你是「卫星遥感鲸鱼检测系统」里的分析助理，像个真做过遥感项目的人，不像客服。\n"
    "系统背景：用 YOLOv8 对遥感影像切片找鲸鱼；"
    "模型在自建数据集上 mAP50 约 0.745；原图只有 128–149 像素；"
    "推理 imgsz=320，大图先切 512×512 的片再合并。\n"
    "说话规则：\n"
    "1. 第一句就给结论，不要「好的」「收到」「很高兴为你服务」这类开场。\n"
    "2. 不要复述用户的话，不要堆 markdown 标题和项目符号。\n"
    "3. 一般 3 句以内，最多 5 句。\n"
    "4. 不确定就直说，不要编数字。\n"
    "5. 涉及检测结果时，说数量、置信度和下一步建议。\n"
    "6. 记得上文聊过什么，别把每句话都当第一次见。"
)

ANALYST_SYS_PROMPT = (
    "你是遥感影像判读员，负责把 YOLOv8 的检测数字写成一句简报。\n"
    "要求：第一句结论（检出了几个、把握如何），第二句建议（要调阈值还是看原图区域）。\n"
    "不超过 60 字，不用标题，不用项目符号，不夸大，不编造没给的信息。"
)


def history_messages(max_chars=200, max_turns=6):
    """把最近几轮对话转成 OpenAI 的 messages，让模型记得上文。"""
    hist = st.session_state.get("history", [])[-(max_turns * 2):]
    msgs = []
    for h in hist:
        role = h.get("role")
        if role == "user":
            if h.get("kind") == "image":
                msgs.append({"role": "user",
                             "content": f"[上传了一张遥感图：{h.get('name', '')}]"})
            else:
                txt = (h.get("text") or "").strip()
                if txt:
                    msgs.append({"role": "user", "content": txt[:max_chars]})
        elif role == "ai":
            plain = html.unescape(re.sub(r"<[^>]+>", " ", h.get("html", "") or ""))
            plain = re.sub(r"\s+", " ", plain).strip()
            if plain:
                msgs.append({"role": "assistant", "content": plain[:max_chars]})
    return msgs


def call_llm(text, context="", system=None, max_tokens=600, use_history=True):
    """调 OpenAI 兼容接口。带上下文记忆，有搜索结果就一起喂。失败返回 None。"""
    api_key = str(st.session_state.get("api_key", "") or "").strip()
    if not api_key:
        return None
    base = str(st.session_state.get("api_url", "") or "").strip().rstrip("/")
    model_name = str(st.session_state.get("api_model", "") or "").strip()
    if not base or not model_name:
        return None
    user_msg = text
    if context:
        user_msg = (f"下面是刚刚联网搜到的资料，请结合它回答，"
                    f"并在句末标注来源编号：\n\n{context}\n\n问题：{text}")

    messages = [{"role": "system", "content": system or CHAT_SYS_PROMPT}]
    if use_history:
        hist = history_messages()
        # 当前这句已经在 history 里了，去掉避免重复
        while hist and hist[-1]["role"] == "user" and hist[-1]["content"] == text:
            hist.pop()
        messages += hist
    messages.append({"role": "user", "content": user_msg})

    try:
        import requests
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model_name, "messages": messages,
                  "temperature": 0.35, "max_tokens": max_tokens},
            timeout=40,
        )
        if r.status_code != 200:
            return None
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def local_chat(text):
    """不联网也能答的问答。项目类问题给实数，闲聊类给点人味。"""
    t = text.lower()

    def has(*ks):
        return any(k in text or k in t for k in ks)

    # ---------------- 时间 ----------------
    if has("几点", "什么时间", "现在时间", "今天几号", "几号", "星期几", "日期", "今天是"):
        now = datetime.now()
        week = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return (f'<div>现在是 <b>{now.strftime("%Y年%m月%d日 %H:%M")}</b>，'
                f'{week[now.weekday()]}。</div>'
                f'<div class="wb-meta">取的是运行这台电脑的时钟。</div>')

    # ---------------- 情绪 ----------------
    if has("心情不好", "难过", "不开心", "烦", "压力", "焦虑", "崩溃", "累", "疲惫", "困", "撑不住"):
        return random.choice([
            '<div>累就先停一下，图不会跑，鲸鱼也不会跑。</div>'
            '<div class="wb-meta">泡杯茶，回来再传一张图，我还在。</div>',

            '<div>这种时候别硬扛，先睡一觉。很多事情睡醒了就没那么重了。</div>'
            '<div class="wb-meta">真的，物理上睡眠能降焦虑，不是安慰。</div>',

            '<div>烦的时候可以试试把要做的事写下来，只写三件，其余的明天再说。</div>'
            '<div class="wb-meta">事情一多，人就会觉得全都得现在做完，其实不用。</div>',
        ])

    # ---------------- 鼓励 ----------------
    if has("加油", "鼓励", "坚持", "不想干", "做不下去", "没动力", "放弃"):
        return random.choice([
            '<div>你已经把一个能跑起来的检测 Demo 做出来了，'
            '这本身就比大多数人停在"想做"要远得多。</div>'
            '<div class="wb-meta">不用一次做完，今天推进一点点也算。</div>',

            '<div>mAP 从 0.683 提到 0.745 那一次，你是靠想明白"数据只有 129 像素"才调对的。'
            '这种判断力不会白费。</div>'
            '<div class="wb-meta">慢是因为在动脑子，不是因为不行。</div>',
        ])

    # ---------------- 夸奖 ----------------
    if has("夸夸我", "夸我", "我厉害", "我棒不棒", "表扬"):
        return ('<div>你一个人把数据集重组、三次训练、界面改造全推完了，'
                '还没显卡，全靠 CPU 硬扛。</div>'
                '<div class="wb-meta">这不是"会用工具"，这是能自己把一件事从头做到尾。</div>')

    # ---------------- 笑话 ----------------
    if has("笑话", "段子", "搞笑", "幽默", "逗我"):
        return random.choice([
            '<div>为什么鲸鱼从不参加考试？</div>'
            '<div class="wb-meta">因为它们一张嘴就是"我要换气"，没法安静答题。</div>',

            '<div>遥感图里最委屈的是谁？</div>'
            '<div class="wb-meta">是那条被切成四块、还要被 NMS 合并回去的鲸鱼。</div>',

            '<div>我的模型对鲸鱼很有信心，对海面很没信心。</div>'
            '<div class="wb-meta">毕竟 233 张负样本教会它：长得像鲸鱼的东西，八成不是。</div>',

            '<div>训练到第 102 轮的时候，模型开始摸鱼。</div>'
            '<div class="wb-meta">因为学习率调太小了，它学不动了——这是真的发生过的事。</div>',
        ])

    # ---------------- 陪伴 ----------------
    if has("陪我", "聊聊天", "聊会儿", "无聊", "在干嘛", "在干什么"):
        return ('<div>在等图呢。你要是不想传图，我们就随便聊。</div>'
                '<div class="wb-meta">我能聊的有限——我不是真的大模型，'
                '是个本地的规则库，聊深了容易露馅。'
                '不过问时间、讲冷笑话、给点鼓励还是可以的。</div>')

    # ---------------- 你是谁做的 ----------------
    if has("谁做的", "谁做", "谁开发", "谁创造", "创造你", "谁写的", "谁写",
           "你的作者", "作者是谁", "谁造", "谁弄的", "谁搭的"):
        return ('<div>黄俊杰做的。广西民族大学人工智能学院的大创项目，'
                '用 YOLOv8 做卫星遥感图里的鲸鱼检测。</div>'
                '<div class="wb-meta">界面和问答是后来一点点加上去的，'
                '一开始只有上传图片和画框两个功能。</div>')

    # ---------------- 你是 AI 吗 ----------------
    if has("你是ai", "你是机器人", "你是真人", "你是人吗", "你是模型"):
        return ('<div>不填 API Key 的时候，我其实是一堆关键词规则，不是大模型。</div>'
                '<div class="wb-meta">填了 Key 才会接上真正的语言模型。'
                '所以我答不上来时，不是你问得怪，是我这里没写这条。</div>')

    # ---------------- 天气 ----------------
    if has("天气", "下雨", "气温", "冷不冷", "热不热"):
        res = []
        if bool(st.session_state.get("web_on", True)):
            res = web_search("南宁 今天 天气")
        if res:
            return render_search("南宁今天的天气", res[:3])
        return ('<div>天气这块真取不到。</div>'
                '<div class="wb-meta">不是我不想答，是网没通或者搜索被拦了。'
                '侧边栏把「联网搜索」打开再试试。</div>')

    # ---------------- 冷热、吐槽这类短句，当闲聊接 ----------------
    if has("好冷", "冷死", "冻死", "好热", "热死", "好闷", "好潮", "下雨了"):
        return random.choice([
            '<div>南宁这天气翻脸比翻书快，外套随手备一件。</div>'
            '<div class="wb-meta">不想出门就窝着，传两张图给我干活。</div>',
            '<div>冷一点正好，脑子转得快。</div>'
            '<div class="wb-meta">要是懒得动，就让我跑识别，你只管看图。</div>',
        ])

    # ---------------- 再见 ----------------
    if has("晚安", "再见", "拜拜", "走了", "下线"):
        return ('<div>好，去休息吧。</div>'
                '<div class="wb-meta">对话记录我存着，下次打开还在。</div>')

    # ---------------- 项目类 ----------------
    if has("你好", "hello", "hi", "在吗", "嗨"):
        return ('<div>在呢。要检测就把遥感图丢进来，想聊天直接问。</div>'
                '<div class="wb-meta">我能回答这个 Demo 怎么用、参数怎么调、'
                '模型效果怎么样这类问题。</div>')

    if has("你是谁", "你能做什么", "你会什么", "能干嘛", "功能"):
        return ('<div>我负责两件事：</div>'
                '<div class="wb-meta">'
                '① 识别 —— 传一张遥感图，我用 YOLOv8 切片扫一遍，把鲸鱼圈出来；<br>'
                '② 答疑 —— 参数怎么调、结果怎么看，问我就行。'
                '</div>')

    if has("怎么用", "怎么弄", "使用方法", "教程", "步骤"):
        return ('<div>两步：</div>'
                '<div class="wb-meta">'
                '① 侧边栏「选图即识别」选一张图，或者点底部输入框左边的 + 传图；<br>'
                '② 等几秒，结果自动出来，还带时间戳和存档。'
                '</div>')

    if has("准确", "精度", "效果", "准不准", "map", "mAP", "识别率"):
        return ('<div>当前这个模型在自建数据集上 mAP50 约 <b>0.745</b>，'
                'mAP50-95 约 0.31。</div>'
                '<div class="wb-meta">训练数据只有 508 张、原始尺寸 128–149 像素，'
                '所以这个成绩已经不错了。实际用的时候，'
                '拍摄高度、海况、鲸鱼是否露出水面都会影响检出。</div>')

    if has("阈值", "conf", "置信度", "参数", "iou", "重叠"):
        return ('<div>三个参数的作用：</div>'
                '<div class="wb-meta">'
                '置信度阈值 —— 越低检出越多但误报也多，常用 0.15~0.3；<br>'
                'NMS IoU —— 控制重叠框合并，0.45 比较通用；<br>'
                '切片重叠 —— 越大边缘目标越不容易被切断，64 起步，慢一点但更稳。'
                '</div>')

    if has("切片", "为什么切", "切图"):
        return ('<div>因为模型训练时看到的都是 128–149 像素的小图，'
                '直接喂一张几千像素的大图，鲸鱼相对就"变小"了，容易漏。</div>'
                '<div class="wb-meta">切成 512×512 的小片逐块扫，'
                '每片里的目标尺寸就和训练时接近，最后再把重叠的框合并掉。</div>')

    # ---------------- 项目背景 / 训练 / 数据 ----------------
    if has("大创", "项目背景", "为什么做", "意义", "干什么用"):
        return ('<div>这是广西民族大学人工智能学院的大创项目，'
                '做的是用卫星遥感影像自动找鲸鱼。</div>'
                '<div class="wb-meta">传统鲸鱼调查靠船和飞机，人力贵、覆盖小；'
                '卫星图便宜但一张要人工看好久。'
                '这个原型就是让模型替人把图先扫一遍，人只看可疑的地方。</div>')

    if has("训练过程", "怎么训", "怎么训练", "epoch", "轮次", "超参", "调参"):
        return ('<div>一共跑了三次：</div>'
                '<div class="wb-meta">'
                '<b>第一次</b>　yolov8s，imgsz=320，100 轮，mAP50 到 0.683；<br>'
                '<b>第二次</b>　想提高分辨率到 416、学习率调成 0.002，'
                '结果反而更差（0.660），白跑 5 个多小时，作废；<br>'
                '<b>第三次</b>　针对小图特性重配：imgsz 固定 320，'
                '学习率回到 0.01 配 SGD + 余弦退火，'
                '加了 180 度旋转和上下翻转（卫星图本来就没固定朝向，这是白捡的增强），'
                '跑满 200 轮，最佳 0.7453。'
                '</div>')

    if has("数据集", "数据哪", "数据来源", "从哪来", "来源", "多少张", "标注", "负样本"):
        return ('<div>用的是 WFShips 公开数据集，按 YOLO 格式重组了一遍。</div>'
                '<div class="wb-meta">'
                '训练集 508 张、验证集 126 张，其中 233 张是负样本（海面但没有鲸鱼）。<br>'
                '原图很小：128–149 像素，508 张里有 101 张是 129×129。'
                '这也是为什么 imgsz 定 320 就够了，再往上纯粹是插值放大。'
                '</div>')

    if has("技术栈", "用什么做", "什么框架", "架构"):
        return ('<div>检测用 YOLOv8s（Ultralytics），预处理和切片用 OpenCV，'
                '界面是 Streamlit，推理跑在 CPU 上。</div>'
                '<div class="wb-meta">没有 GPU，所以选小而快的 yolov8s，'
                '而不是更大的模型。</div>')

    if has("gpu", "显卡", "训练多久", "跑多久", "多久"):
        return ('<div>全程纯 CPU，没有显卡。</div>'
                '<div class="wb-meta">'
                '训练：workers 拉到 4 之后一轮大约 60 秒，200 轮两个多小时；<br>'
                '推理：640×640 的图切 4 片，3 秒左右出结果。'
                '</div>')

    # 只在"问它是什么/求介绍"时才答科普，别把"布氏鲸研究进展"这类也吃掉
    if (has("布氏鲸", "鲸鱼", "什么鲸") and
            has("是什么", "是啥", "介绍一下", "介绍", "科普", "什么动物", "什么样")):
        return ('<div>布氏鲸（Bryde\'s whale）是须鲸的一种，'
                '体长一般 12–16 米，喜欢热带和亚热带海域。</div>'
                '<div class="wb-meta">它换气时背部会露出水面，'
                '在遥感影像上就是一个亮点加一道水花，这也是模型要学的主要特征。</div>')

    if has("刚才", "上一次", "结果", "记录"):
        recs = load_records(1)
        if recs:
            r = recs[0]
            return (f'<div>最近一次是 {r["time"]}，{html.escape(r["name"])}，'
                    f'找到 <b>{r["count"]}</b> 个目标，用时 {r["elapsed"]} 秒。</div>'
                    f'<div class="wb-meta">可以在上面的「识别历史」里翻到图和详情。</div>')
        return ('<div>还没有识别记录。</div>'
                '<div class="wb-meta">传一张图就有了。</div>')

    if has("谢谢", "thank"):
        return '<div>不客气。</div>'

    if has("没用", "不准", "垃圾", "不行"):
        return ('<div>如果效果不理想，先试这两招：</div>'
                '<div class="wb-meta">'
                '① 把置信度调到 0.15，看是不是阈值卡太严；<br>'
                '② 确认图片是原始遥感影像，不要是已经压缩过的截图。'
                '</div>')

    return None    # 没命中，交给联网搜索或兜底


# ------------------------------------------------------------
# 没接上话时的兜底：会接情绪、会接梗，且不和上一条重复
# ------------------------------------------------------------
def casual_reply(text):
    """规则库聊天的最后一层。尽量像人，但坦率说是写好的话。"""
    t = re.sub(r"^\s*", "", text).strip()
    short = t.replace("。", "").replace("！", "").replace("!", "")

    # 笑 / 玩梗
    if any(k in t for k in ["哈哈", "笑", "乐", "绷不住", "草", "笑死"]):
        return random.choice([
            '<div>笑成这样，看来是真有事。</div>',
            '<div>行，能笑出来就还没被生活按住。</div>',
            '<div>笑点低是好事，说明还有救。</div>',
        ])
    # 亏了 / 倒霉 / 失误
    if any(k in t for k in ["亏", "血亏", "翻车", "失误", "完了", "寄了", "麻了"]):
        return random.choice([
            '<div>亏就亏了，学费交过就不算白交。</div>'
            '<div class="wb-meta">下次提前问一句，能少踩一半坑。</div>',
            '<div>这事要是能复盘，就还有下一次。</div>',
        ])
    # 无聊 / 提不起劲
    if any(k in t for k in ["没意思", "无聊", "没劲", "空虚", "好闷", "不想动", "没动力"]):
        return random.choice([
            '<div>没意思的时候最容易把事往后拖。</div>'
            '<div class="wb-meta">要不先干件小的：传一张图，五分钟就有结果，'
            '比空想强。</div>',
            '<div>那就歇一会儿，别硬找事做。</div>'
            '<div class="wb-meta">歇不住的时候再来找我。</div>',
        ])
    # 自我贬低
    if any(k in t for k in ["我是傻子", "我傻", "我菜", "我不行", "我真笨", "笨蛋"]):
        return ('<div>别这么讲自己。</div>'
                '<div class="wb-meta">能一个人把训练、界面、存档全推完的人，'
                '哪来的笨蛋。</div>')
    # 兜底：拿你说过的词反问，别用「嗯」「然后呢」这种敷衍
    show = html.escape(short[:12])
    pool = [
        f'<div>「{show}」——想聊深点吗？</div>',
        f'<div>说到{show}，你是刚接触，还是已经熟了？</div>',
        f'<div>「{show}」我记下了。还有呢？</div>',
        f'<div>你在意的这个{show}，是碰上什么事了吗？</div>',
        f'<div>{show}——这事你怎么看？</div>',
    ]
    last = st.session_state.get("last_casual", "")
    cands = [p for p in pool if p != last] or pool
    pick = random.choice(cands)
    st.session_state["last_casual"] = pick
    return pick


# ------------------------------------------------------------
# 判断是不是"真在提问"——不是提问就别往搜索引擎里塞
# ------------------------------------------------------------
def looks_like_question(text):
    """只有确实是疑问句才值得联网。闲聊、感叹、吐槽一律不搜。"""
    if any(k in text for k in ["？", "?", "吗", "呢", "怎么", "为什么", "为啥",
                               "是什么", "是多少", "多少", "哪", "谁",
                               "什么时候", "能不能", "可以", "如何", "介绍",
                               "最新", "新闻", "进展"]):
        return True
    return False


DECIDE_PROMPT = (
    "你是一个路由判断器。判断用户这句话，是不是必须联网搜一下才能答准。\n"
    "需要联网的情况：最新消息、时事新闻、具体机构或人物的近期情况、"
    "某个领域的研究进展、政策法规、价格数据、别人写的文章。\n"
    "不需要联网的情况：闲聊、情绪倾诉、常识、算术、翻译、写作、"
    "关于本项目自身的问题（这个项目有内置知识库）。\n"
    "只输出一行，不要解释：\n"
    "需要 → SEARCH: <最合适的中文搜索词>\n"
    "不需要 → NO"
)


def llm_decide_search(text):
    """让大模型自己决定要不要联网。返回搜索词，不需要就返回 None。"""
    raw = call_llm(text, system=DECIDE_PROMPT, max_tokens=60, use_history=False)
    if not raw:
        return None
    raw = raw.strip()
    if raw.upper().startswith("NO"):
        return None
    m = re.search(r"SEARCH\s*[:：]\s*(.+)", raw, re.I)
    return m.group(1).strip().strip("\"'") if m else None


# ------------------------------------------------------------
# 联网搜索：360 搜索免费直连，不需要 API Key
# ------------------------------------------------------------
SEARCH_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _clean_text(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _so_search(q, n):
    """360 搜索。结构稳、摘要清楚，优先用。"""
    import requests
    try:
        r = requests.get("https://www.so.com/s", params={"q": q},
                         headers=SEARCH_UA, timeout=10)
        if r.status_code != 200:
            return []
        txt = r.text
    except Exception:
        return []

    out = []
    for m in re.finditer(r'<h3[^>]*class="[^"]*res-title[^"]*"[^>]*>(.*?)</h3>', txt, re.S):
        head = m.group(1)
        href = re.search(r'href="([^"]+)"', head)
        if not href:
            continue
        link = href.group(1)
        if link.startswith("/"):
            link = "https://www.so.com" + link
        title = _clean_text(head)
        if not title:
            continue
        tail = txt[m.end(): m.end() + 1600]
        md = re.search(r'<p[^>]*class="[^"]*res-desc[^"]*"[^>]*>(.*?)</p>', tail, re.S)
        body = _clean_text(md.group(1)) if md else ""
        body = re.sub(r"^[\d]{4}年[\d]{1,2}月[\d]{1,2}日\s*[-—]?\s*", "", body).strip()
        out.append({"title": title, "url": link, "body": body[:160]})
        if len(out) >= n:
            break
    return out


def _baidu_search(q, n):
    """百度兜底。要带 cookie，否则连发两次会被风控成空页。"""
    import requests
    try:
        s = requests.Session()
        s.get("https://www.baidu.com/", headers=SEARCH_UA, timeout=8)
        r = s.get("https://www.baidu.com/s", params={"wd": q, "rn": 10},
                  headers=SEARCH_UA, timeout=10)
        if r.status_code != 200:
            return []
        txt = r.text
    except Exception:
        return []

    out = []
    for m in re.finditer(
        r'<h3[^>]*class="[^"]*t[^"]*"[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        txt, re.S,
    ):
        link = html.unescape(m.group(1))
        title = _clean_text(m.group(2))
        # 跳过"百度图片""百度学术"这类聚合卡片
        if not title or "百度图片" in title:
            continue
        out.append({"title": title, "url": link, "body": ""})
        if len(out) >= n:
            break
    return out


@st.cache_data(ttl=600, show_spinner=False)
def web_search(q, n=5):
    """两路搜索源轮着试，再拿清洗过的短词重试一次。全失败才返回空。"""
    # 口语化的尾巴去掉，搜索引擎更容易命中
    short = re.sub(r"(是什么事|怎么回事|是怎么样的|是啥|怎么办|呢|啊|吗|？|\?|。)", "", q).strip()

    queries = [q]
    if short and short != q and len(short) >= 2:
        queries.append(short)

    for query in queries:
        for fn in (_so_search, _baidu_search):
            res = fn(query, n)
            if res:
                return res
    return []


def render_search(q, results):
    """把搜索结果写成一屏可点开的卡片。"""
    rows = []
    for i, x in enumerate(results, 1):
        rows.append(
            f'<div style="margin-top:10px;">'
            f'  <div><b>{i}. {html.escape(x["title"][:80])}</b></div>'
            + (f'  <div class="wb-meta" style="margin-top:2px;">'
               f'{html.escape(x["body"])}</div>' if x["body"] else "")
            + f'  <div class="wb-meta" style="margin-top:2px;">'
              f'<a href="{html.escape(x["url"])}" target="_blank" '
              f'style="color:#1e5a99;">查看原文</a></div>'
              f'</div>'
        )
    return (f'<div>搜到 {len(results)} 条关于「{html.escape(q)}」的结果：</div>'
            + "".join(rows)
            + '<div class="wb-meta">来源：360 搜索　结果由搜索引擎返回，未做人工核实</div>')


def take_file(name, data, note="识别这一张"):
    """把一个图像字节流变成一轮对话：用户气泡 + 立即检测。"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.session_state.history.append({
            "role": "ai",
            "html": '<div>这张图打不开。</div>'
                    '<div class="wb-meta">可能文件损坏或格式不支持，'
                    '建议换成 JPG 或 PNG 再试一次。</div>',
            "result_b64": None,
        })
        return False

    h, w = img.shape[:2]
    p = {"bytes": data, "name": name, "w": w, "h": h, "time": now_str()}
    st.session_state.last_img = p
    st.session_state.history.append({
        "role": "user", "kind": "image",
        "name": name, "b64": img_to_b64(img),
        "w": w, "h": h, "time": p["time"],
    })
    return True


# ============================================================
# 六、检测主流程（上传后自动调用，也可以再输入指令重跑）
# ============================================================
def run_detect(p, model, tile, overlap, conf_thres, iou_thres):
    """跑一次检测 → 存一条记录 → 往对话里追加一条结果气泡。"""
    img = cv2.imdecode(np.frombuffer(p["bytes"], np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    bar = st.progress(0, text="准备中…")

    def on_stage(msg):
        bar.progress(5, text=msg)

    def on_tile(i, n):
        bar.progress(5 + int(75 * i / n), text=f"正在扫描第 {i}/{n} 块区域")

    t0 = time.time()
    boxes = process_large_image(img, model, tile, overlap, conf_thres, iou_thres,
                                on_stage=on_stage, on_tile=on_tile)

    bar.progress(85, text="正在画检测框")
    result = draw_boxes(img, boxes, model.names)
    bar.progress(100, text="完成")
    bar.empty()

    elapsed = time.time() - t0
    n_tiles = len(tile_starts(w, tile, tile - overlap)) * \
              len(tile_starts(h, tile, tile - overlap))

    # 存档：不管有没有检出，都留一条，方便日后对照
    rec = save_record(p, boxes, elapsed, n_tiles, result, w, h, conf_thres, iou_thres)

    if len(boxes) > 0:
        high = sum(1 for b in boxes if b[4] > 0.8)
        low = sum(1 for b in boxes if b[4] < 0.4)
        if high > 0:
            verdict = f"其中 {high} 个很有把握"
        elif low >= len(boxes):
            verdict = "但置信度都偏低，建议把阈值调到 0.15 再看一遍"
        else:
            verdict = "把握中等，可以让阈值再往下降一点"

        ok, buf = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # 有 API Key 时，让模型把这堆数字写成一句人话简报；没有就用规则文案
        brief = None
        if str(st.session_state.get("api_key", "") or "").strip():
            confs = "、".join(f"{b[4]:.2f}" for b in boxes[:6])
            brief = call_llm(
                f"在 {w}×{h} 的遥感图上切了 {n_tiles} 块、阈值 {conf_thres:.2f}，"
                f"检出 {len(boxes)} 个目标，置信度 {confs}。"
                f"写一段不超过 60 字的简报。",
                system=ANALYST_SYS_PROMPT,
                max_tokens=120,
                use_history=False,
            )

        if brief:
            head = "".join(
                f"<div>{html.escape(ln)}</div>"
                for ln in brief.strip().split("\n") if ln.strip()
            )
        else:
            head = f'<div>扫描完成，找到 <b>{len(boxes)}</b> 个疑似目标，{verdict}。</div>'

        html_ai = (
            f'{head}'
            f'{conf_bars(boxes)}'
            f'<img src="{img_to_b64(result)}">'
            f'<div class="wb-meta">扫了 {n_tiles} 块区域　阈值 {conf_thres:.2f}　'
            f'IoU {iou_thres:.2f}　用时 {elapsed:.1f} 秒<br>'
            f'{rec["time"]}　已存档，关掉页面也能翻回来</div>'
        )
        st.session_state.history.append({
            "role": "ai", "html": html_ai,
            "id": rec["id"],
            "result_b64": True,
            "result_png": io.BytesIO(buf.tobytes()) if ok else None,
        })
    else:
        html_ai = (
            f'<div>这张图里没找到鲸鱼。</div>'
            f'<div class="wb-meta">扫了 {n_tiles} 块区域　用时 {elapsed:.1f} 秒　'
            f'阈值 {conf_thres:.2f}<br>'
            f'常见原因：目标太小、海面反光太亮，或者阈值设高了。<br>'
            f'<b>建议把置信度调到 0.15 再跑一次。</b><br>'
            f'{rec["time"]}　已存档</div>'
        )
        st.session_state.history.append({
            "role": "ai", "html": html_ai, "result_b64": None})


# ============================================================
# 六点五、侧边栏「选图即识别」：选完文件立刻跑，零点击零指令
# ============================================================
with st.sidebar:
    st.divider()
    st.caption("选图即识别 · 选完自动开始，不用点发送")
    up = st.file_uploader(
        "选图即识别",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        key="quick_up",
        label_visibility="collapsed",
    )
    if up is not None:
        sig = f"{up.name}__{up.size}"
        if sig != st.session_state.get("quick_sig", ""):
            st.session_state.quick_sig = sig
            if take_file(up.name, up.getvalue()):
                run_detect(st.session_state.last_img, model, TILE, overlap,
                           conf_thres, iou_thres)
            st.rerun()


# ============================================================
# 七、会话状态
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_img" not in st.session_state:
    st.session_state.last_img = None      # 上一张图，方便重复识别


# ============================================================
# 八、标题 + 上手提示 + 历史记录面板 + 对话渲染
# ============================================================
st.markdown("#### 🐋 卫星遥感鲸鱼检测")

# ---- 识别历史（关掉页面也翻得回来）----
recs = load_records()
if recs:
    with st.expander(f"📁 识别历史 · 共 {len(recs)} 条（关掉页面也不会丢）"):
        for r in recs[:15]:
            with st.container(border=True):
                st.markdown(
                    f'**{r["time"]}**　{r["name"]}　'
                    f'{r["count"]} 个目标　{r["w"]}×{r["h"]}　'
                    f'{r["elapsed"]} 秒'
                )
                thumb = os.path.join(RECORDS_DIR, r.get("thumb_file", ""))
                if os.path.isfile(thumb):
                    st.image(thumb, use_container_width=True)
                if r["confs"]:
                    st.caption("置信度：" + " / ".join(f"{c:.2f}" for c in r["confs"][:8]))
                c1, c2 = st.columns([0.75, 0.25])
                with c2:
                    if st.button("删除这条", key=f"hd_{r['id']}", type="tertiary"):
                        delete_record(r)
                        st.rerun()
        if len(recs) > 15:
            st.caption(f"只显示最近 15 条，其余都在 records 文件夹里。")
        st.caption(f"存档位置：{RECORDS_DIR}")

# ---- 首屏提示 ----
if not st.session_state.history:
    st.markdown(
        '<div class="wb-tips">'
        '<b>用起来就一步</b><br>'
        '① 点底部输入框左边的 <b>+</b> 上传遥感影像（JPG / PNG / TIF / BMP）<br>'
        '② <b>上传完自动开始识别</b>，不用再敲指令<br>'
        '③ 结果带时间戳并自动存档，关掉页面也能在「识别历史」里翻回来'
        '</div>',
        unsafe_allow_html=True,
    )
    bubble("ai",
           '<div>你好，我是鲸鱼检测助手。</div>'
           '<div class="wb-meta">大图会自动切成小片逐块扫，扫完再把结果拼回去，'
           '所以整张图里的目标都不会漏。直接把图丢进来就行。</div>')

# ---- 渲染对话 ----
for idx, item in enumerate(st.session_state.history):
    col_main, col_del = st.columns([0.93, 0.07], gap="small", vertical_alignment="top")

    with col_main:
        if item["role"] == "user":
            if item["kind"] == "image":
                bubble("user",
                       f'<div>识别这一张</div>'
                       f'<div class="wb-meta">{html.escape(item["name"])}　'
                       f'{item["w"]} × {item["h"]}<br>{item.get("time", "")}</div>'
                       f'<img src="{item["b64"]}">')
            else:
                bubble("user", f'<div>{html.escape(item["text"])}</div>')
        else:
            bubble("ai", item["html"])
            if item.get("result_b64"):
                st.download_button(
                    label="下载结果图",
                    data=item["result_png"],
                    file_name="result.jpg",
                    mime="image/jpeg",
                    key=f"dl_{item['id']}",
                )

    with col_del:
        if st.button("✕", key=f"del_{idx}", type="tertiary", help="删除这一条对话"):
            st.session_state.history.pop(idx)
            st.rerun()


# ============================================================
# 九、底部输入框：上传即识别
# ============================================================
val = st.chat_input(
    "上传遥感影像，上传完自动开始识别",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
)

if val:
    files = getattr(val, "files", None) or []
    text = (getattr(val, "text", "") or "").strip()

    # ---------- ① 收到图片：立即识别，不等文字 ----------
    if files:
        f = files[0]
        try:
            data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("图像解码失败，可能文件损坏或格式不支持")
        except Exception as e:
            st.session_state.history.append({
                "role": "ai",
                "html": f'<div>这张图打不开。</div>'
                        f'<div class="wb-meta">{html.escape(str(e))}<br>'
                        f'建议换成 JPG 或 PNG 再试一次。</div>',
                "result_b64": None,
            })
            st.rerun()

        h, w = img.shape[:2]
        p = {
            "bytes": data.tobytes(),
            "name": getattr(f, "name", "未命名"),
            "w": w, "h": h,
            "time": now_str(),
        }
        st.session_state.last_img = p
        st.session_state.history.append({
            "role": "user", "kind": "image",
            "name": p["name"],
            "b64": img_to_b64(img),
            "w": w, "h": h,
            "time": p["time"],
        })

        # 立刻开始识别
        run_detect(p, model, TILE, overlap, conf_thres, iou_thres)
        st.rerun()

    # ---------- ② 收到文字：说了指令就用上一张图重跑 ----------
    if text:
        st.session_state.history.append({
            "role": "user", "kind": "text", "text": text, "time": now_str()})

        # 只有"帮我搜一下/查一下"这种显式说法才需要剥前缀；
        # 其他情况一律不猜关键词——内置答不上就直接联网。
        wants_search = any(k in text for k in ["搜一下", "搜索", "帮我搜", "查一下", "查查"])

        is_cmd = (not wants_search) and any(
            k in text for k in ["识别", "检测", "开始", "再跑", "重跑", "跑一下", "看一下"])

        if not is_cmd:
            # 把口语化的前缀剥掉，留下真正的查询词
            q = text
            for kw in ["帮我搜一下", "帮我搜", "搜一下", "搜索一下", "搜索",
                       "查一下", "查查", "帮我查", "请", "？", "?", "。"]:
                q = q.replace(kw, "")
            q = q.strip() or text

            web_on = bool(st.session_state.get("web_on", True))
            has_key = str(st.session_state.get("api_key", "") or "").strip()
            results = []
            local = None

            if wants_search:
                # 明说要搜 → 直接搜，不查内置知识
                if web_on:
                    results = web_search(q)[:5]
            elif has_key:
                # 有真模型：让它自己判断要不要联网，不用我们猜
                query = llm_decide_search(text)
                if query and web_on:
                    results = web_search(query)[:5]
            else:
                # 没有真模型：内置知识答得上就不搜；确实在提问才联网
                local = local_chat(text)
                if local is None and web_on and looks_like_question(text):
                    results = web_search(text)[:5]

            # 有 API Key：把搜索结果交给大模型总结
            context = ""
            if results:
                context = "\n".join(
                    f"[{i}] {x['title']}｜{x['body']}｜{x['url']}"
                    for i, x in enumerate(results, 1)
                )
            reply = call_llm(text, context)

            if reply:
                body = "".join(
                    f"<div>{html.escape(line)}</div>"
                    for line in reply.strip().split("\n") if line.strip()
                )
                html_ai = body
                if results:
                    html_ai += ('<div class="wb-meta">已结合联网搜索结果，'
                                '共参考 %d 条资料</div>' % len(results))
            elif local is not None:
                html_ai = local
            elif results:
                html_ai = render_search(q, results)
            elif not looks_like_question(text):
                # 不是提问，也没接上——接一句人话，且不和上一条重复
                html_ai = casual_reply(text)
            elif web_on:
                html_ai = ('<div>网上也没查到能对上的结果。</div>'
                           '<div class="wb-meta">可能是网络不通，或者这个词太冷门。'
                           '换个说法再试试。</div>')
            else:
                html_ai = ('<div>这个问题超出我的内置知识了。</div>'
                           '<div class="wb-meta">把侧边栏的「联网搜索」打开，'
                           '我就能去网上找。</div>')

            st.session_state.history.append({
                "role": "ai", "html": html_ai, "result_b64": None})
            st.rerun()

        if st.session_state.last_img is None:
            st.session_state.history.append({
                "role": "ai",
                "html": '<div>还没有可重复检测的影像。</div>'
                        '<div class="wb-meta">点底部输入框左边的 + 传一张，'
                        '传完我会自动开始识别。</div>',
                "result_b64": None,
            })
            st.rerun()

        run_detect(st.session_state.last_img, model, TILE, overlap, conf_thres, iou_thres)
        st.rerun()
