"""数据加载层的可插拔修复模块（sharegpt 原始 schema 上工作）。

设计目标
--------
针对 er_merged_shuffled 多模态数据集，在 jsonl 读入后、align_dataset 之前
对每条原始 sharegpt 样本做一次「在线修复」，不落盘改源数据，便于快速试参。

详细分布对比（修复前 vs 修复后）见 docs/数据探查报告.html。

每个 Fixer 都是一个无状态可调用对象，签名：

    fixer(example: dict, ctx: FixerContext) -> dict | None

  - 输入：raw sharegpt example（dict，含 dataset_attr.messages 指向的字段 + image 列表 + 其余原 jsonl 字段）。
  - 输出：修复后的 example（保持 schema 不变），或 None 表示丢弃这条样本。
  - 任意一步返回 None，pipeline 立即停并返回 None。

为什么不复用 SharegptDatasetConverter 内部?
- align 后字段已被重命名/重排（prompt/response/image），且 <image> 被替换成 template-specific token，
  这一层做"修语义"已经太晚；
- 早一步在 raw sharegpt schema 上做修复，能让 broken_data 路径、视觉 token 注入路径都受益。

注册新 fixer 的方式
-----------------
在本文件内定义新类（继承 BaseFixer 实现 _apply），用 @register_fixer("name") 装饰即可，
yaml 里通过 `data_fixes: name1,name2,...` 启用，无需改 loader。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

from ..extras.logging import get_logger


if TYPE_CHECKING:
    from ..hparams import DataArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 上下文 & 基类
# ---------------------------------------------------------------------------

@dataclass
class FixerContext:
    """传给 fixer 的运行期上下文，避免每个 fixer 都自己解析 dataset_attr。"""

    messages_key: str           # raw 字段名，例如 "conversations"
    image_key: Optional[str]    # raw 字段名，例如 "image"；为 None 表示该数据集声明无 image 列
    role_tag: str               # "from" or "role"
    content_tag: str            # "value" or "content"
    user_tag: str               # "human" or "user"
    image_placeholder: str = "<image>"
    video_placeholder: str = "<video>"
    max_images: int = 10        # cap_images 用

    @classmethod
    def from_args(cls, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "FixerContext":
        return cls(
            messages_key=dataset_attr.messages,
            image_key=dataset_attr.image,
            role_tag=dataset_attr.role_tag,
            content_tag=dataset_attr.content_tag,
            user_tag=dataset_attr.user_tag,
            max_images=int(getattr(data_args, "max_images_per_sample", 10) or 10),
        )


class BaseFixer:
    """统一接口：__call__ 返回 dict（修复后）或 None（丢弃）。"""

    name: str = "base"

    def __call__(self, example: dict, ctx: FixerContext) -> Optional[dict]:
        return self._apply(example, ctx)

    def _apply(self, example: dict, ctx: FixerContext) -> Optional[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 注册中心
# ---------------------------------------------------------------------------

_FIXER_REGISTRY: Dict[str, type] = {}


def register_fixer(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in _FIXER_REGISTRY:
            raise ValueError(f"fixer {name!r} already registered")
        cls.name = name
        _FIXER_REGISTRY[name] = cls
        return cls
    return deco


def available_fixers() -> List[str]:
    return sorted(_FIXER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _iter_turns(example: dict, ctx: FixerContext):
    """yield (turn_dict, value_str) for each turn whose content is a string.

    跳过非 str 的轮（应该不存在，但留个保险）。"""
    msgs = example.get(ctx.messages_key) or []
    if not isinstance(msgs, list):
        return
    for turn in msgs:
        if not isinstance(turn, dict):
            continue
        v = turn.get(ctx.content_tag)
        if isinstance(v, str):
            yield turn, v


def _get_image_list(example: dict, ctx: FixerContext) -> List[str]:
    if ctx.image_key is None:
        return []
    val = example.get(ctx.image_key)
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [x for x in val if isinstance(x, str)]
    return []


def _set_image_list(example: dict, ctx: FixerContext, imgs: List[str]) -> None:
    if ctx.image_key is not None:
        example[ctx.image_key] = imgs


def _select_keep_indices(n: int, k: int) -> List[int]:
    """从 n 张图里选 k 张：首 + 末 + 中间均匀采样。

    与 scripts/profile_er_merged.py 同函数保持一致。
    """
    if n <= k:
        return list(range(n))
    if k < 2:
        return [0]
    if k == 2:
        return [0, n - 1]
    mid_k = k - 2
    middle = [round(i * (n - 1) / (k - 1)) for i in range(1, mid_k + 1)]
    idx = [0] + middle + [n - 1]
    seen = set()
    out = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# 具体 fixer 实现
# ---------------------------------------------------------------------------


@register_fixer("prepend_dhta_task_token")
class PrependDHTATaskTokenFixer(BaseFixer):
    """Deterministically prefix the first user turn with one DHTA route token.

    The route is taken from record metadata rather than inferred from prompt text.
    The transformation is idempotent and fail-closed: unknown task families,
    conflicting tokens, or records without a user turn are rejected. Assistant
    messages are never modified.
    """

    qa_token = "[2D_QA]"
    task_plan_token = "[TASK_PLAN]"
    family_to_token = {
        "2d_qa": qa_token,
        "2d-qa": qa_token,
        "qa": qa_token,
        "task_plan": task_plan_token,
        "task-plan": task_plan_token,
        "task planning": task_plan_token,
    }

    def _apply(self, example, ctx):
        task_family = example.get("task_family") or example.get("task_route")
        normalized_family = str(task_family or "").strip().lower()
        expected_token = self.family_to_token.get(normalized_family)
        if expected_token is None:
            return None

        msgs = example.get(ctx.messages_key) or []
        for turn in msgs:
            if not isinstance(turn, dict) or turn.get(ctx.role_tag) != ctx.user_tag:
                continue

            value = turn.get(ctx.content_tag)
            if not isinstance(value, str):
                return None

            qa_count = value.count(self.qa_token)
            task_plan_count = value.count(self.task_plan_token)
            if qa_count + task_plan_count:
                expected_count = qa_count if expected_token == self.qa_token else task_plan_count
                if expected_count == 1 and qa_count + task_plan_count == 1:
                    return example
                return None

            turn[ctx.content_tag] = f"{expected_token}\n{value.lstrip()}"
            return example

        return None


@register_fixer("split_static_video_to_image")
class SplitStaticVideoToImageFixer(BaseFixer):
    # Move static image files stored in the video column back to the image column.
    # This is opt-in and intended for task-planning C23 mixed media records.

    video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

    def _get_video_key(self, example, ctx):
        if "videos" in example:
            return "videos"
        if "video" in example:
            return "video"
        return None

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [x for x in value if isinstance(x, str)]
        return []

    def _is_video_path(self, path: str) -> bool:
        ext = os.path.splitext(path.split("?", 1)[0])[1].lower()
        return ext in self.video_exts

    def _replace_video_placeholders(self, example, ctx, count: int) -> None:
        remaining = count
        for turn, value in _iter_turns(example, ctx):
            if remaining <= 0:
                break
            pieces = value.split(ctx.video_placeholder)
            if len(pieces) == 1:
                continue
            replace_n = min(remaining, len(pieces) - 1)
            rebuilt = pieces[0]
            for idx in range(1, len(pieces)):
                rebuilt += (ctx.image_placeholder if idx <= replace_n else ctx.video_placeholder)
                rebuilt += pieces[idx]
            turn[ctx.content_tag] = rebuilt
            remaining -= replace_n

    def _apply(self, example, ctx):
        video_key = self._get_video_key(example, ctx)
        if video_key is None:
            return example

        videos = self._as_list(example.get(video_key))
        if not videos:
            return example

        kept_videos, moved_images = [], []
        for item in videos:
            if self._is_video_path(item):
                kept_videos.append(item)
            else:
                moved_images.append(item)

        if not moved_images:
            return example

        images = _get_image_list(example, ctx)
        _set_image_list(example, ctx, images + moved_images)
        example[video_key] = kept_videos
        self._replace_video_placeholders(example, ctx, len(moved_images))
        return example

@register_fixer("remove_video_tag")
class RemoveVideoTagFixer(BaseFixer):
    """去掉所有 <video> 占位符。

    背景：er_vsi500k 用 1 个 <video> 占位表 32 张抽帧图，与多图 ShareGPT 约定不一致，
    且训练侧 video plugin 路径未配置；统一去掉，让后续 align_image_tags 按 image
    列表长度重新生成 <image> 占位即可。
    """

    def _apply(self, example, ctx):
        for turn, v in _iter_turns(example, ctx):
            if ctx.video_placeholder in v:
                turn[ctx.content_tag] = v.replace(ctx.video_placeholder, "")
        return example


@register_fixer("require_image_and_text")
class RequireImageAndTextFixer(BaseFixer):
    """必须同时含图片与有效文本，否则丢弃。

    - 无图 / image 列表为空 → 丢弃（约 0.54%，多为 er_llava 等纯文本 QA）；
    - 去掉所有 <image>/<video> 占位后剩余文本为空 → 丢弃（防止只有占位无内容的样本）。
    """

    def _apply(self, example, ctx):
        if len(_get_image_list(example, ctx)) == 0:
            return None
        non_placeholder = []
        for _, v in _iter_turns(example, ctx):
            stripped = v.replace(ctx.image_placeholder, "").replace(ctx.video_placeholder, "")
            non_placeholder.append(stripped)
        if not "".join(non_placeholder).strip():
            return None
        return example


@register_fixer("cap_images")
class CapImagesFixer(BaseFixer):
    """单样本图片数上限 ctx.max_images，超出按 首 + 末 + 中间均匀采样 抽取。

    只裁 image 列表；<image> 占位符的数量留给后置 align_image_tags 统一重写，
    避免这里既改图又改文本 / 漏对齐。
    """

    def _apply(self, example, ctx):
        imgs = _get_image_list(example, ctx)
        if len(imgs) <= ctx.max_images:
            return example
        keep = _select_keep_indices(len(imgs), ctx.max_images)
        _set_image_list(example, ctx, [imgs[i] for i in keep])
        return example


@register_fixer("align_image_tags")
class AlignImageTagsFixer(BaseFixer):
    """把 <image> 占位符数量与 image 列表长度严格对齐：

      1. 删除所有现存 <image> 占位符；
      2. 把 n = len(image) 个 <image>（每个独占一行）拼到第一条 user 轮的开头。

    动机：原数据 ~3.18% tag/image 错配（详见数据探查报告），多模态 token 注入
    路径默认按 tag 对齐图片，错配样本会出现「部分图未参与训练」/「空占位空对齐」。
    本 fixer 是 cap_images 的天然下游：cap 完图后必须重写 tag。
    """

    def _apply(self, example, ctx):
        imgs = _get_image_list(example, ctx)

        # 1) 删掉所有现存 <image>
        for turn, v in _iter_turns(example, ctx):
            if ctx.image_placeholder in v:
                turn[ctx.content_tag] = v.replace(ctx.image_placeholder, "")

        n = len(imgs)
        if n == 0:
            return example  # 不在这里丢弃，留给 require_image_and_text 决定

        placeholder_block = (ctx.image_placeholder + "\n") * n
        msgs = example.get(ctx.messages_key) or []
        injected = False
        for turn in msgs:
            if not isinstance(turn, dict):
                continue
            if turn.get(ctx.role_tag) == ctx.user_tag:
                v = turn.get(ctx.content_tag) or ""
                if not isinstance(v, str):
                    v = ""
                turn[ctx.content_tag] = placeholder_block + v.lstrip()
                injected = True
                break

        if not injected and msgs and isinstance(msgs[0], dict):
            v = msgs[0].get(ctx.content_tag) or ""
            if not isinstance(v, str):
                v = ""
            msgs[0][ctx.content_tag] = placeholder_block + v.lstrip()

        return example


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class FixerPipeline:
    """串行执行多个 fixer，任一 fixer 返回 None 立即丢弃整条样本。"""

    fixers: List[BaseFixer] = field(default_factory=list)
    ctx: Optional[FixerContext] = None

    def apply(self, example: dict) -> Optional[dict]:
        cur = example
        for fx in self.fixers:
            cur = fx(cur, self.ctx)
            if cur is None:
                return None
        return cur


# 默认 fixer 顺序：去 video → 必有图+文 → cap → 重对齐 tag。
# 这个顺序保证：
#   - 在丢弃判断（require_image_and_text）之前，<video> 已经被去掉，避免把"只有 video 占位
#     的样本"误判为有效；
#   - cap 在 align 之前，align 拿到的是已经截到 max_images 之后的图，重写 <image> 数量正确；
#   - align 放最后，所有上游修改都会被 tag 数量重写覆盖。
DEFAULT_FIXER_SEQUENCE: Sequence[str] = (
    "remove_video_tag",
    "require_image_and_text",
    "cap_images",
    "align_image_tags",
)


def build_pipeline(
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Optional[FixerPipeline]:
    """根据 data_args 构造 pipeline；返回 None 表示未启用修复。"""
    if not getattr(data_args, "apply_data_fixes", False):
        return None

    names_raw = getattr(data_args, "data_fixes", None)
    if names_raw is None or (isinstance(names_raw, str) and not names_raw.strip()):
        names: Sequence[str] = DEFAULT_FIXER_SEQUENCE
    elif isinstance(names_raw, str):
        names = [x.strip() for x in names_raw.split(",") if x.strip()]
    else:
        names = list(names_raw)

    unknown = [n for n in names if n not in _FIXER_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown data fixer(s): {unknown}; available = {available_fixers()}"
        )

    fixers = [_FIXER_REGISTRY[n]() for n in names]
    ctx = FixerContext.from_args(dataset_attr, data_args)
    logger.info(
        "[fixers] enabled for dataset=%s: %s (max_images=%d)",
        getattr(dataset_attr, "dataset_name", "?"),
        " -> ".join(f.name for f in fixers),
        ctx.max_images,
    )
    return FixerPipeline(fixers=fixers, ctx=ctx)


# ---------------------------------------------------------------------------
# HF datasets.map 适配器
# ---------------------------------------------------------------------------

def make_map_fn(pipeline: FixerPipeline) -> Callable[[dict], dict]:
    """返回一个 example -> example 的可调用，供 dataset.map(batched=False) 使用。

    HF map 不能直接「丢弃」样本，所以丢弃的样本会被还原为一条「空样本」：
      - 把 ctx.messages_key 置为 []
      - 把 ctx.image_key 置为 []
    这样下游 SharegptDatasetConverter 会走 broken_data 分支输出 prompt=[]/response=[]，
    再由 loader 层的 .filter(len(prompt)>0) 完成实际丢弃，保持 schema 一致性。
    """
    ctx = pipeline.ctx

    def _fn(example: dict) -> dict:
        fixed = pipeline.apply(example)
        if fixed is None:
            example[ctx.messages_key] = []
            if ctx.image_key is not None:
                example[ctx.image_key] = []
            return example
        return fixed

    return _fn
