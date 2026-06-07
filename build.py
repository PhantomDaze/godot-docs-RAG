#!/usr/bin/env python3
"""
Godot 文档向量知识库 — 构建脚本
=================================
解析 _sources/ 下的 RST 源文件 → 分块 → BGE-M3 嵌入 → 存入 ChromaDB

用法:
    python build.py              # 默认行为
    python build.py --workers 4  # 限制 CPU 核心数
    python build.py --dry-run    # 只解析不分块，不嵌入，不存储
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================================
# 配置
# ============================================================================

SOURCE_DIR = Path("_sources")
PERSIST_DIR = Path("index_data")
LOCAL_MODEL_DIRS = [
    Path("bge-m3-model"),     # 同级目录
    Path("../bge-m3"),         # 上级目录
]
MODEL_NAME = "BAAI/bge-m3"                     # HuggingFace 模型名（备选）
# 优先使用本地模型
_MODEL_DIR = None
for _d in LOCAL_MODEL_DIRS:
    if _d.exists():
        _MODEL_DIR = _d
        break
MODEL_PATH = str(_MODEL_DIR) if _MODEL_DIR else MODEL_NAME
BATCH_SIZE_EMBED = 32          # 嵌入批大小
BATCH_SIZE_DB = 500            # ChromaDB 写入批大小
CHUNK_MIN_CHARS = 100          # 块最小字符数
CHUNK_MAX_CHARS = 2000         # 块最大字符数（超过则再切分）
COLLECTION_NAME = "godot_docs"


# ============================================================================
# RST 清洗
# ============================================================================

# 行首为 .. 且后面紧跟纯注释关键词的视为注释行，直接丢弃
COMMENT_DIRECTIVES = {
    "toctree", "highlight", "code-block", "raw", "include",
    "csv-table", "contents", "container", "class", "rubric",
    "epigraph", "highlights", "topic", "sidebar", "parsed-literal",
    "math", "code", "bibliography", "glossary", "productionlist",
    "versionadded", "versionchanged", "deprecated", "describe",
    "option", "object", "function", "data", "exception",
    "attribute", "module", "currentmodule", "automodule",
    "autoclass", "autofunction", "automethod", "autoattribute",
    "only", "hlist", "pull-quote",
}

# 有语义价值的指令，保留内容文本
SEMANTIC_DIRECTIVES = {
    "note":    lambda body: f"\n[注意] {body.strip()}",
    "warning": lambda body: f"\n[警告] {body.strip()}",
    "seealso": lambda body: f"\n[参见] {body.strip()}",
    "tip":     lambda body: f"\n[提示] {body.strip()}",
    "important": lambda body: f"\n[重要] {body.strip()}",
    "error":   lambda body: f"\n[错误] {body.strip()}",
}


def clean_rst(text: str) -> str:
    """多轮清洗 RST 文本，输出干净的可嵌入文本。"""

    lines = text.split("\n")
    cleaned: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- 跳过空行 ----
        if stripped == "":
            cleaned.append("")
            i += 1
            continue

        # ---- 以 .. 开头的指令/注释行 ----
        if stripped.startswith(".. "):
            directive = stripped[3:].strip()

            # .. _label:  链接目标，丢弃
            if re.match(r"^_[\w-]+:\s*$", directive):
                i += 1
                continue

            # .. rst-class:: *  内部标记，丢弃
            if directive.startswith("rst-class::"):
                i += 1
                continue

            # :github_url:, :allow_comments:  元信息，丢弃
            if directive.startswith(":github_url:") or directive.startswith(":allow_comments:"):
                i += 1
                continue

            # .. image:: / .. video:: / .. figure::  丢弃主体及其选项行
            if re.match(r"^(image|video|figure)::", directive):
                i += 1
                while i < len(lines) and re.match(r"^\s+:(\w+):", lines[i]):
                    i += 1
                continue

            # .. table::  表格开头，跳过直到表格结束
            if directive.startswith("table::"):
                i += 1
                while i < len(lines):
                    s = lines[i].strip()
                    if s.startswith("+") or s.startswith("|"):
                        i += 1
                    else:
                        break
                continue

            # .. tabs::  多语言代码标签
            if directive.startswith("tabs::"):
                i += 1
                continue

            # .. code-tab:: gdscript GDScript  →  [GDScript]
            if directive.startswith("code-tab::"):
                lang_match = re.match(r"code-tab::\s*(\S+)", directive)
                if lang_match:
                    lang = lang_match.group(1)
                    cleaned.append(f"\n[{lang}]")
                i += 1
                continue

            # 语义指令  note:: / warning:: / seealso::
            sem_match = re.match(r"^(note|warning|seealso|tip|important|error)::\s*(.*)$",
                                 directive, re.DOTALL)
            if sem_match:
                directive_type = sem_match.group(1)
                body = sem_match.group(2)
                # 收集后续缩进段落
                i += 1
                while i < len(lines) and (
                    lines[i].strip() == "" or lines[i].startswith("   ") or lines[i].startswith("\t")
                ):
                    if lines[i].strip():
                        body += " " + lines[i].strip()
                    i += 1
                cleaned.append(SEMANTIC_DIRECTIVES[directive_type](body))
                continue

            # 其他未知指令，丢弃该行
            i += 1
            continue

        # ---- 普通行 ----
        cleaned.append(line)
        i += 1

    text = "\n".join(cleaned)

    # ---- 第二轮：内联标记清理 ----

    # :ref:`text<target>`  →  text
    text = re.sub(r":ref:`([^<]+)\s*<[^>]+>`", r"\1", text)

    # :doc:`text <path>`  →  text
    text = re.sub(r":doc:`([^<]+)\s*<[^>]+>`", r"\1", text)

    # :ui:`X` / :kbd:`X` / :inspector:`X` / :button:`X` / :enum:`X` / :class:`X`
    text = re.sub(r":(ui|kbd|inspector|button|enum|class|meth|member|paramtype):`([^`]+)`",
                  r"\2", text)

    # |void| / |virtual| / |const| / |static| / |vararg|
    text = re.sub(r"\|(void|virtual|const|static|vararg)\|", r"\1", text)

    # |bitfield|[<...>]  →  bitfield
    text = re.sub(r"\|bitfield\|\s*\[[^\]]*\]", "bitfield", text)

    # 反斜杠转义的星号  \ **text**  →  **text**
    text = re.sub(r"\\\s*\*\*", "**", text)

    # 反斜杠转义的星号单边  \ *text* → *text*
    text = re.sub(r"\\\s\*", " *", text)

    # 类参考中的分隔线 ---- 在非表格行中表示 item 分界，保留为空行
    # （已在分块逻辑中处理，这里只做标记保留）

    # 反斜杠转义空格  \ ( → (
    text = text.replace("\\ ", " ")

    # 清除 🔗 等 Unicode 链接符号
    text = text.replace("\U0001f517", "")

    # 合并连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去掉首尾空白
    text = text.strip()

    return text


# ============================================================================
# RST 标题检测
# ============================================================================

def is_header_underline(line: str) -> Optional[int]:
    """
    检测一条线是否为 RST 标题下划线。
    返回 1 (=), 2 (-), 3 (~), 4 (^)，否则 None。
    """
    s = line.strip()
    if not s or len(s) < 2:
        return None
    ch = s[0]
    if ch not in "=-~^":
        return None
    if all(c == ch for c in s):
        if ch == "=":
            return 1
        if ch == "-":
            return 2
        if ch == "~":
            return 3
        if ch == "^":
            return 4
    return None


# ============================================================================
# 文档类型判断
# ============================================================================

def classify_doc(relpath: str) -> str:
    """根据相对路径判断文档类型。"""
    # relpath 是相对路径，没有前导斜杠，如 classes/class_node.rst.txt
    if relpath.startswith("classes/") or relpath.startswith("classes\\"):
        return "class_reference"
    if relpath.startswith("tutorials/") or relpath.startswith("tutorials\\"):
        return "tutorial"
    if relpath.startswith("getting_started/") or relpath.startswith("getting_started\\"):
        return "guide"
    return "reference"


# ============================================================================
# 类参考文档分块
# ============================================================================

# 区段标记
_REFTABLE_MARKER = "classref-reftable-group"
_DESC_MARKER = "classref-descriptions-group"
_INTRO_MARKER = "classref-introduction-group"
_SEPARATOR_MARKER = "classref-section-separator"
_ITEM_SEP_MARKER = "classref-item-separator"

# 各 item 类型标记
_ITEM_MARKERS = {
    "signal":   re.compile(r"^\.\. _class_\w+_signal_"),
    "method":   re.compile(r"^\.\. _class_\w+_method_"),
    "property": re.compile(r"^\.\. _class_\w+_property_"),
    "enum":     re.compile(r"^\.\. _class_\w+_constant_|^\.\. _enum_\w+"),
}


def _find_section_markers(raw_lines: List[str]) -> List[Dict]:
    """
    返回类参考中各区段边界：[
      {"type": "introduction", "start": N, "end": M},
      {"type": "reftable",      "start": N, "end": M},
      {"type": "descriptions",  "start": N, "end": M},
    ]
    """
    regions = []
    current_type = None
    current_start = None

    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        new_type = None
        if _INTRO_MARKER in stripped:
            new_type = "introduction"
        elif _REFTABLE_MARKER in stripped:
            new_type = "reftable"
        elif _DESC_MARKER in stripped:
            new_type = "descriptions"
        elif _SEPARATOR_MARKER in stripped:
            new_type = "separator"

        if new_type is not None:
            if current_type is not None:
                regions.append({"type": current_type, "start": current_start, "end": idx})
            current_type = new_type
            current_start = idx

    # 最后一个区段
    if current_type is not None:
        regions.append({"type": current_type, "start": current_start, "end": len(raw_lines)})

    return regions


def _split_classref_items(lines: List[str], doc_type: str) -> List[str]:
    """
    在 descriptions 区段内，按 ---- 分隔符拆分为独立 item 文本。
    """
    items = []
    buf: List[str] = []
    for line in lines:
        # 跳过 rst-class 标记（已在 clean_rst 处理后不应存在，但防一手）
        s = line.strip()
        if s.startswith(".. rst-class::"):
            continue
        if s == "----" and buf:
            # ---- 是 item 分隔符
            items.append("\n".join(buf).strip())
            buf = []
            continue
        buf.append(line)
    if buf:
        items.append("\n".join(buf).strip())
    return items


def chunk_class_reference(cleaned_text: str, raw_text: str, relpath: str) -> List[Dict]:
    """
    类参考专用分块：
    - introduction 区段按标题拆分成块
    - reftable 区段跳过
    - descriptions 区段按 item 拆分，每个 item 一个块
    """
    raw_lines = raw_text.split("\n")
    regions = _find_section_markers(raw_lines)

    doc_title = Path(relpath).stem.replace(".rst", "")
    chunks = []

    # 清洗后的文本行（与 raw 对应）
    cleaned_lines = cleaned_text.split("\n")

    for region in regions:
        if region["type"] == "reftable" or region["type"] == "separator":
            continue

        # 提取对应区域的清洗后文本
        # 由于清洗删除了很多行，行号不对齐。直接用 raw 行范围过滤后清洗。
        region_raw = "\n".join(raw_lines[region["start"]:region["end"]])
        region_clean = clean_rst(region_raw)
        if not region_clean.strip():
            continue

        if region["type"] == "introduction":
            # 按标题分块
            sub_chunks = chunk_by_headers(region_clean, doc_title, relpath, "class_reference")
            chunks.extend(sub_chunks)

        elif region["type"] == "descriptions":
            # 按 item 分隔符拆分
            items = _split_classref_items(region_clean.split("\n"), "class_reference")
            for item_text in items:
                text = item_text.strip()
                if len(text) < CHUNK_MIN_CHARS:
                    continue
                # 尝试提取 item 名称（信号名/方法名/枚举名）
                section = ""
                for line in text.split("\n")[:3]:
                    line = line.strip()
                    if line and not line.startswith("[") and len(line) < 120:
                        section = line
                        break
                chunks.append({
                    "text": text,
                    "source": relpath,
                    "doc_title": doc_title,
                    "section": section or doc_title,
                    "doc_type": "class_reference",
                })

    return chunks


# ============================================================================
# 通用标题分块
# ============================================================================

def chunk_by_headers(cleaned_text: str, doc_title: str, relpath: str,
                     doc_type: str) -> List[Dict]:
    """按 RST 标题层级分块。"""
    lines = cleaned_text.split("\n")
    chunks = []
    buf: List[str] = []
    current_section = ""
    prev_is_header_text = False

    for i, line in enumerate(lines):
        # 检查下一行是否为标题下划线
        next_underline = None
        if i + 1 < len(lines):
            next_underline = is_header_underline(lines[i + 1])

        if next_underline and line.strip():
            # 当前行是标题文本，下一行是下划线
            # 保存上一个块
            text = "\n".join(buf).strip()
            if len(text) >= CHUNK_MIN_CHARS:
                chunks.append({
                    "text": text,
                    "source": relpath,
                    "doc_title": doc_title,
                    "section": current_section,
                    "doc_type": doc_type,
                })
            buf = []
            current_section = line.strip()
            buf.append(line.strip())
            continue

        if line.strip() or (buf and buf[-1] != ""):
            buf.append(line)

    # 最后一个块
    text = "\n".join(buf).strip()
    if len(text) >= CHUNK_MIN_CHARS:
        chunks.append({
            "text": text,
            "source": relpath,
            "doc_title": doc_title,
            "section": current_section,
            "doc_type": doc_type,
        })

    return chunks


# ============================================================================
# 长块切分 & 小碎块合并
# ============================================================================

def split_long_chunks(chunks: List[Dict], max_chars: int = CHUNK_MAX_CHARS) -> List[Dict]:
    """把过长的块按段落边界切分为更小的块。"""
    result = []
    for c in chunks:
        text = c["text"]
        if len(text) <= max_chars:
            result.append(c)
            continue
        # 按双换行切分
        paragraphs = text.split("\n\n")
        current_paras = []
        current_len = 0
        for para in paragraphs:
            if current_len + len(para) > max_chars and current_paras:
                result.append({**c, "text": "\n\n".join(current_paras).strip()})
                current_paras = [para]
                current_len = len(para)
            else:
                current_paras.append(para)
                current_len += len(para)
        if current_paras:
            merged = "\n\n".join(current_paras).strip()
            if len(merged) <= max_chars:
                if len(merged) >= CHUNK_MIN_CHARS:
                    result.append({**c, "text": merged})
            else:
                # 单个段落仍超长 → 按单行切分
                lines = merged.split("\n")
                line_buf = []
                line_len = 0
                for line in lines:
                    if line_len + len(line) > max_chars and line_buf:
                        result.append({**c, "text": "\n".join(line_buf).strip()})
                        line_buf = [line]
                        line_len = len(line)
                    else:
                        line_buf.append(line)
                        line_len += len(line)
                if line_buf:
                    final = "\n".join(line_buf).strip()
                    if len(final) >= CHUNK_MIN_CHARS:
                        result.append({**c, "text": final})
    return result


def merge_small_chunks(chunks: List[Dict], min_chars: int = CHUNK_MIN_CHARS,
                      max_chars: int = CHUNK_MAX_CHARS) -> List[Dict]:
    """合并同一文件的相邻小碎块，达到合适的大小。"""
    if not chunks:
        return []

    # 按 source 分组处理（连续的同源块）
    result = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]

        # 如果块已经够大，直接保留
        if len(chunk["text"]) >= min_chars:
            result.append(chunk)
            i += 1
            continue

        # 合并后续同源的小块
        merged_text = chunk["text"]
        merged_sections = [chunk["section"]]
        j = i + 1
        while j < len(chunks) and chunks[j]["source"] == chunk["source"]:
            candidate = chunks[j]
            combined_len = len(merged_text) + len(candidate["text"]) + 2  # +2 for \n\n
            if combined_len > max_chars:
                break
            merged_text += "\n\n" + candidate["text"]
            if candidate["section"] and candidate["section"] not in merged_sections:
                merged_sections.append(candidate["section"])
            j += 1

        # 只要合并后 >= min_chars 就保留
        if len(merged_text) >= min_chars:
            new_chunk = dict(chunk)
            new_chunk["text"] = merged_text
            new_chunk["section"] = " / ".join(merged_sections) or chunk["section"]
            result.append(new_chunk)
        # 如果合并后仍太小，丢弃
        i = j

    return result


# ============================================================================
# 主流程
# ============================================================================

# 排除列表：这些文件不是文档内容，不应该被索引
SKIP_FILES = {
    "404.rst.txt",
    "index.rst.txt",
    "genindex.rst.txt",
    "search.rst.txt",
    # 纯参考数据，不是文档内容
    "locales.rst.txt",
}


def parse_file(filepath: Path) -> List[Dict]:
    """解析单个 RST 文件，返回块列表。"""
    relpath = str(filepath.relative_to(SOURCE_DIR)).replace("\\", "/")

    # 跳过非内容文件
    if filepath.name in SKIP_FILES:
        return []

    doc_type = classify_doc(relpath)

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    doc_title = filepath.stem.replace(".rst", "")

    if doc_type == "class_reference":
        cleaned = clean_rst(raw_text)
        return chunk_class_reference(cleaned, raw_text, relpath)
    else:
        cleaned = clean_rst(raw_text)
        return chunk_by_headers(cleaned, doc_title, relpath, doc_type)


def build(args: argparse.Namespace):
    """主构建流程。"""
    print("=" * 60)
    print("Godot 文档向量知识库 — 构建")
    print("=" * 60)
    print(f"  源目录:   {SOURCE_DIR}")
    print(f"  输出目录: {PERSIST_DIR}")
    print(f"  模型:     {MODEL_PATH}")
    print(f"  CPU 限制: {args.workers or '自动'}")
    print()

    # 1. 收集文件
    rst_files = sorted(SOURCE_DIR.glob("**/*.rst.txt"))
    print(f"找到 {len(rst_files)} 个 RST 源文件")

    if args.dry_run:
        print("\n[Dry-run] 只解析，不做嵌入和存储\n")

    # 2. 解析所有文件
    print("\n解析 RST 并分块...")
    all_chunks: List[Dict] = []
    errors = 0
    for fp in tqdm(rst_files, desc="解析"):
        try:
            chunks = parse_file(fp)
            all_chunks.extend(chunks)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  [错误] {fp}: {e}", file=sys.stderr)

    if errors:
        print(f"\n共 {errors} 个文件解析出错", file=sys.stderr)

    print(f"初步分块: {len(all_chunks)} 个")

    # 3. 切分长块
    all_chunks = split_long_chunks(all_chunks, CHUNK_MAX_CHARS)
    print(f"切分后:   {len(all_chunks)} 个")

    # 3.5 合并小碎块
    all_chunks = merge_small_chunks(all_chunks, CHUNK_MIN_CHARS, CHUNK_MAX_CHARS)
    print(f"合并后:   {len(all_chunks)} 个")

    if not all_chunks:
        print("没有生成任何块，退出。", file=sys.stderr)
        sys.exit(1)

    # 统计
    sizes = [len(c["text"]) for c in all_chunks]
    print(f"块大小:   min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)//len(sizes)}")

    # 各类型文档数量
    type_counts = {}
    for c in all_chunks:
        t = c["doc_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n} 块")

    if args.dry_run:
        # 打印几个样例
        print("\n--- 样例块 ---")
        for i in range(min(3, len(all_chunks))):
            c = all_chunks[i]
            print(f"\n{'='*50}")
            print(f"来源: {c['source']}")
            print(f"标题: {c['doc_title']} > {c['section']}")
            print(f"类型: {c['doc_type']} | 长度: {len(c['text'])}")
            print(f"{'='*50}")
            print(c["text"][:500])
        return

    # 4. 加载嵌入模型
    print(f"\n加载嵌入模型: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, device="cpu")

    # 5. 分批生成嵌入 + 写入 ChromaDB（节省内存）
    CHUNK_EMBED = 2000  # 每批嵌入的文本数
    print(f"\n生成嵌入并写入 (batch_size={BATCH_SIZE_EMBED}, chunk={CHUNK_EMBED})...")

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    # 删除旧集合
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in all_chunks]
    total_embedded = 0
    for chunk_start in tqdm(range(0, len(texts), CHUNK_EMBED), desc="嵌入进度"):
        chunk_end = min(chunk_start + CHUNK_EMBED, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_chunks = all_chunks[chunk_start:chunk_end]

        # 生成嵌入
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=BATCH_SIZE_EMBED,
            show_progress_bar=False,
        )

        # 写入
        collection.add(
            ids=[f"c{j}" for j in range(chunk_start, chunk_end)],
            embeddings=chunk_embeddings.tolist(),
            documents=chunk_texts,
            metadatas=[{
                "source": c["source"],
                "doc_title": c["doc_title"],
                "section": c["section"],
                "doc_type": c["doc_type"],
            } for c in chunk_chunks],
        )
        total_embedded += len(chunk_texts)

    # 6. 报告
    total_size = sum(f.stat().st_size for f in PERSIST_DIR.rglob("*") if f.is_file())
    print(f"\n{'=' * 60}")
    print(f"完成！")
    print(f"  块数:    {total_embedded}")
    print(f"  索引大小: {total_size / 1024 / 1024:.1f} MB")
    print(f"  输出目录: {PERSIST_DIR.resolve()}")
    print(f"\n现在可以用 ask.py 查询了。")
    print(f"也可以把 index_data/ 整个目录拷贝到其他机器使用。")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Godot 文档向量知识库构建")
    parser.add_argument("--workers", type=int, default=None,
                        help="限制 CPU 线程数 (默认自动)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只解析分块，不做嵌入和存储")
    args = parser.parse_args()

    # 设置 CPU 线程
    if args.workers:
        os.environ["OMP_NUM_THREADS"] = str(args.workers)
        os.environ["MKL_NUM_THREADS"] = str(args.workers)
        try:
            import torch
            torch.set_num_threads(args.workers)
        except ImportError:
            pass

    build(args)
