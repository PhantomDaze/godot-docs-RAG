#!/usr/bin/env python3
"""
Godot 文档 MCP 服务器
=====================
让 Claude Code 直接搜索 Godot 文档向量知识库。

配置 (claude_code settings.json → mcpServers):

    "godot-docs": {
        "command": "python",
        "args": ["mcp_server.py"],
        "cwd": "C:/Users/antom/Documents/gddocs/godot-docs-html-master"
    }

Linux 上把 cwd 改成对应路径即可。
"""

import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# ============================================================================
# 配置
# ============================================================================
CWD = Path(__file__).parent
PERSIST_DIR = CWD / "index_data"
MODEL_DIRS = [CWD / "bge-m3-model", CWD.parent / "bge-m3"]
MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "godot_docs"

# ============================================================================
# 初始化（模块加载时一次，MCP 启动后立即可用）
# ============================================================================
model_path = None
for d in MODEL_DIRS:
    if d.exists():
        model_path = str(d)
        break
if model_path is None:
    model_path = MODEL_NAME

_model = SentenceTransformer(model_path, device="cpu")
_collection = chromadb.PersistentClient(path=str(PERSIST_DIR)).get_collection(COLLECTION_NAME)

# ============================================================================
# MCP 应用
# ============================================================================
mcp = FastMCP(
    "godot-docs-search",
    instructions="Godot 引擎文档向量知识库。支持中英文跨语言搜索。当你需要查找 Godot API、教程、概念时使用此工具。",
)


# ============================================================================
# 搜索工具
# ============================================================================
@mcp.tool()
def search_godot_docs(query: str, top_k: int = 5, context: int = 2) -> str:
    """
    搜索 Godot 引擎文档。支持中英文查询，命中英文文档内容。

    Args:
        query: 搜索字符串，中文英文均可。
        top_k: 返回结果数，默认 5。
        context: 每个命中块前后各取几个上下文块，默认 2。设 0 则仅返回命中块。

    Returns:
        带来源和相似度标注的搜索结果。
    """
    # 编码查询
    embedding = _model.encode([query], show_progress_bar=False)
    results = _collection.query(
        query_embeddings=embedding.tolist(),
        n_results=top_k,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return "未找到相关文档。"

    out = []
    for i, (doc_id, doc_text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        score = round((1 - dist) * 100)
        out.append(
            f"## 结果 #{i + 1} (相关度 {score}%)\n"
            f"**{meta['doc_title']}** › {meta['section']}\n"
            f"来源: `{meta['source']}` ({meta['doc_type']})\n"
        )

        # 上下文块
        if context > 0:
            m = re.match(r"c(\d+)", doc_id)
            if m:
                idx = int(m.group(1))
                ref_src = meta["source"]
                neighbors = []
                for off in range(-context, context + 1):
                    if off == 0:
                        continue
                    cid = f"c{idx + off}"
                    if idx + off < 0 or idx + off >= _collection.count():
                        continue
                    try:
                        r = _collection.get(ids=[cid], include=["documents", "metadatas"])
                        if r["metadatas"] and r["metadatas"][0]["source"] == ref_src:
                            neighbors.append((off, r["documents"][0], r["metadatas"][0]))
                    except Exception:
                        pass
                neighbors.sort(key=lambda x: x[0])

                prev = [n for n in neighbors if n[0] < 0]
                nxt = [n for n in neighbors if n[0] > 0]

                if prev:
                    out.append("> 上文:")
                    for _, t, m in prev:
                        out.append(f"> {_clean(t)[:300]}")
                out.append(f">>> 命中:\n{_clean(doc_text)[:800]}")
                if nxt:
                    out.append("> 下文:")
                    for _, t, m in nxt:
                        out.append(f"> {_clean(t)[:300]}")
                out.append("")
            else:
                out.append(_clean(doc_text)[:800])
                out.append("")
        else:
            out.append(_clean(doc_text)[:800])
            out.append("")

    return "\n".join(out)


def _clean(text: str) -> str:
    """清理 RST 残留。"""
    text = text.replace("\\ ", " ").replace("\\:", ":").replace("\\(", "(")
    text = text.replace("\\)", ")").replace("\\.", ".")
    text = text.replace("\U0001f517", "")
    for t in ["void", "virtual", "const", "static", "vararg"]:
        text = text.replace(f"|{t}|", t)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    mcp.run()
