#!/usr/bin/env python3
"""
Godot 文档向量知识库 — 查询
=============================
用法:
    python ask.py "你的问题"              # 中文/英文都可以
    python ask.py "信号怎么用" -k 10       # 返回 top-10 结果
    python ask.py                         # 交互模式

需要先运行 build.py 生成 index_data/ 目录。
"""

import re
import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================================
# 配置
# ============================================================================

PERSIST_DIR = Path("index_data")
LOCAL_MODEL_DIRS = [
    Path("bge-m3-model"),       # 同级目录
    Path("../bge-m3"),           # 上级目录
]
_MODEL_DIR = None
for _d in LOCAL_MODEL_DIRS:
    if _d.exists():
        _MODEL_DIR = _d
        break
MODEL_NAME = "BAAI/bge-m3"                       # HuggingFace 模型（备选）
MODEL_PATH = str(_MODEL_DIR) if _MODEL_DIR else MODEL_NAME
COLLECTION_NAME = "godot_docs"
DEFAULT_TOP_K = 5


# ============================================================================
# 查询函数
# ============================================================================

class GodotDocsSearch:
    """Godot 文档向量检索器。"""

    def __init__(self):
        if not PERSIST_DIR.exists():
            print(f"错误: 找不到索引目录 {PERSIST_DIR.resolve()}")
            print("请先运行 build.py 构建索引。")
            sys.exit(1)

        print(f"加载模型: {MODEL_PATH} ...", file=sys.stderr)
        self.model = SentenceTransformer(MODEL_PATH, device="cpu")

        print(f"加载索引: {PERSIST_DIR.resolve()} ...", file=sys.stderr)
        self.client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"就绪 (共 {self.collection.count()} 个文档块)\n", file=sys.stderr)

    def search(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict:
        """搜索并返回 ChromaDB 原始结果。"""
        embedding = self.model.encode([question], show_progress_bar=False)
        return self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=top_k,
        )

    def fetch_by_ids(self, ids: list) -> dict:
        """按 ID 批量获取文档。"""
        if not ids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

    def get_neighbors(self, doc_id: str, n: int = 2) -> list:
        """获取某个块前后各 n 个相邻块（仅限同一源文件）。"""
        match = re.match(r"c(\d+)", doc_id)
        if not match:
            return []
        idx = int(match.group(1))
        ref_meta = self.collection.get(ids=[doc_id], include=["metadatas"])
        if not ref_meta["metadatas"]:
            return []
        ref_source = ref_meta["metadatas"][0]["source"]

        # 探测前后各 n 个 ID
        neighbors = []
        for offset in range(-n, n + 1):
            if offset == 0:
                continue
            candidate_id = f"c{idx + offset}"
            if int(idx) + offset < 0 or int(idx) + offset >= self.collection.count():
                continue
            try:
                result = self.collection.get(
                    ids=[candidate_id],
                    include=["documents", "metadatas"],
                )
                if result["metadatas"] and result["metadatas"][0]["source"] == ref_source:
                    neighbors.append({
                        "id": candidate_id,
                        "text": result["documents"][0],
                        "meta": result["metadatas"][0],
                        "offset": offset,
                    })
            except Exception:
                pass

        return sorted(neighbors, key=lambda x: x["offset"])


# ============================================================================
# 输出格式化
# ============================================================================

def clean_display(text: str) -> str:
    """清理 RST 残留标记，输出干净的纯文本。"""
    import re
    # 反斜杠转义（\空格、\冒号等）
    text = text.replace("\\ ", " ")
    text = text.replace("\\:", ":")
    text = text.replace("\\(", "(")
    text = text.replace("\\)", ")")
    text = text.replace("\\.", ".")
    # 移除 🔗 Unicode 链接符号
    text = text.replace("\U0001f517", "")
    # RST 行内标记残留  |void| |virtual| |const| |static| |vararg|
    for token in ["void", "virtual", "const", "static", "vararg"]:
        text = text.replace(f"|{token}|", token)
    # 多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 首尾空白
    return text.strip()


def print_results(question: str, results: dict, searcher=None,
                  context_n: int = 2):
    """格式化打印查询结果，包含每个命中块的前后上下文。"""
    print(f"查询: {question}")
    print()

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("没有找到相关结果。")
        return

    for i, (doc_id, doc_text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        similarity = round(1 - dist, 4)
        print(f"{'─' * 70}")
        print(f"#{i + 1}  [{similarity:.2%}]  {meta['doc_title']} › {meta['section']}")
        print(f"     来源: {meta['source']}  ({meta['doc_type']})")
        print(f"{'─' * 70}")

        # 获取上下文相邻块
        neighbors = []
        if searcher and context_n > 0:
            neighbors = searcher.get_neighbors(doc_id, context_n)

        # 前文块
        prev = [n for n in neighbors if n["offset"] < 0]
        for n in prev:
            txt = clean_display(n["text"])
            print(f"  ... [上文]\n{txt[:300]}")
            if len(n["text"]) > 300:
                print("  ... (截断)")
            print()

        # 命中块主体
        display = clean_display(doc_text)
        print(f"  >>> [命中] <<<")
        print(display[:800])
        if len(doc_text) > 800:
            print("... (截断)")
        print()

        # 后文块
        next_ = [n for n in neighbors if n["offset"] > 0]
        for n in next_:
            txt = clean_display(n["text"])
            print(f"  ... [下文]\n{txt[:300]}")
            if len(n["text"]) > 300:
                print("  ... (截断)")

        print()


# ============================================================================
# 交互模式
# ============================================================================

def interactive(searcher: GodotDocsSearch, context_n: int = 2):
    """交互式查询循环。"""
    print("Godot 文档搜索 (输入 /quit 退出, /help 查看帮助)")
    top_k = DEFAULT_TOP_K

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not q:
            continue

        if q.startswith("/"):
            cmd_parts = q.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd == "/quit" or cmd == "/q":
                print("再见。")
                break
            elif cmd == "/help":
                print("  输入问题即可搜索")
                print("  /help      — 帮助")
                print("  /k N       — 设置返回数量 (默认 5)")
                print("  /c N       — 设置上下文块数 (默认 2)")
                print("  /stats     — 索引统计")
                print("  /quit      — 退出")
            elif cmd == "/k":
                try:
                    top_k = int(arg)
                    print(f"  top_k = {top_k}")
                except ValueError:
                    print(f"  无效数字: {arg}")
            elif cmd == "/c":
                try:
                    context_n = int(arg)
                    print(f"  context = ±{context_n} 块")
                except ValueError:
                    print(f"  无效数字: {arg}")
            elif cmd == "/stats":
                print(f"  总块数: {searcher.collection.count()}")
            else:
                print(f"  未知命令: {cmd} (输入 /help 查看帮助)")
        else:
            results = searcher.search(q, top_k)
            print_results(q, results, searcher, context_n)


# ============================================================================
# 入口
# ============================================================================

def main():
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return

    # 解析参数
    question_parts = []
    top_k = DEFAULT_TOP_K
    context_n = 2
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i].startswith("-k="):
            top_k = int(sys.argv[i][3:])
            i += 1
        elif sys.argv[i] == "-c" and i + 1 < len(sys.argv):
            context_n = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i].startswith("-c="):
            context_n = int(sys.argv[i][3:])
            i += 1
        else:
            question_parts.append(sys.argv[i])
            i += 1

    question = " ".join(question_parts).strip()

    # 加载（stderr 输出加载信息）
    searcher = GodotDocsSearch()

    if question:
        results = searcher.search(question, top_k)
        print_results(question, results, searcher, context_n)
    else:
        interactive(searcher, context_n)


if __name__ == "__main__":
    main()
