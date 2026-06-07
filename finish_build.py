#!/usr/bin/env python3
"""补完 build.py 未完成的部分，小批量嵌入避免段错误。"""

import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 复用 build.py 的解析逻辑
sys.path.insert(0, str(Path(__file__).parent))
from build import parse_file, merge_small_chunks, split_long_chunks, \
    SOURCE_DIR, PERSIST_DIR, COLLECTION_NAME, CHUNK_MIN_CHARS, CHUNK_MAX_CHARS, SKIP_FILES

MODEL_PATH = "../bge-m3"
BATCH_SIZE = 100  # 小批量
EMBED_BATCH = 16

# 1. 解析所有文件
print("重新解析 RST...")
rst_files = sorted(SOURCE_DIR.glob("**/*.rst.txt"))
all_chunks = []
for fp in tqdm(rst_files, desc="解析"):
    if fp.name in SKIP_FILES:
        continue
    try:
        all_chunks.extend(parse_file(fp))
    except Exception as e:
        pass

all_chunks = split_long_chunks(all_chunks, CHUNK_MAX_CHARS)
all_chunks = merge_small_chunks(all_chunks, CHUNK_MIN_CHARS, CHUNK_MAX_CHARS)
print(f"总块数: {len(all_chunks)}")

# 2. 连接 ChromaDB，查已存了多少
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = client.get_collection(COLLECTION_NAME)
stored = collection.count()
print(f"已存储: {stored}")

if stored >= len(all_chunks):
    print("已完成，无需补完。")
    sys.exit(0)

# 3. 补上缺失的
remaining = all_chunks[stored:]
print(f"待补: {len(remaining)} 块")

print(f"加载模型: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, device="cpu")

print(f"嵌入并写入 (batch={BATCH_SIZE})...")
texts = [c["text"] for c in remaining]
for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="补完"):
    batch = remaining[i:i + BATCH_SIZE]
    batch_texts = texts[i:i + BATCH_SIZE]
    embs = model.encode(batch_texts, batch_size=EMBED_BATCH, show_progress_bar=False)
    collection.add(
        ids=[f"c{stored + j}" for j in range(i, i + len(batch))],
        embeddings=embs.tolist(),
        documents=batch_texts,
        metadatas=[{
            "source": c["source"],
            "doc_title": c["doc_title"],
            "section": c["section"],
            "doc_type": c["doc_type"],
        } for c in batch],
    )

print(f"\n完成！总共 {collection.count()} 块")
