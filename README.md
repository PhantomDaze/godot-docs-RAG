# Godot Docs RAG

Godot 官方文档的离线语义搜索系统。基于 BGE-M3 + ChromaDB，支持中英文混合检索，可直接集成 Claude Code 等 AI 助手。

## 快速开始

```bash
pip install -r requirements.txt

# 更新文档源（自动选最快镜像下载）
python update_docs.py

# 构建索引（首次运行自动下载 BGE-M3 模型 ~2GB，耗时约 10-30 分钟）
python build.py

# 限制 CPU 核数（适合在服务器上运行）
python build.py --workers 4
```

## 搜索

```bash
# 单次查询
python ask.py "How to use AnimationPlayer" -k 5

# 交互式 REPL
python ask.py

# 交互命令: /k N 设置结果数  /c N 设置上下文  /stats 统计  /help  /quit
```

## 文档源更新

```bash
python update_docs.py                     # 自动选最快源下载
python update_docs.py --source ghproxy.com # 手动指定代理源
python update_docs.py --list-sources       # 列出所有可用源
python update_docs.py --branch main        # 使用开发分支
python update_docs.py --dry-run            # 预览变更
```

内置 GitHub 直连及多个国内加速源，自动测速选最优。

## MCP 集成（AI 助手调用）

```bash
python mcp_server.py
```

暴露工具 `search_godot_docs(query, top_k=5, context=2)`。

### Claude Code 配置

添加到 `~/.claude/settings.json`：

```json
{
  "mcpServers": {
    "godot-docs": {
      "command": "python",
      "args": ["path/to/mcp_server.py"]
    }
  }
}
```

## 向量模型

本项目使用 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) 作为嵌入模型（1024 维，支持多语言）。

### 模型缓存位置

`sentence-transformers` 首次加载模型时会自动从 HuggingFace Hub 下载：

| 平台 | 缓存路径 |
|------|----------|
| Linux / macOS | `~/.cache/huggingface/hub/models--BAAI--bge-m3/` |
| Windows | `C:\Users\<用户名>\.cache\huggingface\hub\models--BAAI--bge-m3\` |

模型文件约 2.2 GB，下载后只需一次，后续构建和查询均使用缓存。

### 离线部署 / 本地模型

如果需要完全离线运行，可在联网机器下载模型后放入项目目录：

```bash
# 1. 联网机器上下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# 2. 将缓存目录的内容复制到项目下的 bge-m3-model/
cp -r ~/.cache/huggingface/hub/models--BAAI--bge-m3/* /path/to/project/bge-m3-model/

# 3. build.py 启动时会优先检测项目目录下的 bge-m3-model/（或 ../bge-m3/）
#    存在则使用本地文件，不再联网拉取
python build.py
```

支持自动检测的本地模型目录（按优先级）：
1. `./bge-m3-model/`
2. `../bge-m3/`
3. 兜底：从 HuggingFace Hub 在线拉取

## 离线部署（查询端）

构建完成后的 `index_data/` 目录可拷贝到其他机器直接使用：

```bash
# 目标机器只需安装依赖
pip install chromadb sentence-transformers tqdm

# 把 index_data/ 复制过去，连同 ask.py（或 mcp_server.py）即可查询
python ask.py "AnimationPlayer" -k 3
```

> 查询端首次运行同样会自动下载 BGE-M3 模型到缓存目录，也可以按上一节方式预置本地模型。

## 工具一览

| 命令 | 用途 |
|------|------|
| `build.py` | 解析 RST → 清洗分块 → BGE-M3 嵌入 → 写入 ChromaDB |
| `ask.py` | CLI 语义搜索（单次 / REPL） |
| `mcp_server.py` | MCP 服务器，AI 助手可调用文档搜索 |
| `update_docs.py` | 从 GitHub 下载最新 Godot 文档并同步 `_sources/` |
| `finish_build.py` | 构建中断后断点续传 |

## 项目结构

```
_sources/       Godot 文档 RST 源文件（classes/ tutorials/ 等）
index_data/     ChromaDB 向量索引（自动生成）
bge-m3-model/   (可选) 本地模型文件，放入后可离线运行
```

## 许可证

- **代码** — MIT
- **文档** (`_sources/`) — CC BY 4.0 © Godot Engine Community
