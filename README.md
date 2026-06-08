# Godot Docs RAG

Godot 官方文档的离线语义搜索系统。基于 ChromaDB，支持本地 BGE-M3 或 OpenAI 兼容 API，中英文混合检索，可直接集成 Claude Code 等 AI 助手。

## 快速开始

```bash
pip install -r requirements.txt

# 更新文档源（自动选最快镜像下载）
python update_docs.py

# 构建索引（首次运行需选择嵌入引擎）
python build.py

# 或直接指定：
python build.py --provider local    # 本地 BGE-M3
python build.py --provider openai   # OpenAI API
python build.py --device cuda       # GPU 加速
python build.py --dim 512           # 截断向量维度
python build.py --workers 4         # 限制 CPU 核数
```

## 搜索

```bash
python ask.py "How to use AnimationPlayer" -k 5
python ask.py --provider openai "signal"
python ask.py --device cuda "动画播放"
python ask.py                      # 交互式 REPL
```

交互命令: `/k N` 设置结果数  `/c N` 设置上下文  `/stats` 统计  `/help`  `/quit`

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

## 嵌入引擎

支持两种后端，通过 CLI、`.env` 文件或交互式菜单选择：

| 后端 | 命令 | 说明 |
|------|------|------|
| 本地 BGE-M3 | `--provider local` | 免费，首次需下载 ~2GB 模型 |
| OpenAI 兼容 API | `--provider openai` | 按 token 付费，约 $0.05/全量文档 |

### 选择方式（优先级）

1. **CLI 参数**: `python build.py --provider openai`
2. **`.env` 文件**: 复制 `.env.example` 为 `.env`，设置 `EMBED_PROVIDER=openai`
3. **交互菜单**: 无配置且是终端时，自动弹出选择

### GPU 加速

本地 BGE-M3 支持 CUDA (NVIDIA) 和 MPS (Apple Silicon) 加速：

```bash
python build.py --device cuda     # NVIDIA GPU
python build.py --device mps      # Apple Silicon
python ask.py --device cuda        # 查询也用 GPU
```

不指定则自动检测可用设备，检测不到则 CPU 运行。

可通过 `.env` 的 `EMBED_DEVICE=cuda` 持久化设置。

### 向量维度

BGE-M3 支持维度截断，可大幅降低存储和检索成本：

```bash
python build.py --dim 256          # 截断到 256 维
python build.py --dim 512          # 截断到 512 维（默认 1024）
```

OpenAI `text-embedding-3-small/large` 也支持自定义维度：

```bash
EMBED_MODEL=text-embedding-3-small python build.py --dim 256
```

可通过 `.env` 的 `EMBED_DIM=512` 持久化设置。

> ⚠ 切换嵌入引擎或修改维度后必须重建索引。

### OpenAI 配置

在 `.env` 中设置：

```env
EMBED_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
# OPENAI_BASE_URL=https://api.openai.com/v1
# EMBED_MODEL=text-embedding-3-small
```

### 本地模型缓存

`BAAI/bge-m3`（1024 维，多语言）：

| 平台 | 缓存路径 |
|------|----------|
| Linux / macOS | `~/.cache/huggingface/hub/models--BAAI--bge-m3/` |
| Windows | `C:\Users\<用户名>\.cache\huggingface\hub\models--BAAI--bge-m3\` |

**离线部署**: 将模型文件放入 `bge-m3-model/` 目录即可跳过在线下载。

## 工具一览

| 命令 | 用途 |
|------|------|
| `build.py` | 解析 RST → 清洗分块 → 嵌入 → 写入 ChromaDB |
| `ask.py` | CLI 搜索，支持 `--provider` `--device` `--dim` |
| `mcp_server.py` | MCP 服务器，AI 助手可调用文档搜索 |
| `update_docs.py` | 从 GitHub 下载最新 Godot 文档并同步 `_sources/` |
| `finish_build.py` | 构建中断后断点续传 |
| `embedder.py` | 嵌入引擎统一接口 |

## 项目结构

```
_sources/       Godot 文档 RST 源文件
index_data/     ChromaDB 向量索引（自动生成）
bge-m3-model/   (可选) 本地模型文件
.env.example    嵌入引擎配置模板
```

## 许可证

- **代码** — MIT
- **文档** (`_sources/`) — CC BY 4.0 © Godot Engine Community
