# Godot Docs RAG

Godot 官方文档的离线语义搜索系统。基于 BGE-M3 + ChromaDB，支持中英文混合检索，可直接集成 Claude Code 等 AI 助手。

## 快速开始

```bash
pip install -r requirements.txt

# 更新文档源（自动选最快镜像下载）
python update_docs.py

# 构建索引（~1500 个 RST 文件，CPU 运行 10-30 分钟）
python build.py
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
python update_docs.py                     # 自动选择最快源下载
python update_docs.py --source ghproxy.com # 手动指定代理源
python update_docs.py --list-sources       # 列出所有可用源
python update_docs.py --branch main        # 使用开发分支
python update_docs.py --dry-run            # 预览变更
```

内置 `ghproxy.com`、`gh-proxy.cn`、`hk.gh-proxy.com` 等多个国内加速源，自动测速选最优。

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
```

## 许可证

- **代码** — MIT
- **文档** (`_sources/`) — CC BY 4.0 © Godot Engine Community
