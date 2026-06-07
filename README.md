# 🎮 Godot Docs RAG

> 基于 RAG（检索增强生成）的 Godot 引擎文档语义搜索系统  
> A Retrieval-Augmented Generation system for semantic search across Godot Engine documentation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6%2B-orange)](https://www.trychroma.com/)
[![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3-green)](https://huggingface.co/BAAI/bge-m3)
[![MCP](https://img.shields.io/badge/MCP-1.0%2B-purple)](https://modelcontextprotocol.io/)

---

## 📖 简介 | Introduction

**Godot Docs RAG** 是一个专门为 [Godot 游戏引擎](https://godotengine.org/) 官方文档打造的语义搜索系统。它将 Godot 文档的 RST 源文件通过向量化索引到 ChromaDB 中，支持**语义检索**和**跨语言查询**。

无论你是使用中文还是英文提问，系统都能理解你的意图，从数千页文档中找到最相关的内容。

This project indexes Godot's official RST documentation into a vector database (ChromaDB) using the BGE-M3 multilingual embedding model. It enables **semantic search** across the Godot docs — find relevant content by meaning, not just keywords — and supports **cross-language queries** (e.g., ask in Chinese, find English docs).

### ✨ 核心特性 | Features

- **🔍 语义搜索** — 基于句子嵌入的语义理解，而非简单的关键词匹配
- **🌐 跨语言支持** — BGE-M3 多语言模型，支持中英文混合查询
- **📦 离线运行** — 所有嵌入和检索均在本地 CPU 完成，无需 API 调用
- **🧩 MCP 集成** — 通过 [Model Context Protocol](https://modelcontextprotocol.io/) 为 Claude Code 等 AI 助手提供文档搜索工具
- **💻 交互式 CLI** — 支持交互式 REPL 和单次查询两种模式
- **🔄 构建恢复** — 支持断点续传，应对大规模文档索引
- **📄 上下文感知** — 检索结果自动包含前后文，提供更完整的参考信息

---

## 🏗️ 项目结构 | Project Structure

```
godot-docs-RAG/
├── build.py              # 文档索引构建脚本 (ETL pipeline)
├── ask.py                # 命令行语义搜索工具
├── finish_build.py       # 构建恢复工具 (断点续传)
├── mcp_server.py         # MCP 服务器 (AI 助手集成)
├── requirements.txt      # Python 依赖
│
├── _sources/             # 📚 Godot 文档 RST 源文件
│   ├── classes/          #     API 类参考 (1079个文件)
│   ├── tutorials/        #     教程 (2D, 3D, 动画, 着色器等)
│   ├── getting_started/  #     入门指南
│   ├── about/            #     关于 Godot
│   ├── community/        #     社区资源
│   └── engine_details/   #     引擎内部文档
│
├── index_data/           # 🗄️ ChromaDB 向量数据库
│   ├── chroma.sqlite3    #     元数据与文档存储
│   └── <uuid>/           #     HNSW 向量索引文件
│
└── README.md             # 本文件
```

---

## 🚀 快速开始 | Quick Start

### 环境要求 | Prerequisites

- Python 3.10+
- Git
- 至少 4GB 可用内存（加载 BGE-M3 模型）

### 安装 | Installation

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/godot-docs-RAG.git
cd godot-docs-RAG

# 2. 安装依赖
pip install -r requirements.txt
```

### 3. 构建索引 | Build the Index

解析 RST 源文件，分块并生成嵌入，存储到 ChromaDB：

```bash
python build.py
```

> 📦 构建过程在 CPU 上运行，耗时取决于文档数量（~1500+ 文件），首次构建预计 10-30 分钟。  
> 💡 如果构建中断，可使用 `finish_build.py` 恢复：`python finish_build.py`

可用参数：

| 参数 | 说明 |
|------|------|
| `--dry-run` | 仅解析文件，不执行嵌入 |
| `--workers N` | 控制 CPU 线程数（默认使用所有核心） |

### 4. 搜索文档 | Search the Docs

#### 🖥️ 单次查询 | One-shot Query

```bash
python ask.py "How to use AnimationPlayer" -k 5
```

参数：
- `-k, --top-k` — 返回结果数量（默认 5）
- `-c, --context` — 每条结果前后文块数（默认 2）

#### 💬 交互模式 | Interactive Mode

```bash
python ask.py
```

进入交互式 REPL，支持以下命令：

| 命令 | 说明 |
|------|------|
| `/k N` | 设置返回结果数 |
| `/c N` | 设置上下文块数 |
| `/stats` | 显示数据库统计信息 |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

---

## 🤖 MCP 服务器 | MCP Server

MCP 服务器将 Godot 文档搜索暴露为标准工具，供任何 MCP 兼容的 AI 客户端（如 Claude Code）使用。

### 启动 | Start

```bash
python mcp_server.py
```

### 暴露的工具 | Exposed Tool

**`search_godot_docs(query, top_k=5, context=2)`**

- `query` (str) — 搜索查询
- `top_k` (int, default: 5) — 返回结果数
- `context` (int, default: 2) — 每条结果的前后文块数

返回带相关度评分的 Markdown 格式文档片段。

### 在 Claude Code 中配置 | Configure with Claude Code

将以下内容添加到你的 `~/.claude/settings.json`：

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

然后在对话中使用：
> 请帮我搜索 Godot 文档中关于 signal 的内容  
> *→ 自动调用 `search_godot_docs` 工具*

---

## 🔧 技术栈 | Tech Stack

| 组件 | 技术 | 用途 |
|------|------|------|
| **文档源** | Godot RST (reStructuredText) | 官方文档原始格式 |
| **嵌入模型** | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 多语言语义嵌入 (1024维) |
| **向量数据库** | [ChromaDB](https://www.trychroma.com/) | 向量存储与近似最近邻检索 |
| **索引算法** | HNSW (余弦距离) | 高效的近似最近邻搜索 |
| **AI 集成** | [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) | AI 助手工具接口 |

---

## 📊 工作流程 | How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  RST 文档源      │     │  build.py       │     │  ChromaDB       │
│  (_sources/)     │ ──▶ │  清洗 → 分块    │ ──▶ │  向量存储       │
│  ~1582 个文件    │     │  BGE-M3 嵌入     │     │  (index_data/)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                    ┌────────────────────────────────────┤
                    │                    │               │
                    ▼                    ▼               ▼
            ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
            │  ask.py      │   │ MCP Server   │   │ finish_build │
            │  CLI 搜索     │   │ AI 集成       │   │ 断点续传      │
            └──────────────┘   └──────────────┘   └──────────────┘
```

**处理流程：**

1. **提取** — 扫描 `_sources/` 下的所有 `.rst.txt` 文件（1582 个）
2. **清洗** — 去除 RST 标记（指令、标签、格式符号等）
3. **分块** — 按标题层级智能分块，合并碎块，拆分长块
4. **嵌入** — 使用 BGE-M3 模型将文本块转为 1024 维向量
5. **索引** — HNSW 算法构建近似最近邻索引，存入 ChromaDB
6. **检索** — 查询时同样嵌入用户输入，在向量空间中搜索最近邻

---

## 🤝 贡献 | Contributing

欢迎贡献！请随时提交 Issue 或 Pull Request。

---

## 📄 许可 | License

本项目代码基于 MIT 许可证开源。

Godot 文档源文件 (`_sources/`) 归 Godot 引擎社区所有，基于 [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) 许可。
