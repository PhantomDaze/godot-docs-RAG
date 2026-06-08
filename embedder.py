#!/usr/bin/env python3
"""
嵌入引擎 — 统一嵌入接口，支持本地模型和在线 API。

用法:
    from embedder import get_embedder
    embedder = get_embedder()                       # 自动选择
    embedder = get_embedder("openai")               # 指定 OpenAI
    embedder = get_embedder("local", device="cuda") # 指定 GPU 加速
    vecs = embedder.encode(["text1", "text2"])
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional


# ── URL 协议补全 ──────────────────────────────────────────────

def _ensure_scheme(url: Optional[str]) -> Optional[str]:
    """如果 URL 没有协议头，自动补 https://。"""
    if not url:
        return url
    import re
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url.strip()):
        url = "https://" + url.strip()
    return url.rstrip("/")


# ── .env 加载（优先于系统环境变量）──────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


# ============================================================================
# 基类
# ============================================================================

class Embedder:
    """嵌入引擎基类。"""

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """返回向量维度。"""
        raise NotImplementedError


# ============================================================================
# 设备检测
# ============================================================================

def detect_device(override: Optional[str] = None) -> str:
    """自动检测可用设备，返回 'cuda' / 'mps' / 'cpu'。"""
    if override:
        return override.lower()

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def format_device(device: str) -> str:
    """格式化设备名用于显示。"""
    mapping = {"cuda": "CUDA (GPU)", "mps": "Metal (Apple GPU)", "cpu": "CPU"}
    return mapping.get(device, device)


def resolve_dim(model_default: int, override: Optional[int] = None) -> int:
    """解析向量维度，None 则返回模型默认值。"""
    if override is not None:
        return override
    env = os.environ.get("EMBED_DIM", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return model_default


# ============================================================================
# 本地模型 (BGE-M3 via SentenceTransformer)
# ============================================================================

BGE_M3_DEFAULT_DIM = 1024


class LocalEmbedder(Embedder):
    """使用本地 BGE-M3 模型 (SentenceTransformer)。"""

    LOCAL_MODEL_DIRS = [
        Path(__file__).parent / "bge-m3-model",
        Path(__file__).parent.parent / "bge-m3",
    ]
    FALLBACK_MODEL = "BAAI/bge-m3"

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 dim: Optional[int] = None):
        from sentence_transformers import SentenceTransformer

        resolved = model_path or os.environ.get("LOCAL_MODEL_PATH") or self._resolve_model_path()
        if resolved == self.FALLBACK_MODEL and model_path is None and not os.environ.get("LOCAL_MODEL_PATH"):
            print(f"[embedder] 本地模型未找到，从 HuggingFace 下载 {resolved}...",
                  file=sys.stderr)

        self._device = detect_device(device or os.environ.get("EMBED_DEVICE"))
        self._dim = resolve_dim(BGE_M3_DEFAULT_DIM, dim)

        t0 = time.monotonic()
        print(f"[embedder] 加载本地模型: {resolved}", file=sys.stderr)
        print(f"[embedder]   设备: {format_device(self._device)}", file=sys.stderr)
        print(f"[embedder]   维度: {self._dim}", file=sys.stderr)
        self._model = SentenceTransformer(resolved, device=self._device)
        print(f"[embedder] 模型就绪 ({time.monotonic() - t0:.1f}s)", file=sys.stderr)

    @classmethod
    def _resolve_model_path(cls) -> str:
        for d in cls.LOCAL_MODEL_DIRS:
            if d.exists():
                return str(d)
        return cls.FALLBACK_MODEL

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            truncate_dim=self._dim,
        )
        return embeddings.tolist()


# ============================================================================
# OpenAI 兼容 API
# ============================================================================

OPENAI_DEFAULT_DIM = 1536  # text-embedding-3-small 默认维度


class OpenAIEmbedder(Embedder):
    """使用 OpenAI 兼容 API 作为嵌入引擎。"""

    _API_BATCH_SIZE = 2000

    def __init__(self, dim: Optional[int] = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请在 .env 文件中配置或导出环境变量。"
            )

        try:
            import openai
        except ImportError:
            raise ImportError("缺少 openai 包。请执行: pip install openai>=1.0.0")

        base_url = _ensure_scheme(os.environ.get("OPENAI_BASE_URL"))
        self._model_name = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # 解析维度
        self._dim = resolve_dim(OPENAI_DEFAULT_DIM, dim)
        # text-embedding-3-large 默认 3072
        if "large" in self._model_name and self._dim == OPENAI_DEFAULT_DIM:
            self._dim = resolve_dim(3072, dim)

        info = f"[embedder] OpenAI 嵌入: model={self._model_name}"
        if base_url:
            info += f", base_url={base_url}"
        print(info, file=sys.stderr)
        print(f"[embedder]   维度: {self._dim}", file=sys.stderr)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        chunk_size = min(batch_size, self._API_BATCH_SIZE)
        results: List[List[float]] = []
        total_tokens = 0

        kwargs = dict(model=self._model_name)
        # 只有 text-embedding-3 系列支持 dimensions 参数
        if self._dim and self._dim != OPENAI_DEFAULT_DIM:
            kwargs["dimensions"] = self._dim

        for start in range(0, len(texts), chunk_size):
            chunk = texts[start:start + chunk_size]
            try:
                resp = self._client.embeddings.create(
                    input=chunk, **kwargs,
                )
                total_tokens += resp.usage.total_tokens if resp.usage else 0
                results.extend([d.embedding for d in resp.data])
            except Exception as e:
                raise RuntimeError(
                    f"OpenAI embedding 请求失败 (批次 {start}~{start + len(chunk)}): {e}"
                )

        if total_tokens:
            print(f"[embedder] token 消耗: {total_tokens}", file=sys.stderr)

        return results


# ============================================================================
# 单例管理
# ============================================================================

_embedder: Optional[Embedder] = None
_lock = threading.Lock()


def get_embedder(provider: Optional[str] = None,
                 device: Optional[str] = None,
                 dim: Optional[int] = None) -> Embedder:
    """
    获取嵌入引擎实例（单例）。

    参数:
        provider: "local" | "openai" | None
        device:   "cuda" | "mps" | "cpu" (仅 local，None 则自动检测或询问)
        dim:      向量维度 (None 则读取 EMBED_DIM 环境变量或询问)

    provider 解析优先级:
      1. 显式参数
      2. 环境变量 EMBED_PROVIDER
      3. 交互式选择（仅 TTY）或 fallback "local"
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _lock:
        if _embedder is not None:
            return _embedder
        provider = _resolve_provider(provider, dim)
        device = _resolve_device(device, provider)
        dim = _resolve_dim(dim, provider)
        _embedder = _create_embedder(provider, device, dim)
        return _embedder


def get_provider_name() -> str:
    """返回当前已解析的 provider 名称（用于显示）。"""
    e = get_embedder()
    if isinstance(e, LocalEmbedder):
        return f"local (BGE-M3, dim={e.dim})"
    if isinstance(e, OpenAIEmbedder):
        return f"openai ({e._model_name}, dim={e.dim})"
    return type(e).__name__


def reset_embedder() -> None:
    """重置单例（主要用于测试）。"""
    global _embedder
    with _lock:
        _embedder = None


# ============================================================================
# 内部辅助
# ============================================================================

_VALID_PROVIDERS = {"local", "openai"}


def _resolve_provider(provider: Optional[str], dim_for_prompt: Optional[int] = None) -> str:
    """解析 provider。"""
    if provider is not None:
        p = provider.lower()
        if p not in _VALID_PROVIDERS:
            raise ValueError(f"未知嵌入提供商: '{provider}'。可用选项: local, openai")
        return p

    env = os.environ.get("EMBED_PROVIDER", "").strip().lower()
    if env in _VALID_PROVIDERS:
        return env

    if sys.stdin.isatty():
        return _prompt_provider(dim_for_prompt)
    else:
        print("[embedder] 未检测到终端，默认使用本地模型 (BGE-M3)", file=sys.stderr)
        print("[embedder] 可通过 .env 或 EMBED_PROVIDER 环境变量配置", file=sys.stderr)
        return "local"


def _resolve_device(device: Optional[str], provider: str) -> Optional[str]:
    """解析设备（仅 local 有效）。"""
    if provider != "local":
        return None
    if device is not None:
        return device
    env = os.environ.get("EMBED_DEVICE", "").strip().lower()
    if env:
        return env
    # 没有配置 → TTY 则询问，否则自动检测
    if sys.stdin.isatty():
        return _prompt_device()
    return detect_device()


def _resolve_dim(dim: Optional[int], provider: str) -> Optional[int]:
    """解析向量维度。"""
    if dim is not None:
        return dim
    env = os.environ.get("EMBED_DIM", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    # 没有配置 → TTY 则询问
    if sys.stdin.isatty():
        return _prompt_dim(provider)
    return None


def _prompt_provider(dim_for_prompt: Optional[int] = None) -> str:
    """终端交互选择 provider。"""
    print()
    print("=" * 50)
    print("  选择嵌入引擎 (Embedding Provider)")
    print("=" * 50)
    print("  1) 本地模型 (BGE-M3) — 免费，首次需下载 ~2GB 模型文件")
    print("  2) OpenAI API     — 在线，每次按 token 付费")
    print("  选择后可通过 .env 或 --provider 参数预设，不再弹出此菜单")
    print("=" * 50)

    while True:
        try:
            choice = input("  请选择 (1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"

        if choice == "1":
            print("  → 本地模型 (BGE-M3)\n")
            return "local"
        elif choice == "2":
            print("  → OpenAI API\n")
            return "openai"
        else:
            print("  ⚠ 请输入 1 或 2")


def _prompt_device() -> str:
    """终端交互选择计算设备。"""
    avail = {"cpu": "CPU"}
    try:
        import torch
        if torch.cuda.is_available():
            avail["cuda"] = "CUDA (NVIDIA GPU)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            avail["mps"] = "MPS (Apple Silicon GPU)"
    except ImportError:
        pass

    if len(avail) == 1:
        return "cpu"

    print()
    print("=" * 50)
    print("  选择计算设备 (Device)")
    print("=" * 50)
    opts = list(avail.items())
    for idx, (key, label) in enumerate(opts, 1):
        print(f"  {idx}) {label}")
    print("  选择后可通过 .env 的 EMBED_DEVICE 或 --device 参数预设")
    print("=" * 50)

    while True:
        try:
            choice = input(f"  请选择 (1-{len(opts)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(opts):
                selected = opts[idx][0]
                print(f"  → {avail[selected]}\n")
                return selected
        except ValueError:
            pass
        print(f"  ⚠ 请输入 1-{len(opts)}")


def _prompt_dim(provider: str) -> Optional[int]:
    """终端交互选择向量维度。"""
    if provider == "openai":
        defaults = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        default_dim = defaults.get(model, 1536)
        hint = f"默认 {default_dim}（完整维度），也可设为更小的值（如 256、512）"
    else:
        default_dim = BGE_M3_DEFAULT_DIM
        hint = f"BGE-M3 支持截断维度: 1024（完整）、768、512、256"

    print()
    print("=" * 50)
    print("  设置向量维度 (Embedding Dimension)")
    print("=" * 50)
    print(f"  {hint}")
    print("  选择后可通过 .env 的 EMBED_DIM 或 --dim 参数预设")
    print(f"  直接回车 = {default_dim}")
    print("=" * 50)

    while True:
        try:
            choice = input(f"  维度 (回车={default_dim}): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        if choice == "":
            print(f"  → 使用默认维度: {default_dim}\n")
            return default_dim

        try:
            v = int(choice)
            if v > 0:
                print(f"  → 维度: {v}\n")
                return v
        except ValueError:
            pass
        print(f"  ⚠ 请输入正整数或直接回车")


def _create_embedder(provider: str, device: Optional[str] = None,
                     dim: Optional[int] = None) -> Embedder:
    """根据 provider 创建嵌入引擎实例。"""
    if provider == "local":
        return LocalEmbedder(device=device, dim=dim)
    elif provider == "openai":
        return OpenAIEmbedder(dim=dim)
    else:
        raise ValueError(f"未知嵌入提供商: {provider}")


# ============================================================================
# 快捷入口
# ============================================================================

if __name__ == "__main__":
    print(f"Provider: {get_provider_name()}")
    e = get_embedder()
    vec = e.encode(["Hello world"])
    print(f"向量维度: {len(vec[0])}")
    print(f"向量前5维: {vec[0][:5]}")
