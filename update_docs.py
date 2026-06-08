#!/usr/bin/env python3
"""
下载 Godot 最新稳定版英文文档，更新 _sources/ 目录。
Download the latest Godot stable English documentation to update _sources/.

用法 | Usage:
    python update_docs.py                       # 自动选择最快源并下载
    python update_docs.py --branch main         # 使用开发分支
    python update_docs.py --dry-run             # 预览变更
    python update_docs.py --source ghproxy.com  # 手动指定下载源
    python update_docs.py --list-sources        # 列出可用源
"""

import os
import sys
import json
import time
import zipfile
import argparse
import textwrap
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 配置 | Configuration ──────────────────────────────────────────────

REPO_OWNER = "godotengine"
REPO_NAME = "godot-docs"
GITHUB_BRANCH = "stable"

# 需要映射的子目录（相对于 godot-docs 仓库根目录）
INCLUDE_DIRS = [
    "about", "classes", "community", "contributing",
    "engine_details", "getting_started", "tutorials",
]

EXCLUDE_PATTERNS = ["README.rst", "conf.py", "_templates", "_static"]

# ── 下载源 | Download sources ────────────────────────────────────────
# 每个条目: (名称, 测速URL, GitHub URL → 实际URL 的转换函数)
# Each source: (name, probe_url, url_transform)

RAW_SOURCES = [
    ("GitHub (direct)", "https://github.com",
     lambda url: url),
    ("ghproxy.com", "https://ghproxy.com",
     lambda url: f"https://ghproxy.com/{url}"),
    ("gh-proxy.cn", "https://gh-proxy.cn",
     lambda url: f"https://gh-proxy.cn/{url}"),
    ("edgegone.gh.gh-proxy.com", "https://edgegone.gh.gh-proxy.com",
     lambda url: f"https://edgegone.gh.gh-proxy.com/{url}"),
    ("hk.gh-proxy.com", "https://hk.gh-proxy.com",
     lambda url: f"https://hk.gh-proxy.com/{url}"),
    ("gh-proxy.com", "https://gh-proxy.com",
     lambda url: f"https://gh-proxy.com/{url}"),
    ("gh-llkk.cc", "https://gh-llkk.cc",
     lambda url: f"https://gh-llkk.cc/{url}"),
    ("mirror.ghproxy.com", "https://mirror.ghproxy.com",
     lambda url: f"https://mirror.ghproxy.com/{url}"),
]

PROBE_TIMEOUT = 5       # 每个源的探测超时（秒）
PROBE_CONCURRENCY = 5   # 并行探测数


# ── 源探测 | Source probing ──────────────────────────────────────────


def probe_source(name: str, probe_url: str, timeout: float) -> dict:
    """
    探测一个源的延迟。
    返回 {"name": ..., "latency": float | None, "ok": bool, "error": str | None}。
    """
    start = time.monotonic()
    try:
        req = Request(probe_url, method="GET")
        # 只请求第一个字节，最小化流量
        req.add_header("Range", "bytes=0-0")
        req.add_header("User-Agent", "godot-docs-RAG/1.0")
        resp = urlopen(req, timeout=timeout)
        resp.read()  # 确保连接完成
        resp.close()
        latency = time.monotonic() - start
        return {"name": name, "latency": round(latency, 3), "ok": True, "error": None}
    except Exception as e:
        return {"name": name, "latency": None, "ok": False, "error": str(e)}


def find_fastest_source(
    sources: list,
    timeout: float = PROBE_TIMEOUT,
    verbose: bool = True,
) -> tuple:
    """
    并行探测所有源，返回 (name, transform_func) 最快的源。
    如果所有源都不可达，打印错误并退出。
    """
    if verbose:
        print(f"  ⌛ Probing {len(sources)} download sources ...")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=PROBE_CONCURRENCY) as pool:
        fut_map = {
            pool.submit(probe_source, name, p_url, timeout): name
            for name, p_url, _ in sources
        }
        for fut in as_completed(fut_map):
            res = fut.result()
            results.append(res)
            nm = res["name"]
            if res["ok"]:
                if verbose:
                    print(f"    ✓ {nm}  ({res['latency']:.0f} ms)".replace(" ms", "ms"))
            else:
                if verbose:
                    print(f"    ✗ {nm}  ({res['error']})")

    results.sort(key=lambda r: (not r["ok"], r["latency"] if r["ok"] else float("inf")))

    ok_sources = [r for r in results if r["ok"]]
    if not ok_sources:
        print("  ✗ All download sources are unreachable!", file=sys.stderr)
        print("    Try: --source <url> to use a custom mirror, or check your network.", file=sys.stderr)
        sys.exit(1)

    best = ok_sources[0]
    # 找到对应的 transform
    for name, _, transform in sources:
        if name == best["name"]:
            if verbose:
                print(f"\n  → Using {name}  ({best['latency']:.0f} ms)")
            return (name, transform)

    # fallback: 第一个源
    return (sources[0][0], sources[0][2])


# ── 下载 | Download ──────────────────────────────────────────────────


def download_zip(url: str, dest: Path, source_name: str = "") -> None:
    """下载 ZIP 文件，带进度提示。"""
    tag = f" [{source_name}]" if source_name else ""
    print(f"  ↓ Downloading{tag}")
    print(f"    {url}")
    print(f"    → {dest}")
    try:
        urlretrieve(url, dest)
    except URLError as e:
        print(f"  ✗ Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    except HTTPError as e:
        print(f"  ✗ HTTP {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)
    size = dest.stat().st_size
    print(f"  ✓ Done ({size / 1024 / 1024:.1f} MB)")


# ── 文档处理 | Document processing ───────────────────────────────────


def should_include(rel_path: str) -> bool:
    """判断 rel_path （如 'classes/class_node.rst'）是否应该被纳入。"""
    for pat in EXCLUDE_PATTERNS:
        if pat in rel_path:
            return False
    if not rel_path.endswith(".rst"):
        return False
    if INCLUDE_DIRS:
        return any(rel_path.startswith(d + "/") for d in INCLUDE_DIRS)
    return True


def collect_zip_rsts(zip_path: Path) -> list[tuple[str, str]]:
    """扫描 ZIP 中所有 .rst 文件，返回 [(zip_src_path, relative_path), ...]"""
    results: list[tuple[str, str]] = []
    # 确定 ZIP 内的顶级目录名
    with zipfile.ZipFile(zip_path, "r") as zf:
        top_dirs = set()
        for name in zf.namelist():
            parts = name.split("/")
            if parts[0]:
                top_dirs.add(parts[0])
        prefix = next(iter(top_dirs)) if len(top_dirs) == 1 else ""

        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) < 2:
                continue
            rel = "/".join(parts[1:])
            if rel and should_include(rel):
                results.append((name, rel))
    results.sort()
    return results, prefix


def sync_sources(
    zip_path: Path,
    sources_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> tuple[int, int, int, int]:
    """
    将 ZIP 中的 .rst 文件同步到 sources_dir。
    返回 (新增数, 更新数, 跳过数, 写入字节数)。
    """
    entries, prefix = collect_zip_rsts(zip_path)
    print(f"\n  Found {len(entries)} .rst files in ZIP" + (f" ({prefix}/)" if prefix else ""))

    added = updated = skipped = new_bytes = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for zip_src, rel in entries:
            local_rel = Path(rel).with_suffix(".rst.txt")
            dest = sources_dir / local_rel

            info = zf.getinfo(zip_src)
            src_size = info.file_size

            is_new = not dest.exists()
            is_diff = True

            if not is_new:
                is_diff = dest.stat().st_size != src_size

            if not is_new and not is_diff:
                skipped += 1
                continue

            if dry_run:
                print(f"  {'[+]' if is_new else '[~]'} {local_rel}")
                if is_new:
                    added += 1
                else:
                    updated += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(zip_src))
            new_bytes += len(zip_src)

            if is_new:
                added += 1
            else:
                updated += 1

            if verbose and (added + updated) % 200 == 0:
                print(f"  ... {added + updated}/{len(entries)} processed")

    return added, updated, skipped, new_bytes


def remove_stale(sources_dir: Path, zip_path: Path, dry_run: bool = False) -> int:
    """删除本地存在但上游已不存在的文件。返回删除数。"""
    # 收集 ZIP 中的文件路径
    entries, _ = collect_zip_rsts(zip_path)
    zip_files = set()
    for _, rel in entries:
        zip_files.add(Path(rel).with_suffix(".rst.txt").as_posix())

    removed = 0
    for local_path in sorted(sources_dir.rglob("*.rst.txt")):
        rel = local_path.relative_to(sources_dir).as_posix()
        if rel not in zip_files:
            if dry_run:
                print(f"  [-] {rel}")
            else:
                local_path.unlink()
                print(f"  ✗ Removed stale: {rel}")
            removed += 1

    # 清理空目录
    if not dry_run:
        for dirpath, dirnames, filenames in os.walk(sources_dir, topdown=False):
            if not dirnames and not filenames and dirpath != str(sources_dir):
                os.rmdir(dirpath)

    return removed


# ── 主入口 | Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download & sync Godot documentation sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--branch", "-b",
        default=GITHUB_BRANCH,
        help=f"Git branch to download (default: {GITHUB_BRANCH})",
    )
    parser.add_argument(
        "--zip", "-z",
        type=Path,
        default=None,
        help="Local ZIP file (skip download)",
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        metavar="NAME_OR_URL",
        help="Download source: name (ghproxy.com) or custom URL prefix",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available download sources and exit",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=PROBE_TIMEOUT,
        help=f"Probe timeout per source in seconds (default: {PROBE_TIMEOUT})",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview changes without writing",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded ZIP file after sync",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't remove stale local files",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output",
    )
    args = parser.parse_args()

    # ── 列出可用源 ────────────────────────────────────────────────────
    if args.list_sources:
        print("Available download sources:")
        print()
        for name, probe_url, _ in RAW_SOURCES:
            print(f"  {name}")
            print(f"    Probe: {probe_url}")
        print()
        print("You can also use --source <URL-prefix> for custom mirrors.")
        print("Example:")
        print('  python update_docs.py --source "https://your-proxy.example.com/"')
        return

    # ── 确定路径 ──────────────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    sources_dir = script_dir / "_sources"
    github_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/archive/refs/heads/{args.branch}.zip"

    # ── 下载 ──────────────────────────────────────────────────────────
    zip_path: Path | None = args.zip
    if zip_path is None:
        # 确定下载源
        if args.source:
            # 用户指定了源
            name, transform = resolve_custom_source(args.source, github_url)
        else:
            # 自动探测最快源
            name, transform = find_fastest_source(
                RAW_SOURCES,
                timeout=args.timeout,
                verbose=not args.quiet,
            )

        zip_path = script_dir / f"godot-docs-{args.branch}.zip"
        download_url = transform(github_url)
        download_zip(download_url, zip_path, source_name=name)
    else:
        if not zip_path.exists():
            print(f"  ✗ ZIP not found: {zip_path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Using local ZIP: {zip_path}")

    print(f"  Sources dir: {sources_dir}")

    # ── 同步 ──────────────────────────────────────────────────────────
    added, updated, skipped, new_bytes = sync_sources(
        zip_path, sources_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    # ── 清理陈旧文件 ──────────────────────────────────────────────────
    removed = 0
    if not args.no_cleanup:
        removed = remove_stale(sources_dir, zip_path, dry_run=args.dry_run)

    # ── 清理 ZIP ──────────────────────────────────────────────────────
    if not args.keep_zip and args.zip is None:
        zip_path.unlink(missing_ok=True)

    # ── 汇总 ──────────────────────────────────────────────────────────
    print()
    print("─" * 40)
    action = "Would" if args.dry_run else "Did"
    print(f"  {action} add    {added} new file{'s' if added != 1 else ''}")
    print(f"  {action} update {updated} file{'s' if updated != 1 else ''}")
    print(f"  {action} skip   {skipped} (unchanged)")
    if removed:
        print(f"  {action} remove {removed} stale file{'s' if removed != 1 else ''}")
    if not args.dry_run:
        print(f"  Written {new_bytes / 1024:.0f} KB")
    print("─" * 40)
    total = added + updated + skipped
    print(f"  Total: {total} .rst files in _sources/")
    print(f"  {'✓ Sync complete!' if not args.dry_run else '(dry run — no files written)'}")


def _ensure_scheme(url: str) -> str:
    """如果 URL 没有协议头，自动补 https://。"""
    url = url.strip()
    if url and not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
        url = "https://" + url
    return url.rstrip("/")


def resolve_custom_source(source_arg: str, github_url: str) -> tuple[str, callable]:
    """
    解析用户指定的 --source 参数。
    如果是已知源名称，直接匹配；否则视为 URL 前缀。
    """
    # 先检查是否是已知源名称（大小写不敏感）
    for name, _, transform in RAW_SOURCES:
        if source_arg.lower() == name.lower() or source_arg.lower() == name.lower().split(" ")[0].split("(")[0].strip():
            return name, transform

    # 否则视为自定义 URL 前缀
    raw = _ensure_scheme(source_arg)
    return (f"custom ({raw})", lambda url: f"{raw}/{url}")


if __name__ == "__main__":
    main()
