"""Fetch repositories from GitHub and turn their files into LangChain Documents.

Repositories are downloaded as tarballs through the GitHub REST API (no `git`
binary required) and extracted under ``<data_dir>/repos/<owner>/<repo>/``.
The same directory layout is used by the test fixtures, so
:func:`load_documents_from_dir` serves both the real index and the tests.
"""

from __future__ import annotations

import fnmatch
import io
import json
import logging
import shutil
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from langchain_core.documents import Document

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


@dataclass
class SourceSpec:
    """One entry of ``sources:`` in ``config/sources.yaml``: a GitHub user, organisation or single repo."""

    type: str  # user | org | repo
    name: str
    include_forks: bool = False
    exclude: list[str] = field(default_factory=list)


@dataclass
class SourcesConfig:
    """The parsed ``config/sources.yaml``: which accounts to index and which files to keep."""

    sources: list[SourceSpec]
    include_extensions: list[str]
    exclude_globs: list[str]

    @classmethod
    def load(cls, path: Path) -> SourcesConfig:
        """Parse the YAML sources file into a :class:`SourcesConfig`."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sources = [SourceSpec(**s) for s in raw.get("sources", [])]
        return cls(
            sources=sources,
            include_extensions=raw.get("include_extensions", [".py", ".md"]),
            exclude_globs=raw.get("exclude_globs", []),
        )

    def is_excluded(self, owner: str, repo: str) -> bool:
        """True if ``owner/repo`` is listed under ``exclude`` of a matching source (case-insensitive).

        Used both when listing repositories on GitHub and when rebuilding the index from the
        local cache, so an excluded repository never sneaks back in via ``--skip-fetch``.
        """
        for spec in self.sources:
            if spec.type in ("user", "org") and spec.name.lower() == owner.lower():
                if repo.lower() in {e.lower() for e in spec.exclude}:
                    return True
        return False

    def accepts(self, rel_path: str) -> bool:
        """True if a file path inside a repository should be indexed."""
        p = Path(rel_path)
        name = p.name
        ext_ok = p.suffix in self.include_extensions or name in self.include_extensions
        if not ext_ok:
            return False
        norm = rel_path.replace("\\", "/")
        for pattern in self.exclude_globs:
            if fnmatch.fnmatch(norm, pattern) or fnmatch.fnmatch("/" + norm, pattern):
                return False
        return True


@dataclass
class RepoRef:
    """A repository as returned by the GitHub API, reduced to the fields ingestion needs."""

    owner: str
    name: str
    default_branch: str
    html_url: str
    fork: bool = False
    size_kb: int = 0

    @property
    def full_name(self) -> str:
        """The repository name in ``owner/name`` form."""
        return f"{self.owner}/{self.name}"


# --------------------------------------------------------------------------- #
# GitHub API
# --------------------------------------------------------------------------- #


class GitHubClient:
    """Thin wrapper around the GitHub REST API: list repositories and download tarballs."""

    def __init__(self, token: str | None = None, timeout: float = 120.0):
        """Create the HTTP client; a token is optional but raises the rate limit from 60 to 5000 requests/hour."""
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "iu-code-rag"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=GITHUB_API, headers=headers, timeout=timeout, follow_redirects=True)

    def _paginate(self, url: str) -> Iterator[dict]:
        """Yield every item of a paginated list endpoint (100 items per page)."""
        page = 1
        while True:
            resp = self._client.get(url, params={"per_page": 100, "page": page})
            resp.raise_for_status()
            items = resp.json()
            if not items:
                return
            yield from items
            if len(items) < 100:
                return
            page += 1

    def list_repos(self, spec: SourceSpec) -> list[RepoRef]:
        """Resolve a :class:`SourceSpec` to concrete repositories, skipping forks, excluded and empty ones."""
        if spec.type == "repo":
            owner, name = spec.name.split("/", 1)
            data = self._client.get(f"/repos/{owner}/{name}")
            data.raise_for_status()
            items = [data.json()]
        elif spec.type == "user":
            items = list(self._paginate(f"/users/{spec.name}/repos"))
        elif spec.type == "org":
            items = list(self._paginate(f"/orgs/{spec.name}/repos"))
        else:
            raise ValueError(f"unknown source type: {spec.type}")

        refs: list[RepoRef] = []
        excluded = {e.lower() for e in spec.exclude}
        for item in items:
            ref = RepoRef(
                owner=item["owner"]["login"],
                name=item["name"],
                default_branch=item.get("default_branch") or "main",
                html_url=item["html_url"],
                fork=bool(item.get("fork")),
                size_kb=int(item.get("size") or 0),
            )
            if ref.fork and not spec.include_forks:
                log.info("skipping fork %s", ref.full_name)
                continue
            if ref.name.lower() in excluded:
                log.info("skipping excluded %s", ref.full_name)
                continue
            if ref.size_kb == 0:
                log.info("skipping empty repository %s", ref.full_name)
                continue
            refs.append(ref)
        return refs

    def download_repo(self, ref: RepoRef, dest: Path, config: SourcesConfig, max_file_bytes: int) -> int:
        """Download a repo tarball and extract only the accepted text files into *dest*.

        Returns the number of files written.
        """
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        resp = self._client.get(f"/repos/{ref.owner}/{ref.name}/tarball/{ref.default_branch}")
        resp.raise_for_status()
        written = 0
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue
                rel = Path(*parts[1:])  # strip the "<owner>-<repo>-<sha>/" prefix
                rel_str = rel.as_posix()
                if ".." in rel.parts or rel.is_absolute():
                    continue
                if member.size > max_file_bytes or not config.accepts(rel_str):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                if b"\x00" in data:
                    continue  # binary
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written += 1
        meta = {
            "owner": ref.owner,
            "name": ref.name,
            "default_branch": ref.default_branch,
            "html_url": ref.html_url,
            "files": written,
        }
        (dest / ".repo.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return written


# --------------------------------------------------------------------------- #
# Files -> Documents
# --------------------------------------------------------------------------- #

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".ipynb": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".js": "js",
    ".jsx": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".php": "php",
    ".cs": "csharp",
    ".md": "markdown",
    ".rst": "rst",
    ".html": "html",
    ".sql": "sql",
    ".sh": "shell",
    ".r": "r",
    ".R": "r",
}


def notebook_to_text(raw: str) -> str:
    """Flatten a Jupyter notebook into markdown + fenced code cells."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{src}\n```")
        else:
            parts.append(src)
    return "\n\n".join(parts)


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8 text; ``None`` for unreadable or binary files. Notebooks are flattened."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    text = data.decode("utf-8", errors="replace")
    if path.suffix == ".ipynb":
        text = notebook_to_text(text)
    return text


def load_documents_from_repo_dir(
    repo_dir: Path,
    owner: str,
    name: str,
    config: SourcesConfig,
    max_file_bytes: int = 200_000,
    branch: str = "main",
    html_url: str | None = None,
) -> list[Document]:
    """Turn every accepted file below ``repo_dir`` into a LangChain ``Document``.

    Metadata: ``source`` (owner/repo/path), ``owner``, ``repo``, ``path``, ``language`` and the
    GitHub ``url`` of the file on ``branch``.
    """
    docs: list[Document] = []
    html_url = html_url or f"https://github.com/{owner}/{name}"
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file() or path.name == ".repo.json":
            continue
        rel = path.relative_to(repo_dir).as_posix()
        if not config.accepts(rel):
            continue
        if path.stat().st_size > max_file_bytes:
            continue
        text = _read_text(path)
        if not text or not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"{owner}/{name}/{rel}",
                    "owner": owner,
                    "repo": name,
                    "path": rel,
                    "language": LANGUAGE_BY_EXT.get(path.suffix, "text"),
                    "url": f"{html_url}/blob/{branch}/{rel}",
                },
            )
        )
    return docs


def load_documents_from_dir(root: Path, config: SourcesConfig, max_file_bytes: int = 200_000) -> list[Document]:
    """Load every ``<root>/<owner>/<repo>/**`` file as Documents (skipping excluded repositories)."""
    root = Path(root)
    docs: list[Document] = []
    for owner_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for repo_dir in sorted(p for p in owner_dir.iterdir() if p.is_dir()):
            if config.is_excluded(owner_dir.name, repo_dir.name):
                log.info("skipping excluded cached repository %s/%s", owner_dir.name, repo_dir.name)
                continue
            meta_file = repo_dir / ".repo.json"
            branch, html_url = "main", None
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                branch = meta.get("default_branch", branch)
                html_url = meta.get("html_url")
            docs.extend(
                load_documents_from_repo_dir(
                    repo_dir, owner_dir.name, repo_dir.name, config, max_file_bytes, branch, html_url
                )
            )
    return docs


def iter_repo_dirs(root: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield ``(owner, repo, directory)`` for every cached repository below ``root``."""
    for owner_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for repo_dir in sorted(p for p in owner_dir.iterdir() if p.is_dir()):
            yield owner_dir.name, repo_dir.name, repo_dir
