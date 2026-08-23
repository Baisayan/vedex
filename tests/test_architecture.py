from __future__ import annotations

import ast
from pathlib import Path

import vedex

PACKAGE_ROOT = Path(vedex.__file__).resolve().parent


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports_name(path: Path, *, module: str, name: str) -> bool:
    for node in ast.walk(_tree(path)):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and any(alias.name == name for alias in node.names)
        ):
            return True
    return False


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.partition(".")[0])
    return roots


def test_legacy_runtime_modules_and_symbols_stay_removed() -> None:
    assert not (PACKAGE_ROOT / "core.py").exists()
    assert not (PACKAGE_ROOT / "session.py").exists()

    forbidden_symbols = {"SessionStore", "run_agent_loop"}
    found: set[str] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Name) and node.id in forbidden_symbols:
                found.add(node.id)
            elif (
                isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name in forbidden_symbols
            ):
                found.add(node.name)

    assert found == set()


def test_app_runtime_is_the_only_agent_composition_root() -> None:
    importers = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
        if _imports_name(path, module="agent", name="Agent")
    }

    assert importers == {"runtime.py"}
    assert _imports_name(PACKAGE_ROOT / "cli.py", module="runtime", name="AppRuntime")
    assert _imports_name(PACKAGE_ROOT / "headless.py", module="runtime", name="AppRuntime")


def test_tool_modules_do_not_import_filesystem_or_process_implementations() -> None:
    forbidden_imports = {"os", "pathlib", "subprocess"}
    for name in ("base.py", "read.py", "write.py", "edit.py", "bash.py"):
        path = PACKAGE_ROOT / "tools" / name
        assert _import_roots(path).isdisjoint(forbidden_imports), name
        direct_open_calls = [
            node
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
        ]
        assert direct_open_calls == [], name


def test_provider_sdks_stay_inside_their_adapter_modules() -> None:
    sdk_importers: dict[str, set[str]] = {"openai": set(), "google": set()}
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative_path = path.relative_to(PACKAGE_ROOT).as_posix()
        roots = _import_roots(path)
        for sdk_root in sdk_importers:
            if sdk_root in roots:
                sdk_importers[sdk_root].add(relative_path)

    assert sdk_importers == {
        "openai": {"models/openai.py"},
        "google": {"models/gemini.py"},
    }
