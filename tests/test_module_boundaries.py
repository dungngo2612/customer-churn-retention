from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "churn_retention" / "modules"


def _target_module(imported: str, current_module: str) -> str | None:
    prefix = ["churn_retention", "modules"]
    parts = imported.split(".")
    if parts[:2] == prefix and len(parts) > 2:
        return parts[2]
    if imported == "churn_retention.modules":
        return None
    if imported.startswith("."):
        level = len(imported) - len(imported.lstrip("."))
        suffix = imported[level:].split(".") if imported[level:] else []
        current_parts = ["churn_retention", "modules", current_module]
        base = current_parts[: len(current_parts) - level + 1]
        resolved = base + suffix
        if resolved[:2] == prefix and len(resolved) > 2:
            return resolved[2]
    return None


def _cross_module_imports() -> list[str]:
    violations: list[str] = []
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = source_file.relative_to(SOURCE_ROOT)
        current_module = relative.parts[0]
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_name = "." * node.level + (node.module or "")
                target = _target_module(module_name, current_module)
                if target is not None and target != current_module:
                    imported_names.append(module_name)
                elif node.module == "churn_retention.modules":
                    imported_names.extend(
                        f"churn_retention.modules.{alias.name}" for alias in node.names
                    )

            for imported in imported_names:
                target = _target_module(imported, current_module)
                if target is not None and target != current_module:
                    violations.append(
                        f"{source_file.relative_to(SOURCE_ROOT.parent.parent)}:"
                        f"{node.lineno} imports module '{target}'"
                    )
    return violations


def test_modules_do_not_import_other_modules() -> None:
    violations = _cross_module_imports()
    assert not violations, (
        "A module must not import another module's internals. "
        "Move shared contracts to churn_retention.contracts or introduce a port.\n"
        + "\n".join(violations)
    )
