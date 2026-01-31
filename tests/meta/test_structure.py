"""Meta tests that enforce documentation of source files."""

from pathlib import Path

import pytest


def test_src_files_are_documented() -> None:
    """Ensure every Python file in src/ is listed in specs/00_repo_map.md.

    Raises:
        Failed: If specs/00_repo_map.md is missing or any src/ files are not listed.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    specs_path = project_root / "specs" / "00_repo_map.md"
    src_path = project_root / "src"

    if not specs_path.exists():
        pytest.fail(f"Critical spec missing: {specs_path} not found.")

    if not src_path.exists():
        pytest.fail(f"Source directory missing: {src_path} not found.")

    spec_content = specs_path.read_text(encoding="utf-8")
    undocumented_files: list[str] = []

    for file_path in src_path.rglob("*.py"):
        rel_path = file_path.relative_to(project_root).as_posix()
        if rel_path not in spec_content:
            undocumented_files.append(rel_path)

    if undocumented_files:
        error_msg = (
            "\n\n[Documentation Drift Detected]\n"
            f"The following files exist in the codebase but are missing from '{specs_path.name}':\n"
            f"{'-' * 60}\n"
            + "\n".join(f"- {path}" for path in undocumented_files)
            + f"\n{'-' * 60}\n"
            f"Please update {specs_path.name} to include these files."
        )
        pytest.fail(error_msg)