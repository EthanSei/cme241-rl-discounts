import pytest
from pathlib import Path

# Config: Files to ignore in the check
IGNORED_FILES = {"__init__.py"}

def test_src_files_are_documented():
    """
    Consistency Check:
    Ensures that every Python file in src/ is explicitly listed in specs/repo_structure.md.
    This prevents 'Shadow IT'—code that exists but isn't mapped for the agent.
    """
    # 1. Locate Project Root and Critical Files
    # Assumes structure: project_root/tests/meta/test_structure.py
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2] 
    
    specs_path = project_root / "specs" / "repo_structure.md"
    src_path = project_root / "src"

    # Sanity checks
    if not specs_path.exists():
        pytest.fail(f"Critical Spec missing: {specs_path} not found.")
    
    if not src_path.exists():
        pytest.fail(f"Source directory missing: {src_path} not found.")

    # 2. Load the Documentation Map
    spec_content = specs_path.read_text(encoding="utf-8")

    # 3. Walk the Source Tree
    undocumented_files = []
    
    # rglob('*') recursively finds all files
    for file_path in src_path.rglob("*.py"):
        if file_path.name in IGNORED_FILES:
            continue
            
        # Get relative path from project root (e.g., "src/pricing_engine/agents/dp_agent.py")
        # We normalize to POSIX style (forward slashes) to match Markdown conventions
        rel_path = file_path.relative_to(project_root).as_posix()
        
        # 4. Check against the Spec
        # We check if the filename exists. For stricter checking, check the full rel_path.
        # Checking just the filename is usually sufficient and less brittle to minor folder moves.
        if file_path.name not in spec_content:
             undocumented_files.append(rel_path)

    # 5. Fail if Drift Detected
    if undocumented_files:
        error_msg = (
            f"\n\n[Documentation Drift Detected]\n"
            f"The following files exist in the codebase but are missing from '{specs_path.name}':\n"
            f"{'-' * 60}\n" + 
            "\n".join(f"- {f}" for f in undocumented_files) +
            f"\n{'-' * 60}\n"
            f"Please update {specs_path.name} to include these files."
        )
        pytest.fail(error_msg)