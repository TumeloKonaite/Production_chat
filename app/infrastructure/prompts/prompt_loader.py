from __future__ import annotations

from pathlib import Path


class UnknownPromptVersionError(ValueError):
    """Raised when a prompt version does not map to a versioned template file."""

    def __init__(self, prompt_version: str, available_versions: list[str]) -> None:
        versions = ", ".join(available_versions) or "none"
        super().__init__(
            f"Unknown prompt version: {prompt_version}. Available versions: {versions}"
        )


class PromptLoader:
    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = prompts_dir

    def load(self, prompt_version: str) -> str:
        normalized_version = prompt_version.strip()
        available_paths = self._available_prompt_paths()
        prompt_path = available_paths.get(normalized_version)
        if prompt_path is None:
            raise UnknownPromptVersionError(normalized_version, sorted(available_paths))

        return prompt_path.read_text(encoding="utf-8").strip()

    def available_versions(self) -> list[str]:
        return sorted(self._available_prompt_paths())

    def _available_prompt_paths(self) -> dict[str, Path]:
        return {
            path.stem: path
            for path in self.prompts_dir.glob("*.md")
            if path.is_file()
        }
