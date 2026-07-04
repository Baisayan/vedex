from __future__ import annotations

from json import dumps, loads
from pathlib import Path

from coding.paths import VedexPaths


class CredentialStoreError(ValueError):
    """Raised when Vedex credential storage cannot be read or written."""


class FileCredentialStore:  
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or credentials_path()

    def get(self, name: str) -> str | None:
        return self._load().get(name)

    def set(self, name: str, value: str) -> None:
        name = _validate_credential_name(name)
        value = value.strip()
        if not value:
            raise CredentialStoreError("Credential value must not be empty")
        data = self._load()
        data[name] = value
        self._save(data)

    def set_api_key(self, name: str, value: str) -> None:
        self.set(name, value)

    def delete(self, name: str) -> None:
        data = self._load()
        data.pop(name, None)
        self._save(data)

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        raw = loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CredentialStoreError("Credentials must be a JSON object")
        credentials: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise CredentialStoreError("Credential names must be strings")
            if not isinstance(value, str):
                raise CredentialStoreError("Credential values must be strings")
            credentials[key] = value
        return credentials

    def _save(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.path.chmod(0o600)


def credentials_path(paths: VedexPaths | None = None) -> Path:
    return (paths or VedexPaths()).home / "credentials.json"


def _validate_credential_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise CredentialStoreError("Credential name must not be empty")
    return normalized
