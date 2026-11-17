"""In-memory stand-ins for the Rust auth repositories, patched in through ``rust_database.repository``.

Each fake keeps separate reader and writer rows (a writer read sees ``writer`` when it is set, else ``rows``),
logs every lookup, and raises a row that is an exception. Threads porting more auth objects add their fake as
another field of ``RustAuthStores``.
"""

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest
from pydantic import BaseModel

from litellm.proxy._types import LiteLLM_VerificationTokenView
from litellm.repositories import rust_database
from litellm.rust_bridge._native import (
    RustAccessGroupRepository,
    RustJwtKeyMappingRepository,
    RustKeyRepository,
    RustModelAccessGroupRepository,
    RustObjectPermissionRepository,
    RustProjectRepository,
    RustTagRepository,
)


@dataclass
class FakeKeyStore:
    """``combined_view`` rows by hash, written the way the Prisma row was, shaped through the view model."""

    rows: dict[str, Mapping[str, object] | Exception] = field(default_factory=dict)
    writer: dict[str, Mapping[str, object] | Exception] | None = None
    lookups: list[tuple[str, bool]] = field(default_factory=list)

    async def find(self, token: str, *, use_writer: bool = False) -> str | None:
        self.lookups.append((token, use_writer))
        source = self.writer if use_writer and self.writer is not None else self.rows
        row = source.get(token)
        if isinstance(row, Exception):
            raise row
        return None if row is None else LiteLLM_VerificationTokenView(**row).model_dump_json()


@dataclass(frozen=True)
class JwtMapping:
    jwt_claim_name: str
    jwt_claim_value: str
    token: str
    jwt_issuer: str = ""
    is_active: bool = True


@dataclass
class FakeJwtKeyMappingStore:
    mappings: list[JwtMapping] = field(default_factory=list)
    error: Exception | None = None
    lookups: list[tuple[str, str, str | None]] = field(default_factory=list)

    async def mapped_token(
        self, jwt_claim_name: str, jwt_claim_value: str, jwt_issuer: str | None = None
    ) -> str | None:
        self.lookups.append((jwt_claim_name, jwt_claim_value, jwt_issuer))
        if self.error is not None:
            raise self.error
        return next(
            (
                m.token
                for m in self.mappings
                if m.is_active
                and (m.jwt_claim_name, m.jwt_claim_value, m.jwt_issuer)
                == (jwt_claim_name, jwt_claim_value, jwt_issuer or "")
            ),
            None,
        )

    async def mapped_claims(self, tokens: Sequence[str]) -> list[tuple[str, str, str]]:
        if self.error is not None:
            raise self.error
        return [(m.jwt_issuer, m.jwt_claim_name, m.jwt_claim_value) for m in self.mappings if m.token in tokens]


@dataclass(frozen=True)
class GroupBudget:
    spend: float = 0.0
    max_budget: float | None = None
    has_budget_windows: bool = False


@dataclass
class FakeModelAccessGroupStore:
    groups: dict[str, GroupBudget] = field(default_factory=dict)
    error: Exception | None = None
    budget_lookups: list[tuple[str, ...]] = field(default_factory=list)

    async def names(self, limit: int) -> list[str]:
        if self.error is not None:
            raise self.error
        return list(self.groups)[:limit]

    async def budgets(self, names: Sequence[str]) -> list[tuple[str, float, float | None, bool]]:
        self.budget_lookups.append(tuple(names))
        if self.error is not None:
            raise self.error
        return [
            (name, group.spend, group.max_budget, group.has_budget_windows)
            for name, group in self.groups.items()
            if name in names
        ]


Row = BaseModel | BaseException


@dataclass
class RustStoreRows:
    """Rows one fake store serves, with the reader and writer sides apart like the Prisma ``db`` and
    ``writer_db`` clients."""

    reader: MutableMapping[str, Row] = field(default_factory=dict)
    writer: MutableMapping[str, Row] = field(default_factory=dict)
    lookups: list[tuple[str, bool]] = field(default_factory=list)

    def seed(self, key: str, row: Row) -> None:
        """Serve ``row`` for ``key`` on both sides."""
        self.reader[key] = row
        self.writer[key] = row

    def side(self, use_writer: bool) -> MutableMapping[str, Row]:
        return self.writer if use_writer else self.reader


@dataclass
class TagRows(RustStoreRows):
    """Tag rows, which are read from the reader only. ``batches`` logs each ``find_many`` call and
    ``name_listings`` each registry listing's limit; ``names_error`` makes the listing fail."""

    batches: list[tuple[str, ...]] = field(default_factory=list)
    name_listings: list[int] = field(default_factory=list)
    names_error: BaseException | None = None


def _encoded(row: Row | None) -> str | None:
    if isinstance(row, BaseException):
        raise row
    return None if row is None else row.model_dump_json()


class FakeFindStore:
    def __init__(self, rows: RustStoreRows) -> None:
        self._rows: Final = rows

    async def find(self, key: str, *, use_writer: bool = False) -> str | None:
        self._rows.lookups.append((key, use_writer))
        return _encoded(self._rows.side(use_writer).get(key))


class FakeProjectStore:
    def __init__(self, rows: RustStoreRows) -> None:
        self._rows: Final = rows

    async def find(self, project_id: str) -> str | None:
        self._rows.lookups.append((project_id, False))
        return _encoded(self._rows.reader.get(project_id))


class FakeTagStore:
    def __init__(self, rows: TagRows) -> None:
        self._rows: Final = rows

    async def find_many(self, tag_names: Iterable[str]) -> list[str]:
        names: Final = tuple(tag_names)
        self._rows.batches.append(names)
        return [encoded for name in names if (encoded := _encoded(self._rows.reader.get(name))) is not None]

    async def names(self, limit: int) -> list[str]:
        self._rows.name_listings.append(limit)
        if self._rows.names_error is not None:
            raise self._rows.names_error
        return list(self._rows.reader)[:limit]


@dataclass
class RustAuthStores:
    key: FakeKeyStore = field(default_factory=FakeKeyStore)
    jwt_key_mapping: FakeJwtKeyMappingStore = field(default_factory=FakeJwtKeyMappingStore)
    model_access_group: FakeModelAccessGroupStore = field(default_factory=FakeModelAccessGroupStore)
    project: RustStoreRows = field(default_factory=RustStoreRows)
    tag: TagRows = field(default_factory=TagRows)
    access_group: RustStoreRows = field(default_factory=RustStoreRows)
    object_permission: RustStoreRows = field(default_factory=RustStoreRows)

    def by_kind(self) -> Mapping[Callable[..., object], object]:
        return {
            RustKeyRepository: self.key,
            RustJwtKeyMappingRepository: self.jwt_key_mapping,
            RustModelAccessGroupRepository: self.model_access_group,
            RustProjectRepository: FakeProjectStore(self.project),
            RustTagRepository: FakeTagStore(self.tag),
            RustAccessGroupRepository: FakeFindStore(self.access_group),
            RustObjectPermissionRepository: FakeFindStore(self.object_permission),
        }


def install_rust_auth_stores(monkeypatch: pytest.MonkeyPatch, stores: RustAuthStores | None = None) -> RustAuthStores:
    """Answers `rust_database.repository` with these fakes; a kind with no fake answers None, so "No db connected" still fires."""
    installed = stores or RustAuthStores()
    monkeypatch.setattr(rust_database, "repository", lambda kind: installed.by_kind().get(kind))
    return installed


@pytest.fixture
def rust_auth_stores(monkeypatch: pytest.MonkeyPatch) -> RustAuthStores:
    return install_rust_auth_stores(monkeypatch)
