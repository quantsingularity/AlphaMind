"""
Generic async repository base class.

Concrete repositories inherit from ``BaseRepository[ModelT]`` and get
``get``, ``list``, ``create``, ``update``, ``delete``, and ``count`` for free.
Domain-specific queries are added as methods on the subclass.
"""

from __future__ import annotations

from typing import Any, Generic, Sequence, Type, TypeVar

from db.base import Base
from sqlalchemy import func, select  # BUG-7 fixed: func imported at module level
from sqlalchemy.ext.asyncio import AsyncSession

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Async CRUD repository for a single SQLAlchemy model."""

    def __init__(self, model: Type[ModelT], session: AsyncSession) -> None:
        self.model = model
        self.session = session

    async def get(self, pk: Any) -> ModelT | None:
        """Return a single record by primary key, or ``None`` if not found."""
        return await self.session.get(self.model, pk)

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> Sequence[ModelT]:
        """
        Return a page of records, optionally filtered by exact column equality.

        Example::

            await repo.list(limit=10, status="open", portfolio_id="port-001")
        """
        stmt = select(self.model)
        for col, val in filters.items():
            stmt = stmt.where(getattr(self.model, col) == val)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create(self, **kwargs: Any) -> ModelT:
        """Instantiate, add, flush, and return a new record."""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def update(self, pk: Any, **kwargs: Any) -> ModelT | None:
        """Update fields on an existing record; return ``None`` if not found."""
        instance = await self.get(pk)
        if instance is None:
            return None
        for key, value in kwargs.items():
            setattr(instance, key, value)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, pk: Any) -> bool:
        """Delete a record by primary key. Returns ``True`` if deleted."""
        instance = await self.get(pk)
        if instance is None:
            return False
        await self.session.delete(instance)
        await self.session.flush()
        return True

    async def count(self, **filters: Any) -> int:
        """Return the count of records matching optional filters."""
        # func is now imported at module level (BUG-7 fix)
        stmt = select(func.count()).select_from(self.model)
        for col, val in filters.items():
            stmt = stmt.where(getattr(self.model, col) == val)
        result = await self.session.execute(stmt)
        return result.scalar_one()
