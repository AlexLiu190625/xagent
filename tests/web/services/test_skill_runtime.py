from types import SimpleNamespace

import pytest
from sqlalchemy import Column, Integer, create_engine, select
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.skills.library import SkillScopeContext
from xagent.web.services.skill_runtime import (
    SkillRuntimeSessionBoundaryError,
    build_runtime_skill_scope,
    get_skill_runtime_scope,
)

Base = declarative_base()


class _Item(Base):
    __tablename__ = "skill_runtime_items"

    id = Column(Integer, primary_key=True)


def _one_slot_session() -> tuple[Session, QueuePool]:
    engine = create_engine(
        "sqlite://",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    return db, engine.pool


def test_build_runtime_skill_scope_releases_clean_caller_connection() -> None:
    db, pool = _one_slot_session()
    try:
        db.scalars(select(_Item)).all()
        assert pool.checkedout() == 1

        context = build_runtime_skill_scope(
            user_id=7,
            metadata={"team_id": 11},
            caller_db=db,
        )

        assert context == SkillScopeContext(
            user_id=7,
            metadata={"team_id": 11},
        )
        assert pool.checkedout() == 0
    finally:
        db.close()


def test_build_runtime_skill_scope_fails_closed_on_pending_write() -> None:
    db, pool = _one_slot_session()
    try:
        pending = _Item()
        db.add(pending)

        with pytest.raises(
            SkillRuntimeSessionBoundaryError,
            match="pending writes",
        ):
            build_runtime_skill_scope(
                user_id=7,
                caller_db=db,
            )

        assert pending in db.new
        assert pool.checkedout() == 0
    finally:
        db.rollback()
        db.close()


def test_skill_runtime_dependency_detaches_identity_before_route_execution() -> None:
    db, pool = _one_slot_session()
    try:
        db.scalars(select(_Item)).all()
        user = SimpleNamespace(id=7, _saas_team_id=11)
        assert pool.checkedout() == 1

        context = get_skill_runtime_scope(current_user=user, db=db)

        assert context == SkillScopeContext(
            user_id=7,
            metadata={"team_id": 11},
        )
        assert pool.checkedout() == 0
    finally:
        db.close()
