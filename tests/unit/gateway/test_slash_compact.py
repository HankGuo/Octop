"""Tests for /compact summarization slash command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from octop_harness.compaction import CompactResult

from octop.config import OctopConfig
from octop.infra.agents.manager import AgentManager
from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.repos.agents import AgentRepo
from octop.infra.db.repos.sessions import SessionRepo
from octop.infra.db.repos.threads import ThreadRepo
from octop.infra.db.repos.users import UserRepo
from octop.infra.db.services import build_shared_services
from octop.infra.gateway.slash import BufferSink, SlashCommand, build_default_dispatcher
from octop.infra.gateway.slash.ctx import SlashCtx
from octop.infra.gateway.slash.handlers.composite import _short_offload_path, cmd_compact
from octop.infra.gateway.threads import ThreadRegistry
from octop.infra.utils.paths import PathLayout


def _agent_manager(tmp_path: Path, db: SqlitePool) -> AgentManager:
    services = build_shared_services(db=db, paths=PathLayout(tmp_path), config=OctopConfig())
    manager = AgentManager(repos=services.repos, paths=services.paths)
    harness_mgr = MagicMock()
    harness_mgr.get_thread_model.return_value = None
    manager._harness_manager = harness_mgr
    return manager


@pytest.fixture
def ctx(tmp_path: Path) -> SlashCtx:
    db = SqlitePool(tmp_path / "x.db")
    run_migrations(db)
    UserRepo(db).create(username="u", password_hash="h", role="user")
    agent_repo = AgentRepo(db)
    agent_repo.create(agent_id="a1", user_id=1, name="bot")
    registry = ThreadRegistry(session_repo=SessionRepo(db), thread_repo=ThreadRepo(db))
    sk = ThreadRegistry.make_key(agent_id="a1", channel_type="ui", channel_subject_id="1")
    return SlashCtx(
        agent_id="a1",
        user_id=1,
        channel_type="ui",
        session_key=sk,
        thread_registry=registry,
        agent_repo=agent_repo,
        agent_manager=_agent_manager(tmp_path, db),
    )


@pytest.fixture
def dispatcher():
    return build_default_dispatcher()


@pytest.mark.asyncio
async def test_cmd_compact_calls_harness_without_reset(ctx, dispatcher) -> None:
    await ctx.thread_registry.get_or_create_by_key(
        session_key=ctx.session_key,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        channel_type=ctx.channel_type,
    )
    tid = ctx.thread_registry.get_bound_thread_id(ctx.session_key)
    assert tid

    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(
            ok=True,
            summarized_count=12,
            preserved_count=4,
            file_path=f"/ws/conversation_history/{tid}.md",
            reason="ok",
        )
    )
    ctx.agent_manager.get_agent = MagicMock(return_value=harness)
    before = len(ctx.thread_registry.list_threads(agent_id=ctx.agent_id, user_id=ctx.user_id))

    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), ctx, sink)

    harness.acompact_conversation.assert_awaited_once()
    assert harness.acompact_conversation.await_args.args[0] == tid
    assert (
        len(ctx.thread_registry.list_threads(agent_id=ctx.agent_id, user_id=ctx.user_id)) == before
    )
    assert ctx.thread_registry.get_bound_thread_id(ctx.session_key) == tid
    text = "\n".join(sink.lines)
    assert "12" in text
    assert "conversation_history" in text
    assert "/home/" not in text


@pytest.mark.asyncio
async def test_cmd_compact_shows_octop_relative_offload_path(ctx, dispatcher) -> None:
    await ctx.thread_registry.get_or_create_by_key(
        session_key=ctx.session_key,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        channel_type=ctx.channel_type,
    )
    tid = ctx.thread_registry.get_bound_thread_id(ctx.session_key)
    assert tid

    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(
            ok=True,
            summarized_count=3,
            preserved_count=4,
            file_path=f"/Users/wally/.octop/workspaces/X/.octop/conversation_history/{tid}.md",
            reason="ok",
        )
    )
    ctx.agent_manager.get_agent = MagicMock(return_value=harness)

    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), ctx, sink)

    text = "\n".join(sink.lines)
    assert ".octop/conversation_history/" in text
    assert "/Users/wally" not in text
    assert "/home/" not in text


def test_short_offload_path_hides_windows_macos_and_linux_hosts() -> None:
    tid = "thr_win"
    cases = (
        rf"C:\Users\wally\.octop\workspaces\X\.octop\conversation_history\{tid}.md",
        rf"\\?\C:\Users\wally\.octop\workspaces\X\.octop\conversation_history\{tid}.md",
        f"/home/wally/.octop/workspaces/X/.octop/conversation_history/{tid}.md",
        f"/Users/wally/.octop/workspaces/X/.octop/conversation_history/{tid}.md",
        f"C:/Users/wally/.octop/conversation_history/{tid}.md",
    )
    for raw in cases:
        got = _short_offload_path(raw)
        assert got == f".octop/conversation_history/{tid}.md"
        assert "\\" not in got
        assert "C:" not in got
        assert "/Users/" not in got
        assert "/home/" not in got
    assert (
        _short_offload_path(rf"D:\data\conversation_history\{tid}.md")
        == f"conversation_history/{tid}.md"
    )


@pytest.mark.asyncio
async def test_cmd_compact_shows_windows_offload_path(ctx, dispatcher) -> None:
    await ctx.thread_registry.get_or_create_by_key(
        session_key=ctx.session_key,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        channel_type=ctx.channel_type,
    )
    tid = ctx.thread_registry.get_bound_thread_id(ctx.session_key)
    assert tid

    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(
            ok=True,
            summarized_count=3,
            preserved_count=4,
            file_path=rf"C:\Users\wally\.octop\workspaces\X\.octop\conversation_history\{tid}.md",
            reason="ok",
        )
    )
    ctx.agent_manager.get_agent = MagicMock(return_value=harness)

    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), ctx, sink)

    text = "\n".join(sink.lines)
    assert ".octop/conversation_history/" in text
    assert "C:" not in text
    assert "Users\\wally" not in text
    assert "Users/wally" not in text


@pytest.mark.asyncio
async def test_cmd_compact_prefers_display_path_over_host_file_path(ctx, dispatcher) -> None:
    await ctx.thread_registry.get_or_create_by_key(
        session_key=ctx.session_key,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        channel_type=ctx.channel_type,
    )
    tid = ctx.thread_registry.get_bound_thread_id(ctx.session_key)
    assert tid

    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(
            ok=True,
            summarized_count=3,
            preserved_count=4,
            file_path=rf"C:\Users\wally\.octop\workspaces\X\.octop\conversation_history\{tid}.md",
            display_path=f".octop/conversation_history/{tid}.md",
            reason="ok",
        )
    )
    ctx.agent_manager.get_agent = MagicMock(return_value=harness)

    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), ctx, sink)

    text = "\n".join(sink.lines)
    assert f".octop/conversation_history/{tid}.md" in text
    assert "C:" not in text
    assert "/Users/" not in text
    assert "/home/" not in text


@pytest.mark.asyncio
async def test_cmd_compact_uses_composer_model_ref(ctx, dispatcher) -> None:
    await ctx.thread_registry.get_or_create_by_key(
        session_key=ctx.session_key,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        channel_type=ctx.channel_type,
    )
    tid = ctx.thread_registry.get_bound_thread_id(ctx.session_key)
    assert tid
    ctx.model_ref = "Tencent Cloud HAI/MiniMax-M2.7"
    # Sticky /model must not beat the composer selection for this turn.
    ctx.agent_manager._harness_manager.get_thread_model.return_value = "DeepSeek/deepseek-chat"
    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(ok=True, summarized_count=3, reason="ok")
    )
    ctx.agent_manager.get_agent = MagicMock(return_value=harness)

    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), ctx, sink)

    assert harness.acompact_conversation.await_args.kwargs.get("model") == (
        "Tencent Cloud HAI/MiniMax-M2.7"
    )


@pytest.mark.asyncio
async def test_cmd_compact_nothing_to_compact(ctx, dispatcher) -> None:
    await ctx.thread_registry.get_or_create_by_key(
        session_key=ctx.session_key,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        channel_type=ctx.channel_type,
    )
    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(ok=False, reason="nothing_to_compact")
    )
    ctx.agent_manager.get_agent = MagicMock(return_value=harness)

    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), ctx, sink)
    text = "\n".join(sink.lines)
    assert "无可压缩" in text or "Nothing to compact" in text


@pytest.mark.asyncio
async def test_cmd_compact_unavailable_reason(ctx, dispatcher) -> None:
    await ctx.thread_registry.get_or_create_by_key(
        session_key=ctx.session_key,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        channel_type=ctx.channel_type,
    )
    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(ok=False, reason="unavailable", error="graph not ready")
    )
    ctx.agent_manager.get_agent = MagicMock(return_value=harness)

    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), ctx, sink)
    text = "\n".join(sink.lines)
    assert "不可用" in text or "unavailable" in text.lower()
    assert "graph not ready" not in text


@pytest.mark.asyncio
async def test_cmd_compact_respects_model_group_access_and_denies_when_unpermitted(
    tmp_path: Path, dispatcher
) -> None:
    import json

    from octop.infra.users.groups import RESOURCE_MODEL, GroupAccess

    db = SqlitePool(tmp_path / "compact-access.db")
    run_migrations(db)
    user_repo = UserRepo(db)
    user_repo.create(username="admin", password_hash="h", role="admin")
    allowed = user_repo.create(username="allowed", password_hash="h", role="user")
    outsider = user_repo.create(username="outsider", password_hash="h", role="user")
    agent_repo = AgentRepo(db)
    agent_repo.create(agent_id="a1", user_id=1, name="bot")
    registry = ThreadRegistry(session_repo=SessionRepo(db), thread_repo=ThreadRepo(db))
    manager = _agent_manager(tmp_path, db)
    services = build_shared_services(db=db, paths=PathLayout(tmp_path), config=OctopConfig())
    group_a = services.repos.user_group_repo.create(name="ga", description=None)
    group_b = services.repos.user_group_repo.create(name="gb", description=None)
    services.repos.user_group_member_repo.add(group_b.user_group_id, allowed)
    services.repos.provider_repo.create(
        name="deepseek",
        kind="openai",
        base_url="https://api.example.com/v1",
        api_key="sk-a",
        models_json=json.dumps([{"id": "m1", "enabled": True}]),
    )
    services.repos.provider_repo.create(
        name="zhipu",
        kind="openai",
        base_url="https://api.example.com/v1",
        api_key="sk-b",
        models_json=json.dumps([{"id": "m2", "enabled": True}]),
    )
    access = GroupAccess(services.repos)
    access.set_visibility(RESOURCE_MODEL, "deepseek/m1", "groups", [group_a.user_group_id])
    access.set_visibility(RESOURCE_MODEL, "zhipu/m2", "groups", [group_b.user_group_id])

    harness = MagicMock()
    harness.acompact_conversation = AsyncMock(
        return_value=CompactResult(ok=True, summarized_count=2, reason="ok")
    )
    manager.get_agent = MagicMock(return_value=harness)

    allowed_sk = ThreadRegistry.make_key(
        agent_id="a1", channel_type="ui", channel_subject_id=str(allowed)
    )
    await registry.get_or_create_by_key(
        session_key=allowed_sk,
        agent_id="a1",
        user_id=allowed,
        channel_type="ui",
    )
    allowed_ctx = SlashCtx(
        agent_id="a1",
        user_id=allowed,
        channel_type="ui",
        session_key=allowed_sk,
        thread_registry=registry,
        agent_repo=agent_repo,
        agent_manager=manager,
        model_ref="deepseek/m1",
    )
    sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), allowed_ctx, sink)
    assert harness.acompact_conversation.await_args.kwargs.get("model") == "zhipu/m2"

    harness.acompact_conversation.reset_mock()
    outsider_sk = ThreadRegistry.make_key(
        agent_id="a1", channel_type="ui", channel_subject_id=str(outsider)
    )
    await registry.get_or_create_by_key(
        session_key=outsider_sk,
        agent_id="a1",
        user_id=outsider,
        channel_type="ui",
    )
    outsider_ctx = SlashCtx(
        agent_id="a1",
        user_id=outsider,
        channel_type="ui",
        session_key=outsider_sk,
        thread_registry=registry,
        agent_repo=agent_repo,
        agent_manager=manager,
        model_ref="deepseek/m1",
    )
    manager._harness_manager.get_thread_model.return_value = "zhipu/m2"
    denied_sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), outsider_ctx, denied_sink)
    harness.acompact_conversation.assert_not_awaited()
    assert denied_sink.lines
    denied_text = "\n".join(denied_sink.lines)
    assert "deepseek/m1" in denied_text
    assert "zhipu/m2" not in denied_text

    manager._harness_manager.get_thread_model.return_value = None
    auto_outsider_ctx = SlashCtx(
        agent_id="a1",
        user_id=outsider,
        channel_type="ui",
        session_key=outsider_sk,
        thread_registry=registry,
        agent_repo=agent_repo,
        agent_manager=manager,
    )
    auto_denied_sink = BufferSink()
    await cmd_compact(dispatcher, SlashCommand("compact", ""), auto_outsider_ctx, auto_denied_sink)
    harness.acompact_conversation.assert_not_awaited()
    assert "auto" in "\n".join(auto_denied_sink.lines)
