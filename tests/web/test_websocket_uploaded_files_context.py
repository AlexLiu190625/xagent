from types import SimpleNamespace

from xagent.core.agent.trace import get_display_user_message
from xagent.web.api.websocket import (
    _append_uploaded_files_context_to_message,
    _build_uploaded_files_context,
    _display_message_for_user,
    _selected_file_refs_from_task,
)


def test_build_uploaded_files_context_includes_agent_builder_kb_instruction():
    context = _build_uploaded_files_context(
        [
            {
                "file_id": "file-123",
                "name": "faq.docx",
                "original_name": "FAQ.docx",
            }
        ],
        is_agent_builder=True,
    )

    assert "FAQ.docx: file_id=file-123" in context
    assert "create_knowledge_base_from_file" in context
    assert 'file_ids = ["file-123"]' in context
    assert "Do NOT ask the user to upload again" in context


def test_append_uploaded_files_context_to_message_is_idempotent():
    context = _build_uploaded_files_context(
        [{"file_id": "file-123", "name": "faq.docx"}],
        is_agent_builder=False,
    )

    message = _append_uploaded_files_context_to_message("Upload File", context)
    assert message.startswith("Upload File\n\n## UPLOADED FILES")
    assert _append_uploaded_files_context_to_message(message, context) == message


def test_selected_file_refs_from_task_uses_task_create_file_ids():
    class Task:
        agent_config = {
            "selected_file_ids": [" file-123 ", "", 42, "file-456"],
        }

    assert _selected_file_refs_from_task(Task()) == [
        {"file_id": "file-123"},
        {"file_id": "file-456"},
    ]


def test_selected_file_refs_from_task_ignores_missing_config():
    class Task:
        agent_config = None

    assert _selected_file_refs_from_task(Task()) == []


def test_get_display_user_message_reads_agent_context_state():
    context = SimpleNamespace(
        state={
            "display_user_message": "Summarize this document",
        }
    )

    assert (
        get_display_user_message(
            context,
            "Summarize this document\n\n## UPLOADED FILES\nfile_id=file-123",
        )
        == "Summarize this document"
    )


def test_display_message_for_file_only_turn_uses_placeholder():
    assert _display_message_for_user("", has_files=True) == "Uploaded file(s)"
    assert (
        _display_message_for_user("Summarize this document", has_files=True)
        == "Summarize this document"
    )
