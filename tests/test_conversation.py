"""Tests for the conversation models — Role, Message, Conversation."""

import uuid

import pytest

from gluellm.models.conversation import Conversation, Message, Role


class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_is_string_enum(self):
        assert isinstance(Role.USER, str)
        assert Role.USER == "user"


class TestMessage:
    def test_creation(self):
        msg = Message(id="msg-1", role=Role.USER, content="Hello")
        assert msg.id == "msg-1"
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_requires_all_fields(self):
        with pytest.raises(Exception):
            Message(role=Role.USER, content="no id")


class TestConversation:
    def test_default_empty(self):
        conv = Conversation()
        assert len(conv.messages) == 0
        assert conv.messages_dict == []

    def test_auto_generated_id(self):
        conv = Conversation()
        uuid.UUID(conv.id)

    def test_add_message(self):
        conv = Conversation()
        conv.add_message(Role.USER, "Hello")
        assert len(conv.messages) == 1
        assert conv.messages[0].role == Role.USER
        assert conv.messages[0].content == "Hello"

    def test_add_multiple_messages(self):
        conv = Conversation()
        conv.add_message(Role.SYSTEM, "You are helpful.")
        conv.add_message(Role.USER, "Hi")
        conv.add_message(Role.ASSISTANT, "Hello!")
        assert len(conv.messages) == 3

    def test_messages_dict_format(self):
        conv = Conversation()
        conv.add_message(Role.USER, "What is 2+2?")
        conv.add_message(Role.ASSISTANT, "4")
        dicts = conv.messages_dict
        assert dicts == [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

    def test_messages_dict_preserves_order(self):
        conv = Conversation()
        conv.add_message(Role.SYSTEM, "sys")
        conv.add_message(Role.USER, "user")
        conv.add_message(Role.ASSISTANT, "asst")
        conv.add_message(Role.TOOL, "tool result")
        roles = [m["role"] for m in conv.messages_dict]
        assert roles == ["system", "user", "assistant", "tool"]

    def test_message_ids_are_unique(self):
        conv = Conversation()
        conv.add_message(Role.USER, "msg1")
        conv.add_message(Role.USER, "msg2")
        ids = [m.id for m in conv.messages]
        assert ids[0] != ids[1]

    def test_separate_conversations_are_isolated(self):
        c1 = Conversation()
        c2 = Conversation()
        c1.add_message(Role.USER, "only in c1")
        assert len(c1.messages) == 1
        assert len(c2.messages) == 0
