"""
tests/test_browser_agent.py

Tests for the agent loop in agents/browser_agent.py.
Both the LLM and the tool execution are fully mocked.
No API key, no Playwright, no network — pure logic tests.

Run:
    pytest tests/test_browser_agent.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ai_with_tool_call(tool_name: str, args: dict, call_id: str = "call_001") -> AIMessage:
    """AIMessage that contains one tool_call — simulates LLM asking to use a tool."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
    return msg


def _ai_final_answer(text: str) -> AIMessage:
    """AIMessage with no tool_calls — signals the agent to stop."""
    msg = AIMessage(content=text)
    msg.tool_calls = []
    return msg


def _fake_tool_result(content: dict) -> str:
    """JSON string the mock tool executor returns."""
    return json.dumps(content)


# ─────────────────────────────────────────────────────────────────────────────
# Core fixture: mock LLM + mock tool execution
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """
    Provide run_agent with:
      - ChatOpenAI replaced by a controllable mock
      - _execute_tool_call replaced by a mock that returns a canned result
      - close_browser patched to a no-op

    Yields (run_agent_fn, mock_llm, mock_execute)
    so individual tests can set .invoke.side_effect and .return_value freely.
    """
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    mock_execute = MagicMock()
    # Default tool result unless a test overrides it
    mock_execute.return_value = ToolMessage(
        content=_fake_tool_result({"status": "ok"}),
        tool_call_id="call_001",
    )

    with patch("agents.browser_agent.ChatOpenAI", return_value=mock_llm), \
         patch("agents.browser_agent._execute_tool_call", side_effect=mock_execute), \
         patch("agents.browser_agent.close_browser"):
        from agents import browser_agent
        # Force reimport so patches are active
        import importlib
        importlib.reload(browser_agent)
        with patch("agents.browser_agent.ChatOpenAI", return_value=mock_llm), \
             patch("agents.browser_agent._execute_tool_call", side_effect=mock_execute), \
             patch("agents.browser_agent.close_browser"):
            yield browser_agent.run_agent, mock_llm, mock_execute


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoop:

    def test_stops_immediately_on_final_answer(self, agent):
        run_agent_fn, mock_llm, _ = agent
        mock_llm.invoke.return_value = _ai_final_answer("Found 3 Python jobs.")

        result = run_agent_fn(task="Find Python jobs")

        assert result["answer"] == "Found 3 Python jobs."
        assert result["step_count"] == 0
        assert result["steps"] == []

    def test_executes_one_tool_then_stops(self, agent):
        run_agent_fn, mock_llm, mock_execute = agent

        mock_execute.return_value = ToolMessage(
            content=_fake_tool_result({"status": "navigated", "page_title": "MockJobs"}),
            tool_call_id="c1",
        )
        mock_llm.invoke.side_effect = [
            _ai_with_tool_call("navigate", {"url": "http://localhost:8765"}, call_id="c1"),
            _ai_final_answer("Navigation successful."),
        ]

        result = run_agent_fn(task="Go to job board")

        assert result["step_count"] == 1
        assert result["steps"][0]["tool"] == "navigate"
        assert result["steps"][0]["inputs"] == {"url": "http://localhost:8765"}

    def test_executes_multiple_tools_in_sequence(self, agent):
        run_agent_fn, mock_llm, mock_execute = agent

        def tool_result_for(call, *args, **kwargs):
            name = call["name"]
            results = {
                "navigate":      {"status": "navigated", "page_title": "MockJobs"},
                "get_page_text": {"text": "Senior Python Engineer...", "truncated": False},
                "get_page_html": {"count": 8, "elements": ["<div>job1</div>"]},
            }
            return ToolMessage(
                content=_fake_tool_result(results.get(name, {"status": "ok"})),
                tool_call_id=call["id"],
            )

        mock_execute.side_effect = tool_result_for
        mock_llm.invoke.side_effect = [
            _ai_with_tool_call("navigate",      {"url": "http://localhost:8765"}, call_id="c1"),
            _ai_with_tool_call("get_page_text", {},                               call_id="c2"),
            _ai_with_tool_call("get_page_html", {"selector": ".job-card"},        call_id="c3"),
            _ai_final_answer("Found 8 job listings."),
        ]

        result = run_agent_fn(task="List all jobs")

        assert result["step_count"] == 3
        assert [s["tool"] for s in result["steps"]] == [
            "navigate", "get_page_text", "get_page_html"
        ]
        assert result["answer"] == "Found 8 job listings."

    def test_respects_max_steps(self, agent):
        run_agent_fn, mock_llm, _ = agent

        # LLM always asks to scroll — never finishes voluntarily
        mock_llm.invoke.return_value = _ai_with_tool_call(
            "scroll_down", {"pixels": 300}, call_id="c1"
        )

        with patch("agents.browser_agent.MAX_STEPS", 3):
            result = run_agent_fn(task="Scroll forever")

        assert result["step_count"] == 3
        assert "MAX_STEPS" in result["answer"]

    def test_tool_result_is_fed_back_to_llm(self, agent):
        run_agent_fn, mock_llm, mock_execute = agent

        nav_result = _fake_tool_result({"status": "navigated", "page_title": "MockJobs"})
        mock_execute.return_value = ToolMessage(content=nav_result, tool_call_id="c1")
        mock_llm.invoke.side_effect = [
            _ai_with_tool_call("navigate", {"url": "http://localhost:8765"}, call_id="c1"),
            _ai_final_answer("Done."),
        ]

        run_agent_fn(task="Navigate and stop")

        # The second LLM call must receive the ToolMessage in its messages
        second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
        tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "c1"
        assert json.loads(tool_msgs[0].content)["status"] == "navigated"

    def test_step_record_has_correct_structure(self, agent):
        run_agent_fn, mock_llm, mock_execute = agent

        mock_execute.return_value = ToolMessage(
            content=_fake_tool_result({"status": "filled", "value": "Python"}),
            tool_call_id="c1",
        )
        mock_llm.invoke.side_effect = [
            _ai_with_tool_call("fill", {"selector": "#search-input", "text": "Python"}, call_id="c1"),
            _ai_final_answer("Done."),
        ]

        result = run_agent_fn(task="Search Python")

        step = result["steps"][0]
        assert step["step"] == 0
        assert step["tool"] == "fill"
        assert step["inputs"] == {"selector": "#search-input", "text": "Python"}
        assert "result" in step
        assert json.loads(step["result"])["status"] == "filled"

    def test_unknown_tool_returns_error_not_crash(self, agent):
        run_agent_fn, mock_llm, mock_execute = agent

        # Simulate _execute_tool_call returning an error for an unknown tool
        mock_execute.return_value = ToolMessage(
            content=_fake_tool_result({"error": "Unknown tool: fly_to_moon"}),
            tool_call_id="c1",
        )
        mock_llm.invoke.side_effect = [
            _ai_with_tool_call("fly_to_moon", {"destination": "moon"}, call_id="c1"),
            _ai_final_answer("Handled gracefully."),
        ]

        result = run_agent_fn(task="Unknown tool test")

        assert result["step_count"] == 1
        step_result = json.loads(result["steps"][0]["result"])
        assert "error" in step_result

    def test_site_url_included_in_first_message(self, agent):
        run_agent_fn, mock_llm, _ = agent
        mock_llm.invoke.return_value = _ai_final_answer("Done.")

        run_agent_fn(task="Find jobs", site_url="http://localhost:8765")

        first_messages = mock_llm.invoke.call_args_list[0][0][0]
        human_msgs = [m for m in first_messages if isinstance(m, HumanMessage)]
        assert any("http://localhost:8765" in m.content for m in human_msgs)

    def test_system_prompt_is_always_first_message(self, agent):
        run_agent_fn, mock_llm, _ = agent
        mock_llm.invoke.return_value = _ai_final_answer("Done.")

        run_agent_fn(task="Any task")

        first_messages = mock_llm.invoke.call_args_list[0][0][0]
        assert isinstance(first_messages[0], SystemMessage)

    def test_result_always_has_required_keys(self, agent):
        run_agent_fn, mock_llm, _ = agent
        mock_llm.invoke.return_value = _ai_final_answer("Done.")

        result = run_agent_fn(task="Any task")

        assert "answer" in result
        assert "steps" in result
        assert "step_count" in result

    def test_multiple_tool_calls_in_one_turn(self, agent):
        """LLM can return several tool_calls in one response (parallel tool use)."""
        run_agent_fn, mock_llm, mock_execute = agent

        # Response with two tool calls at once
        multi_tool_response = AIMessage(content="")
        multi_tool_response.tool_calls = [
            {"id": "c1", "name": "navigate",      "args": {"url": "http://localhost:8765"}},
            {"id": "c2", "name": "get_page_text", "args": {}},
        ]

        def tool_result_for(call, *args, **kwargs):
            return ToolMessage(
                content=_fake_tool_result({"status": "ok", "tool": call["name"]}),
                tool_call_id=call["id"],
            )

        mock_execute.side_effect = tool_result_for
        mock_llm.invoke.side_effect = [
            multi_tool_response,
            _ai_final_answer("Done."),
        ]

        result = run_agent_fn(task="Navigate and read")

        assert result["step_count"] == 2
        assert result["steps"][0]["tool"] == "navigate"
        assert result["steps"][1]["tool"] == "get_page_text"
