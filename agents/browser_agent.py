"""
agents/browser_agent.py

Browser agent built with LangChain + GitHub Models.
Config is loaded from .env

.env variables used:
    AI_API_KEY   = your GitHub personal access token (ghp_...)
    AI_ENDPOINT  = https://models.inference.ai.azure.com
    AI_MODEL     = any model available on GitHub Models
    MAX_STEPS    = 20
    MAX_TOKENS   = 4096

ReAct loop:
    1. LLM is bound to BROWSER_TOOLS via llm.bind_tools()
    2. Call LLM → if tool_calls present → execute tools → append ToolMessages → repeat
    3. When LLM returns no tool_calls → extract final answer → done
"""

import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from tools.browser_tools import BROWSER_TOOLS, close_browser

# ─────────────────────────────────────────────────────────────────────────────
# Load config from .env
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

AI_API_KEY  = os.getenv("AI_API_KEY")
AI_ENDPOINT = os.getenv("AI_ENDPOINT", "https://models.inference.ai.azure.com")
AI_MODEL    = os.getenv("AI_MODEL", "gpt-4o-mini")
MAX_STEPS   = int(os.getenv("MAX_STEPS", "20"))
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "4096"))

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a browser agent helping to find job listings on a job board website.

You have access to browser tools (navigate, click, fill, get_page_text, etc.).
Use them step by step to accomplish the task given by the user.

Guidelines:
- Always start by navigating to the target URL.
- Use get_page_text() to understand the page structure before interacting.
- Use fill() to type in search boxes, then click() the search button.
- Use get_page_html('.job-card') to list visible job results.
- Use select_option() to apply dropdown filters.
- Click into job cards to read full details when needed.
- When you have collected enough information, stop calling tools
  and write a clear, structured summary of what you found.
- Be efficient: do not repeat the same action twice.
- If a selector fails, try a simpler alternative.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Tool executor
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_MAP = {t.name: t for t in BROWSER_TOOLS}


def _execute_tool_call(tool_call: dict) -> ToolMessage:
    """
    Execute one tool call returned by the LLM and wrap the result in a ToolMessage.

    tool_call format (LangChain standard):
        {"id": "call_abc123", "name": "navigate", "args": {"url": "http://..."}}
    """
    name   = tool_call["name"]
    args   = tool_call["args"]
    use_id = tool_call["id"]

    tool_fn = _TOOL_MAP.get(name)
    if tool_fn is None:
        result = json.dumps({"error": f"Unknown tool: {name}"})
    else:
        try:
            result = tool_fn.invoke(args)
        except Exception as e:
            result = json.dumps({"error": str(e)})

    return ToolMessage(content=str(result), tool_call_id=use_id)


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(task: str, site_url: str | None = None) -> dict:
    """
    Run the browser agent on a task.

    Args:
        task:     Natural language instruction, e.g. "Find remote Python jobs".
        site_url: URL passed to the agent (e.g. the local mock site address).

    Returns:
        {
            "answer":     str,          # LLM final text summary
            "steps":      list[dict],   # each tool call + result
            "step_count": int,
        }
    """
    # ── Build the LLM pointed at the GitHub Models endpoint ──────────────────
    llm = ChatOpenAI(
        model=AI_MODEL,
        api_key=AI_API_KEY,
        base_url=AI_ENDPOINT
    ).bind_tools(BROWSER_TOOLS)

    # ── Initial message history ───────────────────────────────────────────────
    user_content = task
    if site_url:
        user_content += f"\n\nThe job board is available at: {site_url}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    steps        = []
    step_count   = 0
    final_answer = ""

    print(f"\n{'='*60}")
    print(f"MODEL   : {AI_MODEL}")
    print(f"ENDPOINT: {AI_ENDPOINT}")
    print(f"TASK    : {task}")
    print(f"{'='*60}\n")

    try:
        while step_count < MAX_STEPS:

            # ── 1. Call the LLM ──────────────────────────────────────────
            response: AIMessage = llm.invoke(messages)

            print(f"[Step {step_count}] tool_calls: {len(response.tool_calls)}")

            # ── 2. No tool calls → LLM is done ───────────────────────────
            if not response.tool_calls:
                final_answer = response.content
                print(f"\n[DONE] Agent finished after {step_count} tool calls.\n")
                break

            # ── 3. Append assistant turn to history ───────────────────────
            messages.append(response)

            # ── 4. Execute every tool call and collect ToolMessages ───────
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                print(f"  → {name}({json.dumps(args)})")

                tool_message = _execute_tool_call(tool_call)

                preview = tool_message.content[:200]
                suffix  = "..." if len(tool_message.content) > 200 else ""
                print(f"     ↳ {preview}{suffix}")

                steps.append({
                    "step":   step_count,
                    "tool":   name,
                    "inputs": args,
                    "result": tool_message.content,
                })

                # ── 5. Feed result back to the LLM ───────────────────────
                messages.append(tool_message)
                step_count += 1

        else:
            final_answer = f"[Agent stopped: reached MAX_STEPS={MAX_STEPS}]"
            print(f"\n[WARN] Max steps ({MAX_STEPS}) reached.\n")

    finally:
        close_browser()

    return {
        "answer":     final_answer,
        "steps":      steps,
        "step_count": step_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict):
    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(result["answer"])
    print(f"\nTotal tool calls: {result['step_count']}")
    print("\nStep-by-step trace:")
    for s in result["steps"]:
        print(f"  [{s['step']}] {s['tool']}({json.dumps(s['inputs'])})")
