import json
from pathlib import Path
from typing import Any, Dict, List, Optional

"""
Scaffold: Convert a full Azure Agent thread (multi-run, multi-turn) to
Azure AI Evaluation dataset JSONL.

Fill in the SDK integration inside fetch_thread_data() to pull the thread by
THREAD_ID, list all runs and messages, and return them in the expected shape.

Once you have messages, this script groups them into per-run examples compatible
with built-in agent evaluators (IntentResolution, ToolCallAccuracy,
TaskAdherence, Relevance, Groundedness). It also emits optional placeholders for
QA/Completeness/Retrieval if you want to enable those evaluators later.

Usage:
  python export_thread_to_eval_jsonl.py <THREAD_ID> <OUTPUT_PATH>

Notes:
- Keep messages in chronological order per run.
- "query" should be the conversation history visible to the assistant at the
  start of that run (system + prior user/assistant/tool messages).
- "response" should be the assistant/tool messages produced during that run
  up to the assistant's final text.
- If you include tool definitions, add them to each example under
  "tool_definitions" to enable ToolCallAccuracyEvaluator.
- For Retrieval/DocumentRetrieval, include retrieval traces and gold labels
  if available (see placeholders at the bottom of each example).
"""


def fetch_thread_data(thread_id: str) -> Dict[str, Any]:
    """
    TODO: Replace this stub with Azure AI Agents/Projects SDK calls that
    fetch the entire thread, its runs, and messages.

    Expected return structure (example):
    {
      "thread": {"id": thread_id},
      "runs": [
        {
          "run_id": "run_...",
          "messages": [
            {"createdAt": "...", "role": "system"|"user"|"assistant"|"tool", "content": [...]},
            ...
          ]
        },
        ...
      ],
      # Optional: tools available to the agent (for ToolCallAccuracy)
      "tool_definitions": [
        {"name": "get_orders", "description": "...", "parameters": {...}},
        # ...
      ]
    }
    """
    # 1) Use azure.ai.projects / azure.ai.agents client to:
    #   - retrieve thread
    #   - list runs in chronological order
    #   - list messages for the thread (or per run if supported)
    # 2) Normalize messages to the "content" schema expected by evaluators
    #    (type: text | tool_call | tool_result, name/arguments/tool_result, etc.)

    # Placeholder mock: minimal empty structure to let you wire SDK calls
    return {
        "thread": {"id": thread_id},
        "runs": [
            # Fill with actual runs and their messages
        ],
        "tool_definitions": [],  # Optional
    }


def split_query_and_response(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Given all messages for a single run in chronological order, split them into:
    - query: conversation history up to the start of this run (system + prior)
    - response: assistant/tool messages that belong to this run until final answer

    If your SDK returns per-run messages already separated, you can pass those
    directly as response and assemble query by taking all messages prior to this run.

    This function is a placeholder; adjust logic once you inspect your SDK payload.
    """
    # In many agent logs, each run block already contains the assistant's tool calls
    # and final text for that run. For a simple scaffold, treat all inbound roles
    # before the first assistant message in this list as query, and the rest as response.
    query: List[Dict[str, Any]] = []
    response: List[Dict[str, Any]] = []
    seen_assistant = False
    for msg in messages:
        if not seen_assistant and msg.get("role") != "assistant":
            query.append(msg)
        else:
            seen_assistant = True
            response.append(msg)
    return {"query": query, "response": response}


def to_evaluation_examples(thread_blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    tool_defs: List[Dict[str, Any]] = thread_blob.get("tool_definitions", [])

    for run in thread_blob.get("runs", []):
        run_id = run.get("run_id")
        messages = run.get("messages", [])
        parts = split_query_and_response(messages)

        example: Dict[str, Any] = {
            # Agent evaluators expect arrays of message objects for query/response
            "query": parts["query"],
            "response": parts["response"],
        }

        if tool_defs:
            example["tool_definitions"] = tool_defs

        # Optional placeholders if you want to enable more evaluators later
        # Response Completeness / QA
        example.setdefault("response_text", "<ASSISTANT_FINAL_TEXT_OPTIONAL>")
        example.setdefault("ground_truth", "<REFERENCE_OR_EXPECTED_POINTS_OPTIONAL>")
        # Retrieval
        example.setdefault("retrieved_documents", [])
        example.setdefault("gold_documents", [])
        examples.append(example)

    return examples


def write_jsonl(examples: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main() -> None:
    import sys

    if len(sys.argv) < 3:
        print("Usage: python export_thread_to_eval_jsonl.py <THREAD_ID> <OUTPUT_PATH>")
        sys.exit(1)

    thread_id = sys.argv[1]
    output_path = Path(sys.argv[2])

    thread_blob = fetch_thread_data(thread_id)
    examples = to_evaluation_examples(thread_blob)
    write_jsonl(examples, output_path)

    print(f"Wrote {len(examples)} examples to {output_path.resolve()}")
    print(
        "You can now run azure.ai.evaluation.evaluate(data=<output_path>, evaluators={...}, azure_ai_project=<...>)"
    )


if __name__ == "__main__":
    main()
