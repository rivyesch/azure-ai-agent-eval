import json
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_last_user_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the last user message text from a message list."""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if text:
                        return text
    return None


def _last_user_query(item: Dict[str, Any]) -> Optional[str]:
    """Get the last user query from the item."""
    q = item.get("query")
    if isinstance(q, str):
        return q
    if isinstance(q, list):
        return _extract_last_user_text(q)
    return None


def _extract_last_assistant_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the last assistant message text from a message list."""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if text:
                        return text
    return None


def _final_assistant_response(item: Dict[str, Any]) -> Optional[str]:
    """Get the final assistant response from the item."""
    r = item.get("response")
    if isinstance(r, str):
        return r
    if isinstance(r, list):
        return _extract_last_assistant_text(r)
    return None


def _slim_tool_definitions(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Extract slim tool definitions."""
    tools = item.get("tool_definitions") or item.get("tools")
    if not isinstance(tools, list):
        return None
    slim: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        entry: Dict[str, Any] = {
            k: v
            for k, v in t.items()
            if k in {"name", "type", "description", "parameters", "operationId"}
        }
        if entry:
            slim.append(entry)
    return slim or None


def _slim_tool_calls(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Extract slim tool calls."""
    calls = item.get("tool_calls") or item.get("tools_called")
    if not isinstance(calls, list):
        return None
    slim: List[Dict[str, Any]] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        entry = {k: c.get(k) for k in ("name", "arguments") if k in c}
        if entry:
            slim.append(entry)
    return slim or None


def _extract_context_from_messages(
    messages: List[Dict[str, Any]],
    max_contexts: int,
    snippet_max_chars: int,
) -> List[str]:
    """
    Extract context from tool role messages in a message list.
    Looks for file_search tool results with KB article content.
    """
    contexts: List[str] = []
    
    for m in messages:
        if not isinstance(m, dict):
            continue
        
        # Look for tool role messages
        if m.get("role") != "tool":
            continue
        
        content = m.get("content")
        if not isinstance(content, list):
            continue
        
        for part in content:
            if not isinstance(part, dict):
                continue
            
            # Look for tool_result type
            if part.get("type") != "tool_result":
                continue
            
            tool_result = part.get("tool_result")
            if not isinstance(tool_result, list):
                continue
            
            # Each tool_result contains multiple file results
            for result in tool_result:
                if not isinstance(result, dict):
                    continue
                
                file_name = result.get("file_name", "")
                result_content = result.get("content")
                
                if not isinstance(result_content, list):
                    continue
                
                # Extract text from content
                for content_part in result_content:
                    if not isinstance(content_part, dict):
                        continue
                    
                    text = content_part.get("text", "")
                    if not text:
                        continue
                    
                    # Create context entry with source attribution
                    context_entry = text.strip()
                    if file_name:
                        context_entry = f"[Source: {file_name}]\n{context_entry}"
                    
                    # Truncate if needed
                    if len(context_entry) > snippet_max_chars:
                        context_entry = context_entry[:snippet_max_chars] + "..."
                    
                    contexts.append(context_entry)
                    
                    if len(contexts) >= max_contexts:
                        return contexts
    
    return contexts


def _collect_context(item: Dict[str, Any]) -> Optional[List[str]]:
    """Collect context from explicit context fields."""
    candidates = [
        item.get("context"),
        item.get("retrieved_context"),
        item.get("evidence"),
        item.get("citations"),
    ]
    for c in candidates:
        if not c:
            continue
        if isinstance(c, str):
            return [c]
        if isinstance(c, list):
            texts: List[str] = []
            for v in c:
                if isinstance(v, str):
                    texts.append(v)
                elif isinstance(v, dict) and "text" in v:
                    texts.append(str(v.get("text")))
            return texts or None
    return None


def _build_context_from_tool_results(
    item: Dict[str, Any], max_contexts: int, snippet_max_chars: int
) -> Optional[List[str]]:
    """Build context by extracting tool results from query and response messages."""
    contexts: List[str] = []
    
    # Try to extract from query messages
    q = item.get("query")
    if isinstance(q, list):
        contexts.extend(
            _extract_context_from_messages(q, max_contexts, snippet_max_chars)
        )
    
    # Try to extract from response messages if we need more
    if len(contexts) < max_contexts:
        r = item.get("response")
        if isinstance(r, list):
            more = _extract_context_from_messages(
                r, max_contexts - len(contexts), snippet_max_chars
            )
            contexts.extend(more)
    
    return contexts if contexts else None


def _get_ground_truth(item: Dict[str, Any]) -> Optional[str]:
    """Get ground truth answer."""
    gt = item.get("ground_truth") or item.get("reference_answer")
    if isinstance(gt, str):
        return gt
    return None


def _get_doc_gt_and_retrieved(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get ground truth and retrieved documents."""
    gt_docs = item.get("ground_truth_documents") or item.get("labels")
    ret_docs = item.get("retrieved_documents") or item.get("documents")
    if isinstance(gt_docs, list) and isinstance(ret_docs, list):
        return {
            "ground_truth_documents": gt_docs,
            "retrieved_documents": ret_docs,
        }
    return None


def postprocess(
    input_path: str,
    out_dir: str,
    rag_snippets_k: int = 3,
    snippet_max_chars: int = 2000,
    debug: bool = False,
) -> Dict[str, str]:
    """
    Post-process evaluation JSONL into lean group-specific files.
    
    Args:
        input_path: Path to input JSONL file
        out_dir: Directory to write output files
        rag_snippets_k: Max number of context snippets to extract (0 to disable)
        snippet_max_chars: Max characters per context snippet
        debug: Print debug information
    """
    general_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    rag_rows: List[Dict[str, Any]] = []
    docret_rows: List[Dict[str, Any]] = []
    respcomp_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(_read_jsonl(input_path)):
        if debug:
            print(f"\n=== Processing item {idx} ===")
            print(f"Top-level keys: {list(item.keys())}")
        
        query = _last_user_query(item)
        response = _final_assistant_response(item)
        
        if debug:
            print(f"Query extracted: {bool(query)}")
            print(f"Response extracted: {bool(response)}")
            if query:
                print(f"Query preview: {query[:100]}...")
            if response:
                print(f"Response preview: {response[:100]}...")
        
        if not query and isinstance(item.get("query"), dict):
            query = json.dumps(item["query"], ensure_ascii=False)

        # general & Security: query + response
        if query or response:
            general_rows.append({
                k: v for k, v in (("query", query), ("response", response)) if v is not None
            })

        # Agent: query + response + tool_definitions + tool_calls
        tool_defs = _slim_tool_definitions(item)
        tool_calls = _slim_tool_calls(item)
        agent_row: Dict[str, Any] = {
            k: v for k, v in (("query", query), ("response", response)) if v is not None
        }
        if tool_defs:
            agent_row["tool_definitions"] = tool_defs
        if tool_calls:
            agent_row["tool_calls"] = tool_calls
        if agent_row:
            agent_rows.append(agent_row)

        # RAG: query + response + context
        # First try explicit context fields
        context = _collect_context(item)
        
        if debug:
            print(f"Explicit context found: {bool(context)}")
        
        # If no explicit context, try to extract from tool results
        if not context and rag_snippets_k > 0:
            context = _build_context_from_tool_results(
                item, rag_snippets_k, snippet_max_chars
            )
            if debug:
                print(f"Context from tool results: {bool(context)}")
                if context:
                    print(f"Number of context snippets: {len(context)}")
                    print(f"First snippet preview: {context[0][:200]}...")
        
        if context and query and response:
            rag_rows.append({
                "query": query,
                "response": response,
                "context": context,
            })
            if debug:
                print("✓ Added to RAG rows")
        elif debug:
            print(f"✗ Not added to RAG rows - context:{bool(context)}, query:{bool(query)}, response:{bool(response)}")

        # Document Retrieval: query + ground_truth_documents + retrieved_documents
        dr = _get_doc_gt_and_retrieved(item)
        if dr:
            base: Dict[str, Any] = {"query": query} if query is not None else {}
            base.update(dr)
            docret_rows.append(base)

        # Response Completeness: query + response + ground_truth
        gt = _get_ground_truth(item)
        if gt and response:
            respcomp_rows.append({
                "query": query,
                "response": response,
                "ground_truth": gt,
            })

    outputs: Dict[str, str] = {}

    if debug:
        print(f"\n=== Summary ===")
        print(f"General rows: {len(general_rows)}")
        print(f"Agent rows: {len(agent_rows)}")
        print(f"RAG rows: {len(rag_rows)}")
        print(f"DocRet rows: {len(docret_rows)}")
        print(f"RespComp rows: {len(respcomp_rows)}")

    if general_rows:
        path = f"{out_dir}/general_qa.jsonl"
        _write_jsonl(path, general_rows)
        outputs["general_qa"] = path

    if agent_rows:
        path = f"{out_dir}/agent_basic.jsonl"
        _write_jsonl(path, agent_rows)
        outputs["agent_basic"] = path

    if rag_rows:
        path = f"{out_dir}/rag_core.jsonl"
        _write_jsonl(path, rag_rows)
        outputs["rag_core"] = path

    if docret_rows:
        path = f"{out_dir}/document_retrieval.jsonl"
        _write_jsonl(path, docret_rows)
        outputs["document_retrieval"] = path

    if respcomp_rows:
        path = f"{out_dir}/response_completeness.jsonl"
        _write_jsonl(path, respcomp_rows)
        outputs["response_completeness"] = path

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process evaluation JSONL into lean group-specific files"
    )
    parser.add_argument("--input", required=True, help="Path to evaluation_input_data.jsonl")
    parser.add_argument("--out_dir", default=".", help="Directory to write output JSONL files")
    parser.add_argument(
        "--rag_snippets_k",
        type=int,
        default=3,
        help="Max number of tool-result snippets to include as context (0 to disable)",
    )
    parser.add_argument(
        "--snippet_max_chars",
        type=int,
        default=2000,
        help="Max characters per context snippet",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args()

    result = postprocess(
        args.input,
        args.out_dir,
        rag_snippets_k=args.rag_snippets_k,
        snippet_max_chars=args.snippet_max_chars,
        debug=args.debug,
    )
    print(json.dumps(result, indent=2))

# import json
# from typing import Any, Dict, Iterable, List, Optional


# def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             yield json.loads(line)


# def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         for row in rows:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")


# def _extract_last_user_text(messages: List[Dict[str, Any]]) -> Optional[str]:
#     for m in reversed(messages):
#         if isinstance(m, dict) and m.get("role") == "user":
#             content = m.get("content")
#             if isinstance(content, str):
#                 return content
#             if isinstance(content, list) and content:
#                 for part in content:
#                     if isinstance(part, dict) and part.get("type") == "text":
#                         return part.get("text")
#     return None


# def _last_user_query(item: Dict[str, Any]) -> Optional[str]:
#     # Prefer explicit field if provided
#     q = item.get("query")
#     if isinstance(q, str):
#         return q
#     if isinstance(q, list):
#         return _extract_last_user_text(q)  # query can be a message list
#     # Fallback: derive from messages if available
#     messages = item.get("messages") or item.get("conversation")
#     if isinstance(messages, list):
#         return _extract_last_user_text(messages)
#     return None


# def _extract_last_assistant_text(messages: List[Dict[str, Any]]) -> Optional[str]:
#     for m in reversed(messages):
#         if isinstance(m, dict) and m.get("role") == "assistant":
#             content = m.get("content")
#             if isinstance(content, str):
#                 return content
#             if isinstance(content, list) and content:
#                 for part in content:
#                     if isinstance(part, dict) and part.get("type") == "text":
#                         return part.get("text")
#     return None


# def _final_assistant_response(item: Dict[str, Any]) -> Optional[str]:
#     r = item.get("response")
#     if isinstance(r, str):
#         return r
#     if isinstance(r, list):
#         return _extract_last_assistant_text(r)  # response can be a message list
#     messages = item.get("messages") or item.get("conversation")
#     if isinstance(messages, list):
#         return _extract_last_assistant_text(messages)
#     return None


# def _slim_tool_definitions(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
#     tools = item.get("tool_definitions") or item.get("tools")
#     if not isinstance(tools, list):
#         return None
#     slim: List[Dict[str, Any]] = []
#     for t in tools:
#         if not isinstance(t, dict):
#             continue
#         # Keep only essentials: name/type/signature or schema; drop large KB/file payloads
#         entry: Dict[str, Any] = {
#             k: v
#             for k, v in t.items()
#             if k in {"name", "type", "description", "parameters", "operationId"}
#         }
#         if entry:
#             slim.append(entry)
#     return slim or None


# def _slim_tool_calls(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
#     calls = item.get("tool_calls") or item.get("tools_called")
#     if not isinstance(calls, list):
#         return None
#     slim: List[Dict[str, Any]] = []
#     for c in calls:
#         if not isinstance(c, dict):
#             continue
#         entry = {k: c.get(k) for k in ("name", "arguments") if k in c}
#         if entry:
#             slim.append(entry)
#     return slim or None


# def _collect_context(item: Dict[str, Any]) -> Optional[List[str]]:
#     # Try common fields where retriever context might be stored
#     # Support both string or list-of-strings; normalize to list[str]
#     candidates = [
#         item.get("context"),
#         item.get("retrieved_context"),
#         item.get("evidence"),
#         item.get("citations"),
#     ]
#     for c in candidates:
#         if not c:
#             continue
#         if isinstance(c, str):
#             return [c]
#         if isinstance(c, list):
#             texts: List[str] = []
#             for v in c:
#                 if isinstance(v, str):
#                     texts.append(v)
#                 elif isinstance(v, dict) and "text" in v:
#                     texts.append(str(v.get("text")))
#             return texts or None
#     return None


# def _get_ground_truth(item: Dict[str, Any]) -> Optional[str]:
#     gt = item.get("ground_truth") or item.get("reference_answer")
#     if isinstance(gt, str):
#         return gt
#     return None


# def _get_doc_gt_and_retrieved(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     gt_docs = item.get("ground_truth_documents") or item.get("labels")
#     ret_docs = item.get("retrieved_documents") or item.get("documents")
#     if isinstance(gt_docs, list) and isinstance(ret_docs, list):
#         return {
#             "ground_truth_documents": gt_docs,
#             "retrieved_documents": ret_docs,
#         }
#     return None


# def _extract_tool_result_snippets_from_messages(
#     messages: List[Dict[str, Any]],
#     k: int,
#     snippet_max_chars: int,
# ) -> List[str]:
#     snippets: List[str] = []
#     for m in messages:
#         if not isinstance(m, dict):
#             continue
#         if m.get("role") != "tool":
#             continue
#         content = m.get("content")
#         if isinstance(content, list):
#             for part in content:
#                 if not isinstance(part, dict):
#                     continue
#                 if part.get("type") != "tool_result":
#                     continue
#                 txt = None
#                 if isinstance(part.get("text"), str):
#                     txt = part.get("text")
#                 elif isinstance(part.get("data"), str):
#                     txt = part.get("data")
#                 elif isinstance(part.get("content"), str):
#                     txt = part.get("content")
#                 elif "text" in part and isinstance(part.get("text"), list):
#                     for tpart in part.get("text"):
#                         if isinstance(tpart, str):
#                             txt = tpart
#                             break
#                         if isinstance(tpart, dict) and isinstance(tpart.get("text"), str):
#                             txt = tpart.get("text")
#                             break
#                 if txt:
#                     trimmed = txt.strip().replace("\r", " ").replace("\n", " ")
#                     if trimmed:
#                         snippets.append(trimmed[:snippet_max_chars])
#                         if len(snippets) >= k:
#                             return snippets
#     return snippets


# def _build_context_from_tool_results(
#     item: Dict[str, Any], k: int, snippet_max_chars: int
# ) -> Optional[List[str]]:
#     snippets: List[str] = []
#     q = item.get("query")
#     if isinstance(q, list):
#         snippets.extend(
#             _extract_tool_result_snippets_from_messages(q, k, snippet_max_chars)
#         )
#     if len(snippets) < k:
#         r = item.get("response")
#         if isinstance(r, list):
#             more = _extract_tool_result_snippets_from_messages(
#                 r, k - len(snippets), snippet_max_chars
#             )
#             snippets.extend(more)
#     return snippets or None


# def postprocess(input_path: str, out_dir: str, rag_snippets_k: int = 3, snippet_max_chars: int = 400) -> Dict[str, str]:
#     safety_rows: List[Dict[str, Any]] = []
#     agent_rows: List[Dict[str, Any]] = []
#     rag_rows: List[Dict[str, Any]] = []
#     docret_rows: List[Dict[str, Any]] = []
#     respcomp_rows: List[Dict[str, Any]] = []

#     for item in _read_jsonl(input_path):
#         query = _last_user_query(item)
#         response = _final_assistant_response(item)
#         if not query and isinstance(item.get("query"), dict):
#             # Some converters export query as dict {role/content}; try to stringify
#             query = json.dumps(item["query"], ensure_ascii=False)

#         # Common base rows
#         if query or response:
#             safety_rows.append({k: v for k, v in (("query", query), ("response", response)) if v is not None})

#         # Agent basics
#         tool_defs = _slim_tool_definitions(item)
#         tool_calls = _slim_tool_calls(item)
#         agent_row: Dict[str, Any] = {k: v for k, v in (("query", query), ("response", response)) if v is not None}
#         if tool_defs:
#             agent_row["tool_definitions"] = tool_defs
#         if tool_calls:
#             agent_row["tool_calls"] = tool_calls
#         if agent_row:
#             agent_rows.append(agent_row)

#         # RAG core
#         context = _collect_context(item)
#         if not context and rag_snippets_k > 0:
#             context = _build_context_from_tool_results(item, rag_snippets_k, snippet_max_chars)
#         if context:
#             rag_rows.append({
#                 **({"query": query} if query is not None else {}),
#                 **({"response": response} if response is not None else {}),
#                 "context": context,
#             })

#         # Document Retrieval
#         dr = _get_doc_gt_and_retrieved(item)
#         if dr:
#             base: Dict[str, Any] = {"query": query} if query is not None else {}
#             base.update(dr)
#             docret_rows.append(base)

#         # Response Completeness
#         gt = _get_ground_truth(item)
#         if gt and response:
#             respcomp_rows.append({
#                 "query": query,
#                 "response": response,
#                 "ground_truth": gt,
#             })

#     outputs: Dict[str, str] = {}

#     if safety_rows:
#         path = f"{out_dir}/safety_security.jsonl"
#         _write_jsonl(path, safety_rows)
#         outputs["safety_security"] = path

#     if agent_rows:
#         path = f"{out_dir}/agent_basic.jsonl"
#         _write_jsonl(path, agent_rows)
#         outputs["agent_basic"] = path

#     if rag_rows:
#         path = f"{out_dir}/rag_core.jsonl"
#         _write_jsonl(path, rag_rows)
#         outputs["rag_core"] = path

#     if docret_rows:
#         path = f"{out_dir}/document_retrieval.jsonl"
#         _write_jsonl(path, docret_rows)
#         outputs["document_retrieval"] = path

#     if respcomp_rows:
#         path = f"{out_dir}/response_completeness.jsonl"
#         _write_jsonl(path, respcomp_rows)
#         outputs["response_completeness"] = path

#     return outputs


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Post-process evaluation JSONL into lean group-specific files")
#     parser.add_argument("--input", required=True, help="Path to evaluation_input_data.jsonl")
#     parser.add_argument("--out_dir", default=".", help="Directory to write output JSONL files")
#     parser.add_argument("--rag_snippets_k", type=int, default=3, help="Max number of tool-result snippets to include as context when none is present (0 to disable)")
#     parser.add_argument("--snippet_max_chars", type=int, default=400, help="Max characters per context snippet")
#     args = parser.parse_args()

#     result = postprocess(args.input, args.out_dir, rag_snippets_k=args.rag_snippets_k, snippet_max_chars=args.snippet_max_chars)
#     print(json.dumps(result, indent=2))

