import json
import os
from pathlib import Path
from pprint import pprint

from azure.ai.evaluation import evaluate


def write_placeholder_dataset(file_path: Path) -> None:
    """
    Write a minimal JSONL with placeholders for Response Completeness.
    Also includes optional fields for QA and Retrieval evaluators so you can
    enable them later by just filling values.
    """
    rows = [
        {
            # ResponseCompleteness
            "response": "<MODEL_RESPONSE_GOES_HERE>",
            "ground_truth": "<REFERENCE_ANSWER_OR_KEY_POINTS_GO_HERE>",
            # Or provide a checklist instead of a full reference:
            # "expected_points": ["<POINT_A>", "<POINT_B>", "<POINT_C>"],

            # QA (optional)
            "question": "<QUESTION_IF_USING_QA_EVALUATOR>",
            "answer": "<MODEL_ANSWER_IF_USING_QA_EVALUATOR>",
            "reference": "<GOLD_ANSWER_IF_USING_QA_EVALUATOR>",

            # Retrieval/DocumentRetrieval (optional)
            "query": "<USER_QUERY_IF_USING_RETRIEVAL_EVALUATORS>",
            "retrieved_documents": [
                {"id": "doc1", "text": "<RETRIEVED_DOC_TEXT_1>", "score": 1.0},
                {"id": "doc2", "text": "<RETRIEVED_DOC_TEXT_2>", "score": 0.9},
            ],
            "gold_documents": [
                {"id": "doc1"}
            ],
            "citations": [
                {"doc_id": "doc1"}
            ],
        }
    ]
    with file_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


essential_evaluators = {
    # Built-in name for Response Completeness
    "response_completeness": "response_completeness",
    # Enable these later once you fill corresponding fields
    # "qa": "qa",
    # "retrieval": "retrieval",
    # "document_retrieval": "document_retrieval",
}


def main() -> None:
    file_path = Path("response_completeness_data.jsonl")
    if not file_path.exists():
        write_placeholder_dataset(file_path)
        print(
            f"Wrote placeholder dataset to {file_path.resolve()}\n"
            "Fill in the placeholders and re-run."
        )

    azure_ai_project = os.environ.get("AZURE_AI_PROJECT")
    if not azure_ai_project:
        raise RuntimeError(
            "AZURE_AI_PROJECT is not set. Set it to your Foundry project endpoint and re-run."
        )

    response = evaluate(
        data=str(file_path),
        evaluators=essential_evaluators,
        azure_ai_project=azure_ai_project,
    )

    studio_url = response.get("studio_url")
    if studio_url:
        pprint({"AI Foundry URL": studio_url})
    else:
        print("Evaluation completed. No Studio URL returned.")


if __name__ == "__main__":
    main()
