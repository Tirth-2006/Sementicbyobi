"""Audit prompt/answer token boundary crossings without model inference."""
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


INPUT_PATH = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
OUTPUT_PATH = Path("data/generated/token_boundary_audit.json")
MODEL_DIR = Path(".")


def build_prompt(few_shot_prompt: str, question: str) -> str:
    """Reconstruct the original raw-text query prompt."""
    return f"{few_shot_prompt}Question: {question}\nAnswer:"


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    generations = data.get("generations")
    few_shot_prompt = data.get("metadata", {}).get("few_shot_prompt")
    if not isinstance(generations, dict) or not isinstance(few_shot_prompt, str):
        raise ValueError("Generation JSON is missing the required saved fields")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    crossing_cases: list[dict[str, Any]] = []
    first_scored_before_boundary: list[dict[str, Any]] = []
    answer_count = 0
    answer_token_count = 0

    for question_id, record in generations.items():
        question = record.get("question")
        responses = record.get("high_temp_generations", {}).get("responses")
        if not isinstance(question, str) or not isinstance(responses, list):
            raise ValueError(f"Invalid generation record for {question_id!r}")

        prompt = build_prompt(few_shot_prompt, question)
        boundary = len(prompt)
        for answer_index, response in enumerate(responses):
            answer = response.get("response") if isinstance(response, dict) else None
            if not isinstance(answer, str):
                raise ValueError(
                    f"Invalid answer for {question_id!r}:{answer_index}"
                )
            answer_count += 1
            encoded = tokenizer(
                prompt + answer,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            token_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            overlapping_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and start < boundary + len(answer) and end > boundary
            ]
            answer_token_count += len(overlapping_positions)
            for position, (start, end) in enumerate(offsets):
                if start < boundary < end:
                    token_id = int(token_ids[position])
                    token_text = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                    )
                    crossing_cases.append(
                        {
                            "question_id": question_id,
                            "answer_index": answer_index,
                            "token_position": position,
                            "token_id": token_id,
                            "token_text": token_text,
                            "character_offsets": [start, end],
                            "prompt_length": boundary,
                            "answer_text": answer,
                        }
                    )

            if overlapping_positions:
                first_position = overlapping_positions[0]
                first_start, first_end = offsets[first_position]
                if first_start < boundary:
                    token_id = int(token_ids[first_position])
                    first_scored_before_boundary.append(
                        {
                            "question_id": question_id,
                            "answer_index": answer_index,
                            "token_position": first_position,
                            "token_id": token_id,
                            "token_text": tokenizer.decode(
                                [token_id],
                                skip_special_tokens=False,
                            ),
                            "character_offsets": [first_start, first_end],
                            "prompt_length": boundary,
                        }
                    )

    output = {
        "metadata": {
            "input_file": str(INPUT_PATH),
            "model_dir": str(MODEL_DIR),
            "tokenizer_call": "add_special_tokens=True, return_offsets_mapping=True",
            "model_inference_rerun": False,
            "scoring_results_modified": False,
        },
        "summary": {
            "answers_audited": answer_count,
            "answers_with_boundary_crossing_token": len(
                {(
                    case["question_id"],
                    case["answer_index"],
                ) for case in crossing_cases}
            ),
            "boundary_crossing_token_count": len(crossing_cases),
            "answers_with_first_scored_token_before_boundary": len(
                first_scored_before_boundary
            ),
            "overlapping_answer_token_count": answer_token_count,
        },
        "boundary_crossing_cases": crossing_cases,
        "first_scored_token_before_boundary_cases": first_scored_before_boundary,
        "interpretation": {
            "current_behavior": (
                "The scorer includes every token whose offset overlaps the answer "
                "character span, so a crossing token is scored in full."
            ),
            "implication": (
                "When a token spans the prompt/answer boundary, its conditional "
                "probability is for the whole token, although part of its decoded "
                "text belongs to the prompt."
            ),
            "recommendation": (
                "Keep the existing results unchanged. For a new primary analysis, "
                "use full-token scoring with the reconstructed prompt boundary and "
                "report boundary crossings explicitly; do not silently drop or "
                "rewrite crossing tokens. A character-exact alternative would "
                "require a documented retokenized prompt convention."
            ),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print(f"Answers audited: {answer_count}")
    print(
        "Answers with boundary-crossing token: "
        f"{output['summary']['answers_with_boundary_crossing_token']}"
    )
    print(
        "Boundary-crossing tokens: "
        f"{output['summary']['boundary_crossing_token_count']}"
    )
    print(
        "First scored token starts before boundary: "
        f"{output['summary']['answers_with_first_scored_token_before_boundary']}"
    )
    print(f"Saved audit to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
