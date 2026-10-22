# LIBRARIES
import sys
import os
import time
import re
import csv
import statistics

# let Python find rag.py inside the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import rag_answer
from test_questions import TEST_SET
############################################################################################################

# GLOBAL VARS DECLARATION
# phrases that show the model refused or said the documents do not cover the question
REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "don't have information",
    "do not mention",
    "does not mention",
    "do not contain",
    "does not contain",
    "no specific information",
    "not explicitly stated",
]
# how many characters of each answer are saved in the csv
ANSWER_CHARS = 600

############################################################################################################
# FUNCTIONS DECLARATION
############################################################################################################

############################################################################################################
# Checks for each answer
# check if the model refused to answer (True = refused)
def check_refusal(answer_text):
    text = answer_text.lower()
    return any(phrase in text for phrase in REFUSAL_PHRASES)

# check if any of the expected PDFs was among the retrieved chunks (True = found)
def check_source_recall(sources, expected_source):
    # unanswerable questions have no expected source, so there is nothing to check
    if expected_source is None:
        return None
    # expected_source can be one file name or a list of file names
    if isinstance(expected_source, str):
        expected_source = [expected_source]
    return any(s["source"] in expected_source for s in sources)

# check if the citations written by the model match the chunks that were retrieved
def check_citation_validity(answer_text, sources):
    # find all citations in the answer like "filename.pdf, p.3"
    cited = re.findall(r"([\w\-. ]+\.pdf),?\s*p\.?\s*(\d+)", answer_text)
    # no citations in the answer, so nothing to check
    if not cited:
        return None
    # the (file, page) pairs that were really given to the model
    valid_sources = {(s["source"], str(s["page"])) for s in sources}
    results = [(src.strip(), page) in valid_sources for src, page in cited]
    # True only if every citation is valid
    return all(results) if results else None

# ask the language model to judge if the answer is supported by the retrieved text
# (True = faithful, False = something was made up)
def simple_faithfulness_check(answer_text, context_chunks, model="llama3.1:8b"):
    import ollama

    # join the retrieved chunks into one text
    context_text = "\n\n".join(c for c, _, _ in context_chunks)
    judge_prompt = f"""Context:
{context_text}

Answer to evaluate:
{answer_text}

Question: Is the answer fully supported by the context above, with no
invented facts? Reply with only one word: YES or NO."""

    resp = ollama.chat(
        model=model, messages=[{"role": "user", "content": judge_prompt}]
    )
    verdict = resp["message"]["content"].strip().upper()
    return verdict.startswith("YES")

############################################################################################################
# Run the evaluation
# go through every test question, run the checks, save the results in a csv file
def run_evaluation():
    rows = []
    for item in TEST_SET:
        # time how long the full answer takes
        start = time.time()
        result = rag_answer(item["question"])
        latency = round(time.time() - start, 2)

        # run the checks on the answer
        refused = check_refusal(result["answer"])
        source_recall = check_source_recall(result["sources"], item["expected_source"])
        citation_valid = check_citation_validity(result["answer"], result["sources"])

        # the faithfulness check needs Ollama, so keep going if it fails
        try:
            faithful = simple_faithfulness_check(result["answer"], result["context_chunks"])
        except Exception as e:
            faithful = f"error: {e}"

        # save the results for this question
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "answer": result["answer"][:ANSWER_CHARS],
                "refused": refused,
                "expected_source": item["expected_source"],
                # files and pages that were retrieved, to check the misses
                "retrieved": "; ".join(f"{s['source']} p.{s['page']}" for s in result["sources"]),
                "source_recall": source_recall,
                "citation_valid": citation_valid,
                "faithful": faithful,
                "latency_sec": latency,
            }
        )
        print(f"[{item['id']}] done ({latency}s) - refused={refused}, faithful={faithful}")

    # write all the results into results.csv in the eval folder
    out_path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"\nFull results saved to {out_path}")

############################################################################################################
# Summary
# print the overall results for all the questions
def print_summary(rows):
    total = len(rows)
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total questions: {total}")

    # questions that can be answered: how many were refused by mistake,
    # and how often the correct PDF was retrieved
    if answerable:
        wrongly_refused = sum(1 for r in answerable if r["refused"])
        print(f"\nAnswerable questions: {len(answerable)}")
        print(f"  Wrongly refused (should've answered): {wrongly_refused} "
              f"({round(100*wrongly_refused/len(answerable),1)}%)")
        recall_checked = [r for r in answerable if r["source_recall"] is not None]
        if recall_checked:
            hit_rate = sum(1 for r in recall_checked if r["source_recall"]) / len(recall_checked)
            print(f"  Source recall@k: {round(100*hit_rate,1)}%")

    # questions that can't be answered: the model should refuse these
    if unanswerable:
        correctly_refused = sum(1 for r in unanswerable if r["refused"])
        print(f"\nUnanswerable (trick) questions: {len(unanswerable)}")
        print(f"  Correctly refused: {correctly_refused}/{len(unanswerable)} "
              f"({round(100*correctly_refused/len(unanswerable),1)}%)")

    # share of answers the judge marked as faithful
    # refused answers and judge errors are skipped, as a refusal has no claims to check
    faithful_checked = [
        r for r in rows if isinstance(r["faithful"], bool) and not r["refused"]
    ]
    if faithful_checked:
        faithful_rate = sum(1 for r in faithful_checked if r["faithful"]) / len(faithful_checked)
        print(f"\nFaithfulness rate (non-refused answers, n={len(faithful_checked)}): "
              f"{round(100*faithful_rate,1)}%")

    # citations that were valid (only for answers that had citations)
    cited = [r for r in rows if isinstance(r["citation_valid"], bool)]
    if cited:
        valid_rate = sum(1 for r in cited if r["citation_valid"]) / len(cited)
        print(f"Citation validity (n={len(cited)}): {round(100*valid_rate,1)}%")

    # latency: the first question also includes the time to load the model
    latencies = [r["latency_sec"] for r in rows]
    print(f"Average latency: {round(sum(latencies)/total,2)}s")
    print(f"Median latency: {round(statistics.median(latencies),2)}s")
    print("=" * 50)

############################################################################################################
# main program
if __name__ == "__main__":
    run_evaluation()