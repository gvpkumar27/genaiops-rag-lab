import json

API_URL = "http://127.0.0.1:8000/chat"


def score(answer: str, expected_keywords: list[str]) -> float:
    a = answer.lower()
    hit = sum(1 for k in expected_keywords if k.lower() in a)
    return hit / max(1, len(expected_keywords))


def main():
    import requests

    total, ssum = 0, 0.0
    with open("eval/golden_qna.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            expected = ex.get("expected_keywords", [])

            r = requests.post(API_URL, json={"question": q}, timeout=300)
            r.raise_for_status()
            ans = r.json()["answer"]

            sc = score(ans, expected)
            print(f"\nQ: {q}\nScore: {sc:.2f}\nAns: {ans[:250]}...")
            total += 1
            ssum += sc

    print(f"\nOK: Avg score: {ssum/max(1,total):.2f} over {total} questions")


if __name__ == "__main__":
    main()
