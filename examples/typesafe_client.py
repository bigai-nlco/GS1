"""Call a running local GS1 service with the official TypeSafe Python SDK."""

import argparse

from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GS1")
    args = parser.parse_args()
    with TypeSafeClient(
        base_url=args.base_url,
        api_key="local",
        model=args.model,
        retry=RetryPolicy(max_retries=0),
        timeout=30.0,
    ) as client:
        result = client.system_one(
            state="A customer was charged twice and requests a refund.",
            questions={
                "team": Choice(
                    instructions="Which team should handle this ticket?",
                    criteria={"billing": "Payments and refunds", "technical": "Software problems"},
                ),
                "refund": Noul(instructions="Does the customer request a refund?"),
                "urgency": Score(
                    instructions="Rate the urgency of handling this ticket.",
                    criteria=["Routine", "Soon", "Immediate"],
                ),
            },
        )
    print(result.choices["team"].choice)
    print(result.choices["team"].probabilities)
    print(result.nouls["refund"].noul)
    print(result.scores["urgency"].score)
    print("Generated tokens:", result.usage.output_tokens)


if __name__ == "__main__":
    main()
