"""Small offline smoke test that ranks the bundled demo records."""

from pipeline import run_pipeline


IDEA = "offline crop disease detection from leaf photos"


def main() -> None:
    result = run_pipeline(IDEA, use_demo=True)
    print("Top 3 similar prior-work records:")
    for record in result["records"][:3]:
        print(f"- {record['title']} ({record['score']:.3f}) — {record['url']}")


if __name__ == "__main__":
    main()
