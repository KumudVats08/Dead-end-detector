"""Research collection and analysis pipeline for Research Dead-End Detector."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
CACHE_DIR = ROOT / "cache"
DEMO_PATH = ROOT / "data" / "demo_records.json"
FIELD_NAMES = ("problem", "input", "method", "output", "dataset_eval", "deployment")
UNKNOWN = "unknown"
_embedding_model: SentenceTransformer | None = None


def _json_from_text(text: str) -> Any:
    """Parse JSON returned by an LLM, accepting a surrounding Markdown fence."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        list_start = cleaned.find("[")
        candidates = [index for index in (start, list_start) if index >= 0]
        if not candidates:
            raise ValueError("LLM response did not contain JSON")
        start = min(candidates)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end < start:
            raise ValueError("LLM response contained incomplete JSON")
        return json.loads(cleaned[start : end + 1])


def _cache_key(kind: str, payload: Any) -> Path:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{kind}_{digest}.json"


def _cached_with_status(
    kind: str, payload: Any, operation: Callable[[], Any]
) -> tuple[Any, bool]:
    path = _cache_key(kind, payload)
    if path.exists():
        with path.open(encoding="utf-8") as cache_file:
            return json.load(cache_file), True

    result = operation()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False
    ) as cache_file:
        json.dump(result, cache_file, ensure_ascii=False)
        temp_path = Path(cache_file.name)
    temp_path.replace(path)
    return result, False


def _cached(kind: str, payload: Any, operation: Callable[[], Any]) -> Any:
    result, _ = _cached_with_status(kind, payload, operation)
    return result


def _llm_endpoint_request(
    prompt: str, base_url: str, api_key: str, model: str, provider: str
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "research-dead-end-detector/1.0",
        },
        method="POST",
    )

    rate_limit_retries = 0
    transient_waits = iter((1, 2))
    while True:
        try:
            with urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except HTTPError as error:
            detail = error.read().decode("utf-8", "ignore")[:300]
            if error.code == 429:
                if rate_limit_retries >= 3:
                    raise RuntimeError(
                        f"{provider} request failed with HTTP 429 after retries: {detail}"
                    ) from error
                rate_limit_retries += 1
                retry_after = error.headers.get("Retry-After")
                wait = _retry_after_seconds(retry_after) if retry_after else None
                if wait is None:
                    wait = 30
            elif error.code >= 500:
                wait = next(transient_waits, None)
                if wait is None:
                    raise RuntimeError(
                        f"{provider} request failed with HTTP {error.code}: {detail}"
                    ) from error
            else:
                raise RuntimeError(
                    f"{provider} request failed with HTTP {error.code}: {detail}"
                ) from error
            time.sleep(wait)
        except URLError as error:
            wait = next(transient_waits, None)
            if wait is None:
                raise RuntimeError(f"{provider} request failed: {error.reason}") from error
            time.sleep(wait)


def _retry_after_seconds(value: str) -> float | None:
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _llm_request(prompt: str) -> str:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key == "your_openai_compatible_api_key":
        raise RuntimeError("Set a real LLM_API_KEY in .env or your environment")
    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    try:
        return _llm_endpoint_request(prompt, base_url, api_key, model, "Primary LLM")
    except RuntimeError as primary_error:
        fallback_url = os.getenv("LLM_FALLBACK_BASE_URL")
        fallback_key = os.getenv("LLM_FALLBACK_KEY")
        fallback_model = os.getenv("LLM_FALLBACK_MODEL")
        if not all((fallback_url, fallback_key, fallback_model)):
            raise
        try:
            return _llm_endpoint_request(
                prompt, fallback_url, fallback_key, fallback_model, "Fallback LLM"
            )
        except RuntimeError as fallback_error:
            raise RuntimeError(
                f"Primary and fallback LLM requests failed. "
                f"Primary: {primary_error}. Fallback: {fallback_error}"
            ) from fallback_error


def llm(prompt: str) -> str:
    """Call the configured OpenAI-compatible LLM, caching the response."""
    return _cached("llm", {"prompt": prompt}, lambda: _llm_request(prompt))


def _llm_with_cache_status(prompt: str) -> tuple[str, bool]:
    response, was_cached = _cached_with_status(
        "llm", {"prompt": prompt}, lambda: _llm_request(prompt)
    )
    return response, was_cached


def gen_queries(idea: str) -> list[dict[str, str]]:
    prompt = f"""Create exactly five distinct web search queries for this project idea:
{idea}

Return only a JSON array of five objects, each with "query" and "purpose".
Include at least one general query, one query containing site:arxiv.org, and
one containing site:github.com. Make one object a broad research-paper query;
that object must also contain "domain_type": "research_paper". Do not add
domain_type to the other objects."""
    parsed = _json_from_text(llm(prompt))
    if not isinstance(parsed, list) or len(parsed) != 5:
        raise ValueError("Query generation must return exactly five query objects")

    queries: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError("Each generated query must include a query string")
        query = {
            "query": item["query"].strip(),
            "purpose": str(item.get("purpose", "Related prior work")),
        }
        if item.get("domain_type") == "research_paper":
            query["domain_type"] = "research_paper"
        queries.append(query)
    query_text = " ".join(item["query"] for item in queries).lower()
    if "site:arxiv.org" not in query_text or "site:github.com" not in query_text:
        raise ValueError("Generated queries must include arXiv and GitHub searches")
    if not any(
        "site:" not in item["query"].lower()
        and "domain_type" not in item
        for item in queries
    ):
        raise ValueError("Generated queries must include a general web search")
    if not any(item.get("domain_type") == "research_paper" for item in queries):
        raise ValueError("A query must use domain_type=research_paper")
    return queries


def _sdk_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    attributes = (
        "results",
        "errors",
        "title",
        "url",
        "text",
        "snippet",
        "site_name",
        "error",
    )
    fields = {name: getattr(value, name) for name in attributes if hasattr(value, name)}
    if fields:
        return fields
    return value


def _tinyfish_call(operation: Callable[[], Any]) -> Any:
    for attempt in range(3):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(error, "status_code", None) or getattr(
                response, "status_code", None
            )
            if status != 429 or attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("TinyFish request failed after retries")


def _search(query: dict[str, str]) -> list[dict[str, Any]]:
    def call() -> list[dict[str, Any]]:
        api_key = os.getenv("TINYFISH_API_KEY")
        if not api_key or api_key == "your_tinyfish_api_key":
            raise RuntimeError("Set a real TINYFISH_API_KEY in .env or your environment")
        from tinyfish import TinyFish

        client = TinyFish()
        kwargs: dict[str, str] = {}
        if query.get("domain_type"):
            kwargs["domain_type"] = query["domain_type"]
        if query.get("purpose"):
            kwargs["purpose"] = query["purpose"]
        response = _sdk_value(
            client.search.query(query=query["query"], **kwargs)
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        return [_sdk_value(result) for result in results]

    return _cached("tinyfish_search", query, lambda: _tinyfish_call(call))


def _fetch(urls: list[str], retry_attempt: int = 0) -> dict[str, Any]:
    def call() -> dict[str, Any]:
        api_key = os.getenv("TINYFISH_API_KEY")
        if not api_key or api_key == "your_tinyfish_api_key":
            raise RuntimeError("Set a real TINYFISH_API_KEY in .env or your environment")
        from tinyfish import TinyFish

        response = _sdk_value(TinyFish().fetch.get_contents(urls=urls))
        if not isinstance(response, dict):
            raise ValueError("TinyFish fetch returned an invalid response")
        return response

    return _cached(
        "tinyfish_fetch",
        {"urls": urls, "retry_attempt": retry_attempt},
        lambda: _tinyfish_call(call),
    )


def _source_type(url: str) -> str:
    lowered = url.lower()
    if "arxiv.org" in lowered or "doi.org" in lowered or "scholar.google" in lowered:
        return "paper"
    if "github.com" in lowered or "gitlab.com" in lowered:
        return "github"
    return "web"


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def _url_priority(url: str) -> int | None:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    excluded_domains = (
        "researchgate.net",
        "linkedin.com",
        "medium.com",
        "ieee.org",
        "springer.com",
        "sciencedirect.com",
        "wiley.com",
    )
    if any(_domain_matches(host, domain) for domain in excluded_domains):
        return None
    if _domain_matches(host, "arxiv.org"):
        return 0
    if _domain_matches(host, "github.com"):
        return 1
    return 2


def collect(
    queries: list[dict[str, str]],
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Search, deduplicate, and fetch up to eight pages."""
    if progress_callback:
        progress_callback("Searching")
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        for result in _search(query):
            url = result.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            priority = _url_priority(url)
            if priority is None:
                continue
            found.append(
                {
                    "title": result.get("title") or url,
                    "url": url,
                    "snippet": result.get("snippet") or "",
                    "site_name": result.get("site_name") or "",
                    "source_type": _source_type(url),
                    "_url_priority": priority,
                }
            )
    found.sort(key=lambda record: record["_url_priority"])
    found = found[:15]
    if not found:
        raise RuntimeError("TinyFish search returned no usable results")

    if progress_callback:
        progress_callback("Fetching")
    pages: dict[str, dict[str, Any]] = {}
    fetch_error_by_url: dict[str, str] = {}
    for offset in range(0, len(found), 10):
        batch = found[offset : offset + 10]
        response = _fetch([record["url"] for record in batch])
        for page in response.get("results", []):
            page = _sdk_value(page)
            if isinstance(page, dict) and page.get("url"):
                pages[page["url"]] = page
                fetch_error_by_url.pop(page["url"], None)
        errors = [_sdk_value(error) for error in response.get("errors", [])]
        for error in errors:
            if (
                isinstance(error, dict)
                and error.get("url")
                and str(error["url"]) not in pages
            ):
                fetch_error_by_url[str(error["url"])] = str(
                    error.get("error", "fetch failed")
                )

    failed_records = [record for record in found if record["url"] not in pages]
    for offset in range(0, len(failed_records), 10):
        batch = failed_records[offset : offset + 10]
        response = _fetch([record["url"] for record in batch], retry_attempt=1)
        for page in response.get("results", []):
            page = _sdk_value(page)
            if isinstance(page, dict) and page.get("url"):
                pages[page["url"]] = page
                fetch_error_by_url.pop(page["url"], None)
        errors = [_sdk_value(error) for error in response.get("errors", [])]
        for error in errors:
            if (
                isinstance(error, dict)
                and error.get("url")
                and str(error["url"]) not in pages
            ):
                fetch_error_by_url[str(error["url"])] = str(
                    error.get("error", "fetch failed")
                )
        for record in batch:
            if record["url"] not in pages and record["url"] not in fetch_error_by_url:
                fetch_error_by_url[record["url"]] = "fetch failed"

    records = []
    for record in found:
        page = pages.get(record["url"])
        if not page:
            continue
        record["text"] = str(page.get("text") or "")[:3000]
        record["title"] = page.get("title") or record["title"]
        record.pop("_url_priority", None)
        records.append(record)
    records = records[:8]
    fetch_errors = [
        {"url": record["url"], "error": fetch_error_by_url[record["url"]]}
        for record in found
        if record["url"] in fetch_error_by_url
    ]
    if not records:
        if fetch_errors:
            failures = "; ".join(
                f"{item['url']}: {item['error']}" for item in fetch_errors[:5]
            )
            raise RuntimeError(f"TinyFish fetched no pages successfully: {failures}")
        raise RuntimeError("TinyFish fetched no page contents")
    records[0]["_fetch_errors"] = fetch_errors
    return records


def _clean_extraction(parsed: Any, text: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Extraction response must be a JSON object")
    evidence = parsed.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    cleaned: dict[str, Any] = {}
    cleaned_evidence: dict[str, str] = {}
    for field in FIELD_NAMES:
        value = parsed.get(field, UNKNOWN)
        quote = evidence.get(field, "")
        if not isinstance(value, str) or not value.strip():
            value = UNKNOWN
        value = value.strip()
        if value.lower() == UNKNOWN:
            value = UNKNOWN
        if not isinstance(quote, str) or not quote.strip() or quote.strip() not in text:
            if value != UNKNOWN:
                value = UNKNOWN
            cleaned_evidence[field] = ""
        else:
            cleaned_evidence[field] = quote.strip()
        cleaned[field] = value
    cleaned["evidence"] = cleaned_evidence
    return cleaned


def _extract_prompt(record: dict[str, Any]) -> str:
    return f"""Extract facts only from the source text below. Return only strict JSON
with exactly these string fields: problem, input, method, output, dataset_eval,
deployment, and an evidence object with the same six fields. Each evidence value
must be a short verbatim quote copied exactly from the source text. For any fact
not explicitly stated, set its field to "unknown" and its evidence to "".
Never infer or fill gaps from general knowledge.

Source: {record.get("title", "")}
URL: {record.get("url", "")}
Source text:
{record.get("text", "")}"""


def _extract(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    prompt = _extract_prompt(record)
    response, was_cached = _llm_with_cache_status(prompt)
    return _clean_extraction(
        _json_from_text(response), str(record.get("text", ""))
    ), was_cached


def extract_records(
    records: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if progress_callback:
        progress_callback("Extracting")
    previous_request_was_live = False
    for index, record in enumerate(records):
        prompt = _extract_prompt(record)
        cached_response = _cache_key("llm", {"prompt": prompt}).exists()
        if index and previous_request_was_live and not cached_response:
            time.sleep(15)
        extraction, was_cached = _extract(record)
        record.update(extraction)
        if not was_cached:
            previous_request_was_live = True
    return records


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def rank(
    idea: str,
    records: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if progress_callback:
        progress_callback("Ranking")
    if not records:
        return []
    model = _get_embedding_model()
    descriptions = [
        f"{record.get('problem', '')} {record.get('method', '')}"
        for record in records
    ]
    embeddings = model.encode(
        [idea, *descriptions], convert_to_tensor=True, normalize_embeddings=True
    )
    scored = []
    for index, record in enumerate(records):
        item = dict(record)
        item["score"] = float((embeddings[0] * embeddings[index + 1]).sum().item())
        scored.append(item)
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:8]


def _landscape(records: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    methods: dict[str, int] = {}
    for record in records:
        method = record.get("method", UNKNOWN)
        if isinstance(method, str) and method.lower() != UNKNOWN:
            label = method.strip()
            methods[label] = methods.get(label, 0) + 1
    methods = dict(sorted(methods.items(), key=lambda item: item[1], reverse=True)[:8])
    blind_spots = []
    for field in FIELD_NAMES:
        unknown_count = sum(
            not isinstance(record.get(field), str)
            or record.get(field, UNKNOWN).strip().lower() in ("", UNKNOWN)
            for record in records
        )
        if records and unknown_count > len(records) / 2:
            blind_spots.append(field)
    return methods, blind_spots


def analyze(
    idea: str,
    top: list[dict[str, Any]],
    demo_hypotheses: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback:
        progress_callback("Analyzing")
    methods, blind_spots = _landscape(top)
    if demo_hypotheses is not None:
        hypotheses = demo_hypotheses
    else:
        sources = [
            {
                "title": record.get("title", ""),
                "url": record.get("url", ""),
                "problem": record.get("problem", UNKNOWN),
                "method": record.get("method", UNKNOWN),
                "dataset_eval": record.get("dataset_eval", UNKNOWN),
                "deployment": record.get("deployment", UNKNOWN),
            }
            for record in top
        ]
        prompt = f"""Assess this idea only within the supplied sources:
{idea}

Computed method counts: {json.dumps(methods)}
Fields mostly unknown: {json.dumps(blind_spots)}
Sources: {json.dumps(sources, ensure_ascii=False)}

Return only JSON with a "hypotheses" array of exactly two possible research gaps.
Each item has "hypothesis" (cautious wording such as "within the sources
searched"), and "source_urls" (one or more URLs copied from the supplied source
list). Do not claim the idea is original, novel, or that no one has done it.
Treat missing information only as a gap in these sources, not proof it does not
exist."""
        parsed = _json_from_text(llm(prompt))
        hypotheses = parsed.get("hypotheses") if isinstance(parsed, dict) else None
        valid_urls = {record.get("url") for record in top}
        if not isinstance(hypotheses, list) or len(hypotheses) != 2:
            raise ValueError("Gap analysis must return exactly two hypotheses")
        cleaned_hypotheses = []
        for item in hypotheses:
            if not isinstance(item, dict) or not isinstance(item.get("hypothesis"), str):
                raise ValueError("Each gap hypothesis must have a description")
            urls = [
                url
                for url in item.get("source_urls", [])
                if isinstance(url, str) and url in valid_urls
            ]
            if not urls:
                raise ValueError("Each gap hypothesis must cite a supplied source URL")
            cleaned_hypotheses.append(
                {"hypothesis": item["hypothesis"].strip(), "source_urls": urls}
            )
        hypotheses = cleaned_hypotheses
    return {
        "method_counts": methods,
        "blind_spots": blind_spots,
        "hypotheses": hypotheses,
    }


def load_demo_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with DEMO_PATH.open(encoding="utf-8") as data_file:
        demo = json.load(data_file)
    return demo["records"], demo["possible_gaps"]


def run_pipeline(
    idea: str,
    progress_callback: Callable[[str], None] | None = None,
    use_demo: bool = False,
) -> dict[str, Any]:
    if use_demo or os.getenv("DEMO") == "1":
        records, demo_gaps = load_demo_data()
        top = rank(idea, records, progress_callback)
        analysis = analyze(
            idea, top, demo_hypotheses=demo_gaps, progress_callback=progress_callback
        )
        return {
            "records": top,
            "analysis": analysis,
            "demo": True,
            "queries": [],
            "urls_fetched": 0,
            "fetched_urls": [],
        }

    queries = gen_queries(idea)
    records = collect(queries, progress_callback)
    extracted = extract_records(records, progress_callback)
    fetch_errors = extracted[0].pop("_fetch_errors", [])
    top = rank(idea, extracted, progress_callback)
    analysis = analyze(idea, top, progress_callback=progress_callback)
    return {
        "records": top,
        "analysis": analysis,
        "demo": False,
        "queries": queries,
        "urls_fetched": len(extracted),
        "fetched_urls": [record["url"] for record in extracted],
        "fetch_errors": fetch_errors,
    }
