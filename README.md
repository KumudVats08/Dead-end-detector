# Research Dead-End Detector

Research Dead-End Detector helps students, researchers, and early-stage teams
get an evidence-led view of what may already exist around a project idea. Enter
an idea in plain language, such as **“offline crop disease detection from leaf
photos,”** and the app searches for related papers, repositories, and other
open web sources.

It is a starting point for scoping and refining a project and not a patent search,
systematic literature review, or guarantee that an idea is novel.

## What you get from an idea

After you submit an idea, the app:

1. Generates five focused search queries, including searches for arXiv papers
   and GitHub projects.
2. Searches with TinyFish, removes duplicate results, and prioritizes arXiv,
   GitHub, and other open pages. Results from ResearchGate, LinkedIn, Medium,
   and selected paywalled publishers are excluded.
3. Fetches up to 15 candidate URLs. If a URL fails, it retries that URL once.
   From the successfully fetched pages, it keeps at most eight for analysis.
4. Extracts source details and ranks the records by local embedding similarity
   to your idea.
5. Summarizes repeated methods and fields that are often unspecified in this
   small source set, then suggests two cautious, source-linked research-gap
   hypotheses.

### Results on the page

- **Similar prior work:** Up to eight ranked result cards with each source's
  title, link, type (`paper`, `github`, or `web`), and similarity score. The
  score is cosine similarity between an embedding of your idea and embeddings
  of each source's extracted problem and method. It is a relative ranking aid,
  not a probability or quality score.
- **Source details and evidence:** Each card and the comparison table show the
  stated problem, input, method, output, dataset/evaluation, and deployment
  information. Evidence is shown as a short verbatim quote when one can be
  verified in the fetched text. If the source does not explicitly state a
  field—or its evidence cannot be verified—the field is shown as `unknown`.
- **Method comparison:** A table puts the ranked records side by side so you
  can spot similarities and differences across their methods and study setup.
- **Crowded areas:** A count of repeated method descriptions found in the
  results. These counts reflect how methods were described in this small
  result set; they are not a field-wide prevalence measure.
- **Possible gaps:** Two hypotheses about what appears underexplored or
  underreported within the sources searched, each linked to relevant results.
  Treat these as ideas for further investigation, not proof that the work has
  never been done.
- **Live integration log:** The sidebar shows the generated queries, fetched
  URLs, failed fetch count, and number of records analyzed.

The app displays **“Based on limited sources searched; not a novelty
guarantee.”** Search coverage and page availability are limited; verify
important findings in the original sources and do a broader literature search
before deciding a project is novel.

## Run locally

Requires Python 3.11.

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Configure credentials in a project-root `.env` file:

   - `TINYFISH_API_KEY` for web search and page fetching.
   - `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` for an
     OpenAI-compatible LLM. The defaults in the application target Gemini;
     use a key and model supported by the configured provider.
   - Optionally, set `LLM_FALLBACK_BASE_URL`, `LLM_FALLBACK_KEY`, and
     `LLM_FALLBACK_MODEL` to enable a backup LLM provider.

   The app loads `.env` from the project folder. Put real credentials in your
   local `.env` only; never commit or share the file.

3. Start Streamlit:

   ```powershell
   streamlit run app.py
   ```

4. Enter a project idea and select **Find similar work**.

### Demo mode

To use the bundled crop-disease records without live TinyFish/LLM search:

```powershell
$env:DEMO = "1"
streamlit run app.py
```

If live research fails, the UI also falls back to the bundled demo dataset and
shows a warning. Demo records and their summaries are illustrative; verify
claims against linked sources before relying on them.

### Smoke test

Run the offline ranking smoke test:

```powershell
python test_pipeline.py
```

The first ranking run downloads the local `all-MiniLM-L6-v2` embedding model.
TinyFish and LLM responses are cached as JSON under `cache/` to avoid repeating
identical calls.

## How TinyFish is used

TinyFish receives five generated search queries. Its search results are
deduplicated and prioritized, then up to 15 candidate URLs are sent to
`TinyFish.fetch.get_contents` in batches of at most ten. Failed URLs are
retried once. Fetch errors do not discard other successfully fetched pages.
The pipeline analyzes at most eight successful pages and truncates each page
to 3,000 characters before extraction.

LLM extraction is sequential and processes one page per call, with about
15 seconds between uncached extraction calls to reduce rate-limit pressure.
LLM responses are cached. HTTP 429 responses honor `Retry-After` when supplied;
otherwise they wait 30 seconds, retrying up to three times. If configured, the
backup LLM provider is tried when the primary request fails.

## Pricing and revenue concept

A possible model is a free student tier for individual project exploration and
a paid lab/incubator tier with shared workspaces, saved reports, and team usage
controls. As an illustrative sizing scenario—not a verified market estimate—
10,000 labs or incubators paying $1,000 annually would represent $10 million
in annual contract value. The customer count and willingness to pay would need
independent validation.
