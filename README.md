# ConfoundSearch

Organization Science proposal: ConfoundSearch

See our submission documents for more.

![coverage](devops/badges/coverage.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

![The interface: candidate confounds of organizational structure and firm performance, the diagram, and the papers behind one link](assets/example.png)

*The interface on known confounds for organizational structure and firm performance. For an interactive demo, download
[`assets/known_confounds.html`](assets/known_confounds.html) and open it in a browser.*

## Quick start

```bash
pip install confoundsearch        # 1. install (Python 3.8+)
confoundsearch setup              # 2. save your keys in a .env file (language model, Semantic Scholar)
confoundsearch demo               # 3. the happy path: a short run on papers that ship with the package, then opens it
confoundsearch install-skill      # 4. optional: add the Claude skill

# A run can be slow; we are working to reduce latency. See below.
# 5A. in Claude Code:  /confoundsearch iv="Organizational structure" dv="Firm performance"
# 5B. or run the package directly (it searches every source you have set up):
confoundsearch run --model qwen/qwen3-max \
  --iv "Organizational structure" --dv "Firm performance" \
  --max-budget 1.00 --out results/org.html

confoundsearch review --in results/org.html     # 6. open the interface: the confounds as a causal graph
```

`demo` needs only a language model key (no searching), costs a few cents and takes a couple of minutes. We suggest running
longer jobs in a tmux/screen session or side terminal. See the technical appendix for more.

## Keys

`confoundsearch setup` asks for your keys and keeps them in a private `.env` file in the folder you run from (it is added to
`.gitignore`, and a key is only ever sent to the service it belongs to). Why each one:

- **Language model** (`OPENROUTER_API_KEY` or `OPENAI_API_KEY`): required. The agent reads each paper's abstract and
  judges whether it supports a causal claim, so every run makes model calls. Any OpenAI-compatible model works (`--model`).
- **OpenAI** (`OPENAI_API_KEY`): also set this, even if the agent uses another model. Searching the keyword vectors for a
  phrase that is not already in the pre-indexed set embeds that phrase with an OpenAI embedding model.
- **Semantic Scholar** (`S2_KEY`, free from semanticscholar.org/product/api): optional but worth the two minutes. Without a
  key you share a rate limit with every anonymous user, searches get refused (HTTP 429) and runs slow down or stop early.
- **OpenAlex** (`OPENALEX_EMAIL` or `OPENALEX_API_KEY`): optional; an email puts you in OpenAlex's faster, more reliable pool.

A run usually costs less than $1 in model fees; the searches are free. Cap it with `--max-budget USD`: the run stops once the
estimated model cost reaches that, and keeps what it found so far.

## Data directory

ConfoundSearch downloads papers as it searches and keeps them in a data directory, the way `transformers` keeps models in
`HF_HOME`. It holds one small SQLite database per question (independent and dependent variable), so the same search is never
made twice: reruns are fast, use no rate limit, and return the same papers. Each paper is stored once, however many sources
found it (same title, fuller abstract kept), and searches never wait for the disk.

By default it is a folder in your system's temporary directory (`/tmp/confoundsearch-<you>` on Linux), which the system
clears from time to time, so nothing there is precious. To keep it, set `CONFOUNDSEARCH_CACHE=~/confoundsearch-data`
(`confoundsearch setup` asks, and saves it in `.env`) or pass `--cache-dir`. To empty it yourself run `confoundsearch flush`
(everything) or `confoundsearch flush --iv "..." --dv "..."` (one question). `--no-cache` skips it for one run.

## What's in this code base?

The application is divided into four layers, as described in the technical appendix. Each is one directory:

- **Data** ([`confoundsearch/data/`](confoundsearch/data)): where papers come from (hosted Elasticsearch, Semantic Scholar, OpenAlex, or your file).
- **Logic** ([`confoundsearch/logic/`](confoundsearch/logic)): the Markov decision process (graph, moves, reward, policy) and the audit log.
- **Language model** ([`confoundsearch/model/`](confoundsearch/model)): any OpenAI-compatible model, including OpenRouter.
- **Interface** ([`confoundsearch/interface/`](confoundsearch/interface)): the web app, command line, JSON output and agent skill.

## Other key directories

- [`tests/`](tests): the unit tests, with 100% branch coverage. They also enforce the layer boundaries described in the technical appendix.
- [`devops/`](devops): scripts for releasing, checking the package on every Python version, and publishing the anonymous review copy.
- [`evals/`](evals): code and data for the pilot studies described in the technical appendix.
- [`confoundsearch/data/elastic/`](confoundsearch/data/elastic): loads the corpus into Elasticsearch Serverless.
