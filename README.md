# ConfoundSearch

Supporting code for our Organization Science submission: ConfoundSearch. Please see our submission documents for more.

![coverage](devops/badges/coverage.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

![The interface: candidate confounds of organizational structure and firm performance, the diagram, and the papers behind one link](assets/example.png)

*The interface showing possible confounds for organizational structure and firm performance. For an interactive demo, download
[`assets/known_confounds.html`](assets/known_confounds.html) and open it in a browser.*

## Quick start

```bash
pip install confoundsearch        # 1. install (Python 3.8+)
confoundsearch demo               # 2. run a demo
```

ConfoundSearch has latency and requires API keys. The demo gives you a sense of how it works before a full setup. Running `confoundsearch demo` walks through a full run from start to finish and opens the result, for the independent variable `organizational structure` and the dependent variable `firm performance`. 
The demo runs in about a minute without any setup. We recommend trying out the demo to start and then proceeding with a full setup.

### Full setup

When you are ready to run your own queries:

```bash
confoundsearch setup              # 3. save your keys in a .env file (language model, Semantic Scholar). See below for more.
confoundsearch install-skill      # 4. optional: add the Claude skill

# A run can be slow; we are working to reduce latency. We suggest a tmux/screen session or side terminal.
# 5A. in Claude Code:  /confoundsearch iv="Organizational structure" dv="Firm performance"
# 5B. or run the package directly.
# The command below uses a default OpenAI model. You can also use an open-weight model via OpenRouter with
# the --model parameter, or run with your own self-hosted open-weight model (see below).
confoundsearch run \
  --iv "Organizational structure" --dv "Firm performance" \
  --max-budget 1.00 --out results/org.html

confoundsearch review --in results/org.html     # 6. open the interface: the confounds as a causal graph
```

## How to setup API keys

`confoundsearch setup` asks for API keys and saves them in a file called `.env` in the folder where you run the code (if `.env` files are
unfamiliar, [this page](https://github.com/theskumar/python-dotenv#readme) explains the idea). Your keys stay on your computer. Some keys are required; others are strongly recommended because they make things run faster.

### Required keys

| Variable name | API service / how to sign up | Why |
| --- | --- | --- |
| `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | <ul><li>[OpenRouter](https://openrouter.ai/keys) sells access to hundreds of models from many providers behind one key</li><li>[OpenAI](https://platform.openai.com/api-keys) offers API access to its LLMs. [Here is how to get started with the OpenAI API](https://platform.openai.com/docs/quickstart) if that is unfamiliar.</li></ul> | The agent reads each paper's abstract and judges whether it supports a causal claim, so every run makes model calls. It also synthesizes evidence into a causal graph. |

You need one language-model key to run your own questions. Technical users who want to run a locally hosted model should set `OPENAI_BASE_URL` to its address (for example `http://localhost:8000/v1`, or pass `--base-url`); a local model needs no key.

### Strongly recommended keys

| Variable name | API service / how to sign up | Why |
| --- | --- | --- |
| `S2_KEY` | [Semantic Scholar](https://www.semanticscholar.org/product/api) is a free academic search index run by the Allen Institute for AI; the key is free | Everything runs faster. Without a key you share a rate limit with every anonymous user, so searches get refused (HTTP 429) and runs slow down or stop early. |
| `OPENALEX_API_KEY` or `OPENALEX_EMAIL` | [OpenAlex](https://openalex.org/settings/api) is a free, open index of scholarly papers run by the nonprofit OurResearch; the key needs a free account | Runs faster. A key or an email puts you in OpenAlex's faster, more reliable search tier, so searches finish sooner and fewer fail. |
| `OPENAI_API_KEY` (for embeddings) | The same [OpenAI key](https://platform.openai.com/api-keys) as above | ConfoundSearch uses OpenAI embeddings for [query expansion](https://en.wikipedia.org/wiki/Query_expansion), among other methods. If you set this key, it makes a (cheap) call to the OpenAI embeddings API to get vectors for your keywords, to find similar query terms. This is optional but helpful. Set it even if the agent uses a different model. |

Most ConfoundSearch runs cost a few cents. If you are concerned, `--max-budget USD` stops at a given dollar limit.

## Data directory

As it searches, ConfoundSearch downloads papers and saves them in a *data directory*, a folder on your computer,
so the same search is never made twice. This is a common
pattern. For example, [Transformers](https://huggingface.co/docs/transformers) keeps models in a [cache folder](https://huggingface.co/docs/hub/local-cache) set by `HF_HOME`.
For more on the caching system, see [the code](confoundsearch/data/cache.py).

**Where the cache is installed.** By default the data directory lives in your [temporary folder](https://en.wikipedia.org/wiki/Temporary_folder), which your
computer automatically clears from time to time. The cache can get flushed periodically and the tool will still run fine.

**If you want a permanent cache.** If you don't want your cache to get flushed, set the [environment variable](https://en.wikipedia.org/wiki/Environment_variable) `CONFOUNDSEARCH_CACHE`, for example
`CONFOUNDSEARCH_CACHE=~/confoundsearch-data`. `confoundsearch setup` asks for it and saves it in `.env`; `--cache-dir` sets it for one run.

**If you need space and want to clear your cache.** Run `confoundsearch flush` (everything) or `confoundsearch flush --iv "..." --dv "..."` (for a given input pair).

**To keep a permanent record of a search.** Use a `--out` flag to save both an HTML interface and a jsonl audit log, as shown above.

## To inspect, modify, or fork ConfoundSearch

ConfoundSearch is open-source software with a permissive MIT License. This section is for programmers interested in the underlying code.

### What's in this code base?

The application is divided into four layers, as described in the technical appendix. Each is one directory, and if you are
curious about any of them, feel free to look into the details:

- **Data** ([`confoundsearch/data/`](confoundsearch/data)): code used to collect evidence (hosted Elasticsearch, Semantic Scholar, OpenAlex, or your file).
- **Logic** ([`confoundsearch/logic/`](confoundsearch/logic)): the Markov decision process (graph, moves, reward, policy) and the audit log.
- **Language model** ([`confoundsearch/model/`](confoundsearch/model)): code to interface with any OpenAI-compatible model, including models hosted on OpenRouter.
- **Interface** ([`confoundsearch/interface/`](confoundsearch/interface)): the web app, command line, JSON output and agent skill.

### Other key directories

- [`tests/`](tests): the unit tests, with 100% coverage. They also enforce the layer boundaries described in the technical appendix.
- [`devops/`](devops): scripts for releasing, checking the package on every Python version, and publishing the anonymous review copy.
- [`evals/`](evals): code and data for the pilot studies described in the technical appendix.
