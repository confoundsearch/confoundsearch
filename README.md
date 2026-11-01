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
confoundsearch install-skill      # 3. optional: add the Claude skill

# A run can be slow; we are working to reduce latency. See below.
# 4A. in Claude Code:  /confoundsearch iv="Organizational structure" dv="Firm performance"
# 4B. or run the package directly:
confoundsearch run --corpus semanticscholar --model qwen/qwen3-max \
  --iv "Organizational structure" --dv "Firm performance" \
  --out results/org.html

confoundsearch review --in results/org.html     # 5. open the interface: the confounds as a causal graph
```

We suggest running in a tmux/screen session or side terminal. See the technical appendix for more.

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
