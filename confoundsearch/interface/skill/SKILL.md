---
name: confoundsearch
description: Find candidate confounds (omitted variables, alternative explanations) for a study of whether X affects Y, with the published passages that support each one and a verifiable audit trail. Use when the user asks which confounds, controls or alternative explanations to consider for a causal claim, or wants literature evidence that a variable influences both X and Y. Runs the confoundsearch command line; each run costs money (AI model calls).
---

# ConfoundSearch

ConfoundSearch is an AI agent that searches a collection of papers and proposes **confounds**: constructs that
plausibly influence both a study's independent variable (X) and its dependent variable (Y). For each one it keeps
the passages that support the claim, and it writes an **audit trail** of everything it searched, retrieved and
judged. You run it through its command line and report what it found.

It suggests; the researcher decides. Never present its output as a verdict or as proof of causation.

## When to use it

Use it when the user wants confounds, omitted variables, controls or alternative explanations for "X affects Y",
or literature evidence for a candidate confound. Do not use it for anything else, and do not run it on your own
initiative: it spends the user's money.

## Before you run it

1. Check it is installed: `confoundsearch --version`. If not found, tell the user to install it with
   `pip install confoundsearch` and stop.
2. Check the settings exist **without printing them**:
   - an AI model: `OPENAI_API_KEY` is set; or `OPENROUTER_API_KEY` is set (it is used automatically when it is the only key;
     `--openrouter` forces it) and the model name is an OpenRouter slug such as `openai/gpt-4.1`; or `OPENAI_BASE_URL` points at a compatible
     server. The model name always goes in `--model`;
   - papers: by default the hosted index, which needs `ES_URL` and `ES_API_KEY`. Alternatives are
     `--corpus semanticscholar` (a free key in `S2_KEY` lifts the rate limit), `--corpus openalex`, or
     `--corpus-file passages.jsonl`. To try the tool quickly, `confoundsearch demo` runs a curated example (organizational structure and firm
     performance) on a small set of papers shipped with the package, with saved model answers: it takes about a second,
     costs nothing and needs no keys. `--live` runs the same question on the real sources. With no `--corpus`, every source that is set up is searched and merged.
   Variables can live in a `.env` file in the working directory.
3. If something is missing, tell the user exactly which variable to set in `.env` and stop.
   **Never ask the user to paste a key into the chat, and never echo one.**
4. Tell the user what you are about to do and what it costs. By default the run asks the AI model for about eight
   confounds, gets synonyms for each of them and for the two variables, searches for all of them, searches
   again in papers' own words (the measures of each variable and the mechanisms of each confound), then has the
   model read everything it retrieved at once to pick the papers that support each link (the
   synthesis): roughly 30 to 90 requests in all, and many searches. It ends by itself.

## How you are invoked

Usually as `/confoundsearch iv="..." dv="..."`: `iv` is X (the cause) and `dv` is Y (the outcome). Plain
language works too ("what confounds X and Y?"). If either is missing, ask for it.

## Run it

```bash
confoundsearch run --json --model gpt-4.1 \
    --iv "IT investment" --dv "firm performance" \
    --out confoundsearch-result.html
```

Standard output is a single JSON document and nothing else; progress and errors go to standard error. Add
`--confounds N` to look for more or fewer confounds (default 8), `--no-synthesis` to keep only papers that use the confound's own words, `--no-mechanisms` to skip the second round of searches, `--queries N` and `--synonyms N` to search more wordings of each pair (default 3 and 3; slower, more thorough), and `--max-budget USD` to stop once the language model has cost about
that much (an estimate from the tokens used; a run usually costs under $1). The default plan has no randomness: the same answers
from the model give the same run (a hosted model may still answer differently). Searches are saved in a data directory (the system's temporary folder, or `$CONFOUNDSEARCH_CACHE`), so a repeated search is
answered from disk (one database per independent/dependent variable pair); `confoundsearch flush` empties that folder
(`--iv`/`--dv` for one question) and `--no-cache` skips it. The run also writes
the standalone page named by `--out`, the audit trail next to it as `confoundsearch-result.audit.jsonl`, and a log
as `confoundsearch-result.log.jsonl` (`--log PATH`): every search request with its status, every step, and any error.

## Reading the result

Top-level fields: `question`, `ended`, `steps`, `score`, `confounds`, `model_calls`, `models`, `files`, `audit`.

If the run stopped early the JSON also has an `error` string, the exit status is 1, and what was found so far is kept.
If it stopped because the `--max-budget` was reached, the JSON has a `stopped` string instead, the exit status is 0, and
what was found so far is kept.

Each item of `confounds` has: `label`, `phrasings`, `linked_to_both_variables`, `supported_on_both_links`,
`evidence_to_iv`, `evidence_to_dv`, `rejected_passages`. Each passage in the evidence lists has `id`, `title`,
`url`, `quote`, `found_by_query`, `judged_at_step`.

- `supported_on_both_links: true` means the judge accepted at least one passage for "this affects X" **and** for
  "this affects Y". These are the candidates to report.
- `linked_to_both_variables: true` with `supported_on_both_links: false` means the agent proposed the link but found
  no accepted evidence for one side. Mention these as weak or unsupported, not as findings.
- `files.audit` is the path of the audit trail; `audit` holds its status (`events`, `intact`, `complete`, `head`).

## Report it

- List the supported confounds, and for each one the papers that back each link: `title`, `url`, and the `quote`.
  **Cite only passages that appear in the output.** Do not add references from memory.
- Say how the run ended (`ended`), how many `steps` and `model_calls` it used, and the `models`.
- Verify the log: run `confoundsearch audit <the audit file>`. Exit status 0 means the hash chain is intact and the
  run finished. Report the result, and give the user the file path and the `head` hash if they want to keep a record.
- State the limits briefly: the judge is an AI model's judgment of each passage; only papers with abstracts are
  searched; results vary between runs; and the confounds are suggestions for the researcher to assess.

## If it fails

| Message contains | Meaning | What to tell the user |
| --- | --- | --- |
| `needs ES_URL` | the hosted index is not configured | set `ES_URL` and `ES_API_KEY` in `.env`, or use `--corpus semanticscholar` / `openalex` / `--corpus-file` |
| `HTTP 401` | a key is wrong | check the named key in `.env` |
| `HTTP 402` | the AI account is out of credits | the user needs to add credits (for OpenRouter, at openrouter.ai/credits) |
| `HTTP 429` and `semanticscholar` | rate limited | request a free key at https://www.semanticscholar.org/product/api and set `S2_KEY` |
| `HTTP 429` and `OpenAlex` | rate limited | set `OPENALEX_EMAIL` or `OPENALEX_API_KEY` |
| `could not reach` | network or wrong address | check the address and connection |

If a run fails, read the end of the `.log.jsonl` file (the `run_failed` or `crashed` line) and report it.
Do not retry in a loop: each retry costs money. Report the message and let the user decide.
