---
name: confoundsearchui
description: Open the ConfoundSearch web interface on the latest result, to review the candidate confounds as a causal graph with the passages behind each link. Use after a ConfoundSearch run, when the user asks to see, open or review the results as a graph or in the interface. It only opens a page; it does not search and costs nothing.
---

# ConfoundSearch interface

Opens the page a ConfoundSearch run wrote, so the researcher can review it as a causal graph.

## Find the page

1. Use the page the run wrote: `confoundsearch-result.html` in the working directory, or the file given to
   `--out`. If the user named a file, use that.
2. For a file of known confounds (JSON with `iv`, `dv` and `controls`), make a page first:
   `confoundsearch review --in FILE.json`, which writes `FILE.html` and opens it.
3. If there is no page, say so and suggest `/confoundsearch` first. Do not start a run yourself: it spends money.

## Open it

```bash
confoundsearch review --in confoundsearch-result.html
```

It opens the page in the user's browser. If there is no desktop to open it on (a remote or cloud session), it prints
the full path to the file; tell the user that path. `confoundsearch review --in FILE --format text` prints the confounds
and their papers in the terminal instead.

## Tell the user how to read it

The diagram shows the two variables and the first five confounds. Clicking a confound in the list, or a node or
arrow in the diagram, shows the papers behind that link, each with its quoted passage. It suggests; the researcher
decides. For the full audit trail of the run, `confoundsearch audit FILE`.
