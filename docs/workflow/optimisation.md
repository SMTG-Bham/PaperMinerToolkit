# Optimising a recipe

`pmt recipe optimise` refines a recipe's wording until scraped records match a
hand-curated validation CSV more closely. Each round:

1. scrapes the validation papers with the current recipe;
2. scores the scraped records against the validation CSV automatically;
3. sends the scores and mismatches to a language model, which proposes a
   revised recipe; and
4. scrapes again with the proposal, keeping it only if the score improves.

The loop stops after `--iterations` revisions, after `--patience` consecutive
revisions fail to improve, or when the F1 score reaches 1.

## Prepare the inputs

The optimiser uses the same validation CSV as `pmt validate`. Create it with
`pmt validate template` and fill it as described in {doc}`validation`. Every
validation paper must also be in the corpus with downloaded text (or an
abstract when `--mode abstract` is used). Papers are matched to the corpus by
DOI, then paper ID, then title; papers that cannot be matched, or that have no
downloaded content, are skipped with a message.

## Run the optimiser

```bash
pmt recipe optimise papers.db band_gap_validation band_gap_validation.csv \
  --output-dir band_gap_optimisation \
  --iterations 5 \
  --holdout 0.3
```

The extraction model is the configured text profile, and can be overridden with
`--model`, `--provider`, and `--base-url` as for `pmt scrape`. The model that
rewrites the recipe defaults to the extraction model. A stronger model often
writes better revisions, so it can be chosen separately:

```bash
pmt recipe optimise papers.db ./my_recipe.json validation.csv \
  --model qwen3-32b --provider local --base-url http://localhost:8000/v1 \
  --optimiser-model claude-opus-5-5 --optimiser-provider anthropic
```

With `--optimiser-provider claude-code` the revisions come from a logged-in
Claude Code CLI instead of an API key: each request runs `claude -p` with every
tool disabled, and `--optimiser-model` takes a Claude Code model name or alias
such as `opus`. The CLI is found as `claude` on `PATH`, or set
`PAPERMINERTOOLKIT_CLAUDE_CODE_BIN` to its path. Reasoning effort and thinking
settings of the extraction profile are not passed to an optimiser on another
provider.

Every round scrapes every validation paper, so one run makes up to
`(iterations + 1) × papers` extraction requests plus one optimiser request per
revision. Start with a small validation set and a few iterations.

## What the optimiser may change

The search-field keys, their order, and the identity fields define the columns
of the validation CSV, so they are fixed. The optimiser may edit:

- the record definition's `subject`, `singular`, `plural`, and `unit`;
- `additional prompts`; and
- each field's `prompt` and `example`.

Aliases and storage units are always restored from the starting recipe, fields
the optimiser leaves out keep their current wording, and a proposal that adds
or renames a field is rejected. The optimiser is
told not to copy paper-specific values into the recipe, but always read the
best recipe before using it on a full corpus.

## Avoid overfitting

With a small validation set, a recipe can improve on those papers while
getting worse on new ones. Use `--holdout` to keep a fraction of the papers
away from the optimiser: their mismatches are never shown to it, and a revision
is kept only when it improves the held-out score. `--seed` fixes the split.

## Automatic scoring

Scoring follows the same counting rules as the review GUI, without a human:

- records within a paper are paired one-to-one by similarity of their identity
  fields, ignoring word order;
- each populated scalar, list item, or dictionary key is one decision;
- numbers match within a relative `--numeric-tolerance` (default 2%), so
  `3.30 eV` matches `3.3 eV` and `3.3`;
- other text matches after case and punctuation normalization, or when its
  similarity reaches `--text-threshold` (default 0.9); and
- values in unpaired scraped records are false positives, and values in
  unpaired validation records are false negatives.

Because extra values cannot be judged automatically, every extra counts as a
false positive. An incomplete validation set therefore lowers precision. Use
`pmt validate gui` on the final scraped CSV for a reported score.

## Values missing from the paper text

A validation set is usually curated from the whole paper, including tables,
figures, and supporting information. Downloaded text often lacks some of
these, so an expected value may simply not be in the text the extraction
model receives. No recipe can recover it, and counting it as a miss pushes the
optimiser towards rules that encourage guessing.

The optimiser therefore checks every expected value that contains a decimal
number against the numbers printed in the paper text. If any such number is
absent, that value is **unreachable**. For a list of dictionaries, such as
band-gap items, the whole item shares one decision, so its method and gap type
are excluded along with its missing value.

- Unreachable values are left out of the F1 used to choose recipes.
- A scraped value that matches an unreachable one still counts as correct.
- Mismatches sent to the optimiser list unreachable parts under
  `not_in_text`, and the optimiser is told not to chase them.
- `score.json` reports an `unreachable` count for every field and an
  `f1_including_unreachable` score that counts them as misses, for
  comparison with the review GUI.

The run prints how many expected values were unreachable. A large share
suggests trying `pmt scrape --mode text-images` or checking the downloaded
text. Pass `--include-unreachable` to choose recipes by the stricter score
instead.

Numbers are compared with the same `--numeric-tolerance` as scoring, so a value
printed rounded in the text, or with a middle-dot or comma decimal such as
`2·70`, is still reachable. Only decimal numbers are checked, so values
without one, such as material names, methods, or integers, are always treated
as reachable.

## Output files

```text
band_gap_optimisation/
├── best_recipe.json      # highest-scoring recipe
├── history.json          # every attempt, its score, rationale, and whether it was kept
├── iteration_00/         # baseline round with the starting recipe
│   ├── recipe.json
│   ├── scraped.csv       # opens in pmt validate gui
│   └── score.json        # per-field precision, recall, and F1
└── iteration_01/ ...
```

Use the best recipe like any custom recipe:

```bash
pmt scrape papers.db band_gap_optimisation/best_recipe.json --mode text
```
