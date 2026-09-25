# Validating extracted records

`pmt validate` compares a curated reference CSV with a scraped CSV using the
same recipe that defined the extraction. The local browser interface records
review decisions and calculates field-level precision, recall, and F1. It does
not modify either source CSV.

## Create the validation set

Generate a blank CSV for a bundled recipe:

```bash
pmt validate template band_gap_validation band_gap_validation.csv
```

A custom recipe JSON file can be used in the same way:

```bash
pmt validate template ./my_recipe.json my_validation.csv
```

The template contains `Identifier`, `Title`, and `DOI`, followed by every
recipe field in recipe order. Fill it using these rules:

- Add one row for every record defined by the recipe. For a material recipe,
  each material, sample, composition, phase, or structure therefore gets its
  own row.
- Repeat the paper metadata on every row when a paper has multiple records.
- Keep every recipe identity field in the CSV. The reviewer uses those fields
  to suggest record pairs within a paper.
- Write list and dictionary values as JSON. Python-style list and dictionary
  literals are also read, but JSON is the portable form.
- Use an empty cell, `None`, `null`, or `nan` for an absent scalar value.

Each CSV must contain at least one paper identifier: DOI, paper ID, or title.
The reviewer accepts `Identifier`, `Paper id`, or `paper_id` for a paper ID;
`DOI` or `doi` for a DOI; and `Title` or `Paper title` for a title. A scraper's
unnamed CSV index column is ignored.

To represent a paper with zero expected records, add one row containing its
paper metadata, leave the recipe identity fields empty or set them to `None`,
and leave the other recipe fields empty or use `[]`. The paper remains in the
review, but the placeholder row is not treated as an expected record.

## Open the reviewer

Start the local GUI and select both CSVs and a recipe in the browser:

```bash
pmt validate gui
```

The recipe menu includes bundled recipes and **Custom recipe JSON…** for a
standalone recipe file. To open the same inputs directly on later runs, pass
all three paths:

```bash
pmt validate gui \
  --validation band_gap_validation.csv \
  --scraped temp_scraped_materials.csv \
  --recipe band_gap_validation
```

Use a custom recipe path with `--recipe` when required. The three input options
must be supplied together. Other useful options are:

`--output REVIEW.json`
: Store the review decisions at a chosen path. Without this option, decisions
  are kept in the local review directory.

`--no-browser`
: Start the server without opening a browser and print its URL instead.

The server listens only on `127.0.0.1`. Stop it with {kbd}`Ctrl+C` in the
terminal that started it.

## Check the suggested pairs

The reviewer first groups rows into papers, using normalized DOI, then paper
ID, then normalized title. It suggests one scraped paper for each validation
paper using a unique DOI match first, followed by paper ID and title. Conflicts,
duplicate candidates, and ambiguous metadata are shown for manual resolution.

Within a paired paper, the reviewer suggests one-to-one record pairs from the
recipe's identity fields. Suggestions are navigation aids only: a suggested
paper or record pair does not mark any value correct.

Use the **Scraped paper** and **Scraped entry** menus to change a pairing. Mark
an expected entry **missing** when the scraper produced no corresponding
record. Scraped papers or entries with no validation partner remain available
to mark as extra. Papers deliberately represented with zero expected records
show any scraped entries without inventing a validation record.

## Review fields and structured values

The interface has separate scrollable panels for papers, entries, and the
current comparison. For an ordinary scalar field, compare the validation and
scraped values, then choose **correct**, **incorrect**, or **unreviewed**. Notes
are optional and are stored with the field decision.

**Mark entry correct** confirms every populated part of the current paired
entry. Use it only after checking the whole entry.

Lists and dictionaries are reviewed at their natural structure:

1. Pair each validation list item with its corresponding scraped item.
2. Mark an expected item missing when no scraped item corresponds to it.
3. Score each scalar value or dictionary key independently.
4. Mark unpaired scraped items or keys extra, then decide whether each extra is
   correct or incorrect.

The formatted view presents structured values as readable items. **Raw cell**
retains the exact original CSV text for inspection. Nested lists and
dictionaries use the same controls recursively.

An extra is not automatically an error. A scraped value omitted from an
incomplete validation set can be marked correct; a fabricated or wrongly
attributed extra should be marked incorrect.

## Autosave and resume

Decisions autosave locally after changes. By default, reviews and cached input
copies are stored under:

```text
~/.config/paperminertoolkit/validation-reviews/
```

The setup screen lists in-progress reviews, most recent first. Select one to
resume it without locating the input files again. The cached copies are local
snapshots used only by the reviewer; the original validation and scraped CSVs
remain unchanged.

A review is keyed by a fingerprint of the complete validation CSV, scraped
CSV, and recipe. Changing any of those inputs starts a separate review and
preserves the earlier decisions. If `--output` was used, pass that path again
when reopening the same custom review file.

## Interpret the scores

Open **Scoring summary** at any point. Metrics use the decisions completed so
far, while the **Pending** column shows how much remains unreviewed. These
in-progress values are provisional until the review is complete.

Each populated scalar value counts separately. A dictionary in a list, for
example, contributes one decision for each populated key. The field total and
the expandable per-part totals use the following rules:

| review result | TP | FP | FN |
|---|---:|---:|---:|
| correct paired value | 1 | 0 | 0 |
| incorrect paired value | 0 | 1 | 1 |
| missing expected value | 0 | 0 | 1 |
| correct extra value | 1 | 0 | 0 |
| incorrect extra value | 0 | 1 | 0 |
| empty on both sides | 0 | 0 | 0 |
| unreviewed value | 0 | 0 | 0, and 1 pending |

The reported ratios are:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

If a denominator cannot yet be calculated, the GUI shows **pending** while the
field is incomplete and **undefined** after completion.

Before archiving a scoring run, complete these fields in **Scoring summary**:

- extraction mode;
- model identifier;
- provider;
- context length;
- software commit;
- run date; and
- numeric matching rule or tolerance.

A run is complete only when every paper and record is resolved, every
assessable field or structured part has a decision, every extra is judged, and
all run details are present.

## Export the scoring run

Choose **Download scoring run** to save a JSON archive containing the decisions
and calculated score report. The report records:

- TP, FP, FN, pending count, precision, recall, and F1 for every field;
- per-part metrics for structured fields;
- the run metadata and scoring protocol;
- the number of validation papers;
- unresolved pairing and metadata counts; and
- the names and SHA-256 fingerprints of both CSVs and the recipe.

An incomplete download is named as a draft. Treat provisional metrics as
working feedback and use a complete scoring run for reported results.
