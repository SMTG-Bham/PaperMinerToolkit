# Recipes and prompt construction

Recipes define the structured records that PaperMiner asks a language or vision model to extract. They are not restricted to materials: a record can represent a material, experiment, reaction, device, organism, intervention, measurement series, or another unit that can be described by a paper.

A recipe controls four related parts of the workflow:

1. The subject and granularity of one output record.
2. The exact fields in every output object.
3. Domain-specific extraction and reconciliation rules.
4. Column aliases and optional unit conversion during storage.

Use the same recipe for `ps_scrape` and `ps_store`. Mixing recipes can produce incompatible columns, aliases, and units.

## Complete example

The following recipe extracts one record per electrochemical cycling experiment. It deliberately contains no materials-specific terminology.

```json
{
  "record definition": {
    "subject": "electrochemical cycling experiments",
    "singular": "experiment",
    "plural": "experiments",
    "unit": "a distinct cell and cycling protocol",
    "identity fields": [
      "Cell identifier",
      "Protocol"
    ]
  },
  "additional prompts": "Keep measurements from different temperatures in separate records. Do not treat a protocol mentioned only in the introduction as an experiment performed in this paper.",
  "search fields": {
    "Cell identifier": {
      "prompt": "Reported cell label or identifier",
      "example": "Cell A",
      "aliases": [
        "Cell",
        "Sample identifier"
      ]
    },
    "Protocol": {
      "prompt": "Reported cycling protocol, including charge and discharge rates",
      "example": "Charge at C/10 and discharge at C/5",
      "aliases": [
        "Cycling protocol"
      ]
    },
    "Temperature": {
      "prompt": "Temperature at which cycling was performed",
      "example": "25 °C",
      "unit": "K",
      "aliases": [
        "Test temperature"
      ]
    },
    "Capacity": {
      "prompt": "Reported discharge capacity with cycle number where available",
      "example": "142 mAh g^-1 at cycle 100",
      "aliases": [
        "Discharge capacity"
      ]
    }
  }
}
```

Save a custom recipe as a standalone JSON file and pass its path in place of a bundled recipe name:

```bash
ps_scrape papers.db ./cycling_recipe.json --mode text
ps_store papers.db temp_scraped_materials.csv cycling_results.csv ./cycling_recipe.json --assume-yes
```

A standalone file may contain the recipe object directly, as above, or exactly one named recipe:

```json
{
  "cycling": {
    "record definition": {
      "subject": "cycling experiments",
      "singular": "experiment",
      "plural": "experiments",
      "unit": "a distinct cell and cycling protocol",
      "identity fields": [
        "Cell identifier"
      ]
    },
    "additional prompts": "",
    "search fields": {
      "Cell identifier": {
        "prompt": "Reported cell label or identifier",
        "example": "Cell A"
      }
    }
  }
}
```

## Record definition

`record definition` supplies the vocabulary and record boundaries used by the general prompts.

`subject`
: A plural or collective description of the information being extracted. It completes the sentence “You extract structured records about …”. Examples include `lithium-conducting solid electrolytes`, `polymer degradation experiments`, and `clinical interventions and outcomes`.

`singular`
: The noun or short noun phrase used for one record subject, such as `material`, `experiment`, or `intervention`.

`plural`
: Its explicit plural form. PaperMiner does not guess plurals because scientific terms and multi-word phrases are not reliably pluralised automatically.

`unit`
: A precise description of what one JSON object represents. This is the most important granularity instruction. State which changes create a new record, for example `a distinct cell and cycling protocol` or `a distinct material, composition, phase, or sample`.

`identity fields`
: Search-field names that provide the strongest evidence that text-derived and image-derived records refer to the same subject. Every name must exactly match a key in `search fields`. An empty list is allowed when no field is consistently identifying, in which case reconciliation uses the complete record context.

Do not put formatting rules or long domain policies in `unit`. Keep it focused on the boundary of one record and put detailed rules in `additional prompts`.

## Search fields

The keys under `search fields` are the output schema. The model must return every key exactly once in every object and must not add keys.

Each field can define:

`prompt`
: Required for scraping. Describes what evidence and value belong in the field. Field-specific requirements should live here rather than being repeated in the general prompt.

`example`
: Required for scraping. Supplies the value used in the generated example record. Its JSON type matters: use a list or object when that is the required output type, not a string that merely looks like one.

`aliases`
: Optional alternative headings accepted by `ps_store`. Aliases do not change the extraction prompt. Keep aliases unique across fields to avoid ambiguous storage matches.

`unit`
: Optional target unit used by `ps_store`. It does not silently rewrite the extraction prompt. During storage, supported non-missing values can be converted to the configured unit and the final column receives the unit suffix.

For example, a field that must always contain a JSON list should say so explicitly and show a list-valued example:

```json
"Reported values": {
  "prompt": "JSON list of all values reported by this study, including when only one value is present",
  "example": [
    {
      "value": "1.2 eV",
      "method": "optical absorption"
    }
  ],
  "aliases": [
    "Study values"
  ]
}
```

The general prompt normally uses lists only for multiple values, but explicitly list-valued field instructions and additional recipe instructions take precedence.

## Additional prompts

`additional prompts` is a string containing rules specific to the recipe. It is included in both the extraction prompt and the text/image reconciliation prompt.

Use it for rules such as:

- evidence classification specific to the domain;
- distinctions that create separate records beyond the short `unit` description;
- conditions under which apparently similar records must not be merged;
- controlled vocabularies;
- required nested JSON structures;
- exclusions, such as results attributed only to previous work; and
- source-preserving requirements such as copying identifiers exactly as printed.

Material-specific matching rules belong here or in the material recipe's record definition, not in the package-wide prompt. For example, the solid-electrolyte recipe tells the model to distinguish formulas, stoichiometries, dopants, substituted phases, composites, and sample labels. A recipe for experiments can instead distinguish cells, protocols, cohorts, or conditions without inheriting irrelevant chemistry language.

Avoid duplicating every field definition in `additional prompts`. Field-specific prompts are inserted later as part of the schema. Use additional prompts for relationships between fields, record-level policies, and domain-wide evidence rules.

## How the extraction prompt is assembled

PaperMiner constructs a system prompt in a fixed order:

1. **Task statement.** `subject` identifies the target information and the source is named as paper text or a paper image.
2. **Record definition.** `unit`, `singular`, and `plural` define record granularity and provide natural terminology.
3. **General evidence rules.** These require source-supported values, preserve relevant methods and conditions, use the general missing value `"None"`, and prohibit invented values.
4. **Source rules.** Text prompts distinguish this paper's results from citations and background. Image prompts cover captions, tables, plots, labels, readable values, and optional text context.
5. **Schema.** Each `search fields` key and its `prompt` are rendered in recipe order.
6. **Additional recipe instructions.** The complete `additional prompts` string is inserted without template substitution.
7. **Output contract.** The model is required to return only a JSON array, include every schema key, and return `[]` when no relevant records exist.
8. **Example output.** The `example` value from every field is combined into one example JSON object.

The paper text itself is sent as the user message; it is not interpolated into the system prompt. For image extraction, images and optional context are supplied through the configured vision provider.

In simplified form, the generated prompt begins like this:

```text
You extract structured records about {subject} from a scientific {source}.

Record definition:
- Each output object represents {unit}.
- The recipe terminology is "{singular}" for one record subject and "{plural}" for multiple record subjects.
```

Recipe values are inserted only into fixed prompt sections. PaperMiner does not run arbitrary string formatting over `additional prompts`, so braces in JSON examples or scientific notation are preserved literally.

## Complete rendered prompt examples

The examples below use a shortened version of the cycling recipe so the substitutions are easy to follow:

```json
{
  "record definition": {
    "subject": "electrochemical cycling experiments",
    "singular": "experiment",
    "plural": "experiments",
    "unit": "a distinct cell and cycling protocol",
    "identity fields": ["Cell", "Protocol"]
  },
  "additional prompts": "Keep experiments performed at different temperatures in separate records.",
  "search fields": {
    "Cell": {
      "prompt": "Reported cell label or identifier",
      "example": "Cell A"
    },
    "Protocol": {
      "prompt": "Reported charge and discharge protocol",
      "example": "C/10 charge; C/5 discharge"
    },
    "Capacity": {
      "prompt": "Reported discharge capacity with cycle number",
      "example": "142 mAh g^-1 at cycle 100",
      "unit": "mAh g^-1"
    }
  }
}
```

Select a heading to expand the exact example. Line wrapping displayed by a browser does not add line breaks to the prompt.

<details>
<summary><strong>Text or abstract extraction prompt</strong></summary>

This is the system prompt used for downloaded text, PDF-derived text, and abstracts:

```text
You extract structured records about electrochemical cycling experiments from a scientific paper text.

Record definition:
- Each output object represents a distinct cell and cycling protocol.
- The recipe terminology is "experiment" for one record subject and "experiments" for multiple record subjects.

Extraction rules:
- Use only information supported by the provided paper text. Do not infer missing values from general domain knowledge.
- Prefer explicit reported values over derived or assumed values. Preserve relevant conditions, methods, units, qualifiers, and uncertainties when they are part of the reported value.
- If multiple distinct experiments are reported, return one record per experiment according to the record definition. Do not duplicate records for repeated mentions of the same experiment.
- If a field is not supported by the provided paper text, set that field to "None".
- Use lists only when multiple values are reported for the same field in the same record, unless the field or additional recipe instructions explicitly require a list.
- Keep values concise but complete enough to preserve scientific meaning.
- Treat abstracts, captions, tables, experimental sections, results, and supporting text as valid evidence.
- Do not use references, citations, or background discussion as evidence for the paper's own results or observations unless the text clearly states the information belongs to this work.

Schema. Use these keys exactly and do not add extra keys:
- "Cell": Reported cell label or identifier
- "Protocol": Reported charge and discharge protocol
- "Capacity": Reported discharge capacity with cycle number

Additional recipe instructions:
Keep experiments performed at different temperatures in separate records.

Output contract:
- Return a JSON array of objects. Return [] if no relevant experiments are present.
- Every object must contain every schema key exactly once.
- Return only JSON. Do not include markdown fences, comments, explanations, or prose.

Example output shape:
[
  {
    "Cell": "Cell A",
    "Protocol": "C/10 charge; C/5 discharge",
    "Capacity": "142 mAh g^-1 at cycle 100"
  }
]
```

The user message contains the paper text or one text chunk. Chunking does not alter the system prompt; each chunk is sent in a separate request with the same prompt.

</details>

<details>
<summary><strong>Image extraction prompt without paper-text context</strong></summary>

This is the system prompt used when images are the only evidence supplied to the vision model:

```text
You extract structured records about electrochemical cycling experiments from a scientific paper image.

Record definition:
- Each output object represents a distinct cell and cycling protocol.
- The recipe terminology is "experiment" for one record subject and "experiments" for multiple record subjects.

Extraction rules:
- Use only information supported by the provided paper image. Do not infer missing values from general domain knowledge.
- Prefer explicit reported values over derived or assumed values. Preserve relevant conditions, methods, units, qualifiers, and uncertainties when they are part of the reported value.
- If multiple distinct experiments are reported, return one record per experiment according to the record definition. Do not duplicate records for repeated mentions of the same experiment.
- If a field is not supported by the provided paper image, set that field to "None".
- Use lists only when multiple values are reported for the same field in the same record, unless the field or additional recipe instructions explicitly require a list.
- Keep values concise but complete enough to preserve scientific meaning.
- Use only information visible in the supplied image or images, including captions, tables, plot labels, axes, legends, annotations, and readable text.
- Actively inspect figures and tables in the image for the requested properties, including captions, table headers, table rows, plot axes, legends, labels, annotations, and inset text.
- For plots, report values only when they can be read from labels, annotations, tables, or clearly interpretable plotted data.
- If several images are supplied together, combine evidence across them only when they clearly describe the same record subject or observation.
- If the supplied image or images are decorative, unreadable, or irrelevant to the schema, return [].

Schema. Use these keys exactly and do not add extra keys:
- "Cell": Reported cell label or identifier
- "Protocol": Reported charge and discharge protocol
- "Capacity": Reported discharge capacity with cycle number

Additional recipe instructions:
Keep experiments performed at different temperatures in separate records.

Output contract:
- Return a JSON array of objects. Return [] if no relevant experiments are present.
- Every object must contain every schema key exactly once.
- Return only JSON. Do not include markdown fences, comments, explanations, or prose.

Example output shape:
[
  {
    "Cell": "Cell A",
    "Protocol": "C/10 charge; C/5 discharge",
    "Capacity": "142 mAh g^-1 at cycle 100"
  }
]
```

The provider payload contains one image batch. There is no accompanying paper-text context in this mode.

</details>

<details>
<summary><strong>Image extraction prompt with paper-text context</strong></summary>

When `--image-context paper-text` is used, the system prompt becomes:

```text
You extract structured records about electrochemical cycling experiments from a scientific paper image.

Record definition:
- Each output object represents a distinct cell and cycling protocol.
- The recipe terminology is "experiment" for one record subject and "experiments" for multiple record subjects.

Extraction rules:
- Use only information supported by the provided paper image. Do not infer missing values from general domain knowledge.
- Prefer explicit reported values over derived or assumed values. Preserve relevant conditions, methods, units, qualifiers, and uncertainties when they are part of the reported value.
- If multiple distinct experiments are reported, return one record per experiment according to the record definition. Do not duplicate records for repeated mentions of the same experiment.
- If a field is not supported by the provided paper image, set that field to "None".
- Use lists only when multiple values are reported for the same field in the same record, unless the field or additional recipe instructions explicitly require a list.
- Keep values concise but complete enough to preserve scientific meaning.
- You may use the supplied paper text as context, but values should still be tied to the image, caption, table, plot labels, legend, or nearby visual evidence.
- Actively inspect figures and tables in the image for the requested properties, including captions, table headers, table rows, plot axes, legends, labels, annotations, and inset text.
- For plots, report values only when they can be read from labels, annotations, tables, or clearly interpretable plotted data.
- If several images are supplied together, combine evidence across them only when they clearly describe the same record subject or observation.
- If the supplied image or images are decorative, unreadable, or irrelevant to the schema, return [].

Schema. Use these keys exactly and do not add extra keys:
- "Cell": Reported cell label or identifier
- "Protocol": Reported charge and discharge protocol
- "Capacity": Reported discharge capacity with cycle number

Additional recipe instructions:
Keep experiments performed at different temperatures in separate records.

Output contract:
- Return a JSON array of objects. Return [] if no relevant experiments are present.
- Every object must contain every schema key exactly once.
- Return only JSON. Do not include markdown fences, comments, explanations, or prose.

Example output shape:
[
  {
    "Cell": "Cell A",
    "Protocol": "C/10 charge; C/5 discharge",
    "Capacity": "142 mAh g^-1 at cycle 100"
  }
]
```

The full request therefore contains the image system prompt, one image batch, and the accompanying paper text. The context can help identify a figure or abbreviation, but the rule keeps extracted values tied to visual evidence.

</details>

<details>
<summary><strong>Text and image reconciliation prompt</strong></summary>

This system prompt is sent only when both extraction paths returned records for the same paper:

```text
You reconcile two sets of structured records about electrochemical cycling experiments extracted from the same paper.

Each output object represents a distinct cell and cycling protocol. The recipe terminology is "experiment" for one record subject and "experiments" for multiple record subjects.
Compare the text-derived and image-derived records and merge only records that refer to the same experiment.

Primary identity fields:
"Cell", "Protocol"

Rules:
- Use the recipe schema keys exactly. Do not add extra keys.
- Use compatible values in the primary identity fields as the strongest evidence that two records describe the same experiment. Also consider the full record context and the configured record unit.
- If one record has a more specific identity than another, merge them only when the less-specific record clearly applies to that same experiment.
- Keep records separate when their identity fields or defining context conflict.
- Do not merge records only because they share a reported value, method, source location, or broad category.
- Prefer explicit non-None values over None.
- If text and image records provide complementary fields for the same experiment, combine them into one record.
- If text and image records conflict, keep the value that is more specific or better supported; if the conflict cannot be resolved, keep both values as a list.
- Keep genuinely distinct experiments as separate records according to the configured record unit.
- Do not invent values. If neither source supports a field, use "None".
- Return a JSON array of reconciled records only. Do not include markdown, comments, or prose.

Additional recipe instructions:
Keep experiments performed at different temperatures in separate records.

Schema. Use these keys exactly and do not add extra keys:
- "Cell": Reported cell label or identifier
- "Protocol": Reported charge and discharge protocol
- "Capacity": Reported discharge capacity with cycle number

Example output shape:
[
  {
    "Cell": "Cell A",
    "Protocol": "C/10 charge; C/5 discharge",
    "Capacity": "142 mAh g^-1 at cycle 100"
  }
]
```

The accompanying user message contains both record sets as JSON:

```json
{
  "text_extracted_records": [
    {
      "Cell": "Cell A",
      "Protocol": "C/10 charge; C/5 discharge",
      "Capacity": "None"
    }
  ],
  "image_extracted_records": [
    {
      "Cell": "Cell A",
      "Protocol": "C/10 charge; C/5 discharge",
      "Capacity": "142 mAh g^-1 at cycle 100"
    }
  ]
}
```

</details>

<details>
<summary><strong>Unit-conversion prompt during storage</strong></summary>

When `ps_store` converts the `Capacity` field to its recipe unit of `mAh g^-1`, it sends this system prompt:

```text
Convert the following values of Capacity to mAh g^-1. Each result should be returned as a decimal on a separate line. If the input contains multiple values on one line, return the converted values as a python list on the same line. Only put values in square brackets if multiple values are provided on the line. Do not include the units. If you are unsure how to do the conversion, just return the original value. If a range is given, report this as two decimals with a hyphen/dash inbetween (For example: 1-10). If the value is already in the desired unit, just convert it to a decimal. Do not return "None". Do not return the value as an addition. If text is given and cannot be meaningfully converted, return the same text. Convert "RT" or "Room temperature" to the equivalent of 298.15K. Do not use quotation marks. Make sure that there are as many output values as input.
```

The user message contains only the non-missing source values, one per line:

```text
0.142 Ah g^-1
155 mAh g^-1
```

Missing inputs are removed before the request and restored as missing values afterward. Large batches may be split across several requests without changing the system prompt.

</details>

Optional context compression does not construct another recipe-specific LLM prompt. It uses the selected extraction prompt to preserve relevant content before the normal request is made.

## Text and image reconciliation

In `text-images` mode, text and images are extracted independently. If both produce records, the text model receives a separate reconciliation prompt and both JSON record sets.

The reconciliation prompt uses:

- `subject`, `singular`, `plural`, and `unit` from the record definition;
- `identity fields` as the strongest matching evidence;
- `additional prompts` for domain-specific separation and merging rules; and
- the same search-field schema and example record.

It merges complementary fields only when records refer to the same configured record unit. Conflicting identity fields keep records separate. An unresolved conflict in a non-identity value is retained as a list rather than silently discarded.

Reconciliation occurs between text and image results from the same paper. Records produced by separate text chunks are not currently reconciled, so use a suitable model context limit when record duplication across chunks would be costly.

## Validation and common errors

A loaded recipe must contain non-empty `record definition` and `search fields` objects. The record definition must contain all five keys, its textual values must be non-empty strings, and `identity fields` must be a list of valid search-field names.

Common failures include:

- using an identity-field alias instead of its exact search-field key;
- using a singular `subject`, which produces awkward task wording;
- defining a vague unit such as `a result` that does not explain when records split;
- showing a string example when the output must be a list or object;
- placing paper-specific facts in the recipe rather than extraction rules; and
- sharing one final CSV between recipes with unrelated schemas.

Before a large scrape, inspect the prompt and run a small representative batch. The Python prompt builders can be used without calling a model:

```python
from paperscraper.extract import build_image_extraction_prompt, build_text_extraction_prompt
from paperscraper.recipes import load_recipe

recipe = load_recipe("./cycling_recipe.json")
print(build_text_extraction_prompt(recipe))
print(build_image_extraction_prompt(recipe, with_context=True))
```

Check that the record unit is unambiguous, list-valued examples have the intended JSON shape, and no terminology appears unless it belongs to the general scientific rules or the selected recipe.
