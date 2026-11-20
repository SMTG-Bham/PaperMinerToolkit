# Emerging fields

LDA trends follow one fixed set of broad topics through time, so a small new field is absorbed into an older topic and never shows. `pmt topics emergence` works from embedded abstracts instead. It builds one clustering tree and cuts it at three depths:

- **areas**, about a dozen broad parts of the corpus;
- **topics**, a few dozen inside them;
- **niches**, a few hundred clusters of roughly forty papers, small enough for a new field to have one of its own.

Each cluster is then measured by how its *share* of the corpus changes, not its paper count, which rises with the whole field.

## Build a corpus that samples every year evenly

Emergence is read from changes in share between years, so every year must be sampled the same way. A relevance-ranked search does not do that: it can over-represent recent years, which then look like growth. OpenAlex queries accept scope terms for this:

```bash
pmt search '"solid electrolyte" from:2010 to:2025 sample:per-year' papers.db \
  --source openalex --count 20000 --store-abstract
```

`from:` and `to:` take a year or an ISO date and bound the publication date. `sample:per-year` counts each year's matches and, if they total more than `--count`, draws the same seeded random fraction from every year, so each year keeps its share. With fewer matches than `--count` it fetches them all. The query is recorded with the search, so the sample can be reproduced.

OpenAlex matches the query against titles and abstracts unless `in:` says otherwise (see {doc}`corpus`). Keep that default here. With `in:all`, which adds full text, 63% of the papers in a 20,000-paper sample did not mention the phrase in their title or abstract, and the backtest below fell to 0.12.

Abstracts matter. Add `--store-abstract`, or fill gaps with `pmt download papers.db --format abstract`, and check coverage with `pmt corpus stats`. Papers without an abstract are left out unless `--include-title-only` is given; their title-only embeddings are noisier. In the sample above, 70% of the records carried one.

Size matters too. A niche of forty papers spread over sixteen years has only a few papers a year, so plan on ten thousand or more abstracts; the backtest below used 13,642.

Search records carry no citation counts. Run `pmt enrich papers.db` to add them if you want the niche table's impact column.

Choose the query with its side effects in mind. `"solid electrolyte"` also matches *solid electrolyte interphase*, which brings in liquid-electrolyte and zinc-anode work, and other fields that use the phrase, such as CO₂ electrolysers with a solid electrolyte. Those appear as areas of their own, which can be informative or noise depending on the question.

## Run the analysis

Embedding needs PyTorch:

```bash
pip install 'paperminertoolkit[bert]'
pmt topics emergence papers.db emergence --watch halide --watch garnet
```

The default model is [SPECTER2](https://huggingface.co/allenai/specter2_base), trained on citation links between scientific papers, so papers close together in its space are related work. It is downloaded on first use. Twenty thousand abstracts take about eight minutes on 32 CPU cores and seconds on a GPU. Embeddings are cached in the output directory; pass `--embedding-cache` to share one cache between runs, and `--overwrite` to rerun into the same directory.

The analysis year defaults to the latest complete year in the corpus. It needs at least `--growth-window` (default 8) years up to then: a corpus collected from one or two years is rejected, because it has no history to compare against.

## Read the report

`report.md` opens with the **field map**: each area, the topics inside it, the flagged niches inside each topic, the three papers nearest each topic's centre, and the terms rising and fading inside it. Every area and topic carries a status from its share over the growth window:

| Status | Meaning |
|---|---|
| emerging | share growing (growth z ≥ 2) and concentrated in the last three years (recency lift ≥ 1.5) |
| growing | share growing, but with a longer history |
| stable | no significant change in share |
| declining | share falling (growth z ≤ −2) |

*Growth* is the slope of a quasi-Poisson fit to the cluster's yearly counts with the corpus size as an offset, reported as a yearly share multiplier and a z statistic. *Recency lift* is the cluster's fraction of papers from the recent window over the corpus fraction from it.

A straight-line slope cannot tell a field that is speeding up from one that is levelling off, so each cluster also has a **shape**. A second fit adds a squared year term, and when that curvature is significant (|z| ≥ 2) the shape names it:

| Shape | Share over the growth window |
|---|---|
| accelerating | rising, faster at the end |
| growth slowing | rising, slower at the end |
| peaked | rising at the start, falling at the end; the turning year is shown |
| rebounding | falling at the start, rising at the end; the turning year is shown |
| decline easing | falling, more slowly at the end |
| decline steepening | falling, faster at the end |
| steady | no significant curvature |

`start_growth` and `end_growth` in the CSVs give the fitted yearly share multiplier in the window's first and last year, and `curvature_z` the test. The share column of the field map and niche table draws each year's share since the corpus began (at most 25 years), scaled to the cluster's own peak, as a row of blocks such as `▁▁▂▃▅▇█▇`. `cluster_years.csv` holds the numbers behind it: papers, corpus papers, and share for every area, topic, and niche in every year.

The map shows large, established fields as well as new ones. In the solid-electrolyte sample above, garnet/LLZO is the largest part of an oxide-electrolyte area of about 1,000 papers. Its two topics lose 6–9% of their share a year as other fields grow faster. Halide electrolytes are a growing topic of about 340 papers, with its share rising 14% a year, and it holds the Li₂ZrCl₆ and Li₃InCl₆ niches. Areas and topics are named by their most distinctive terms, so look at a cluster's papers in `paper_clusters.csv` before naming it.

A topic holding steady can still be changing inside. Under each topic, **rising** and **fading** list terms whose share of the topic's papers grew or shrank over the growth window (growth z ≥ 2, the same fit as for clusters) and that are at least 1.5 times as common in the topic's recent papers (for rising) or earlier papers (for fading) as in the corpus's. Only field vocabulary is listed: terms whose papers gather in a few niches. Without that, the drift in academic wording, such as *crucial*, *insights*, and *comprehensive*, rises inside most topics. Gathering is judged two ways, because each fails on words of one frequency:

- **Concentration:** one minus the normalised entropy of the term's papers over niches, at least 0.25. A rare word looks concentrated even when it is scattered, because its few papers cannot cover many niches.
- **Dispersion:** Pearson's chi-square of its niche counts against niche sizes, over the degrees of freedom, at least 2. A scattered word scores about 1 however common it is, but a very common word gathered only mildly scores high.

In the solid-electrolyte sample, the halide topic gains *halide solid*, *Ta* and *ZrCl* while *closo* borohydrides fade, and a garnet topic gains *LLZTO* and *critical current* while *garnet structure* and *pellets* fade. `topic_terms.csv` has up to ten of each per topic, with their yearly counts.

The **emerging niches** table ranks niches by the mean percentile of growth z and recency lift. A niche needs `--min-papers` dated papers, three of them recent, to be ranked, and one with most of its papers in a single earlier year is treated as a one-off batch (a patent family, a special issue, duplicates) and left out. *Novelty* is how unlike all earlier papers the niche's first papers were, and *impact* is its median citation count relative to papers from the same year; neither is scored, and impact is shown only when the corpus has citation counts. Labels are class-based TF-IDF terms; the report lists the most central and newest titles of the top niches, and `niches.csv` lists them for every niche.

The **emergence timeline** dates each flag, for the top niches and every niche flagged now, over the `--timeline-years` (default 6) years before the analysis year. It gives two dates:

- **First flagged.** Each year column scores today's niches on the papers published up to that year, as the analysis would have scored them then, and shows the percentile among the niches ranked that year, with `*` where the niche met the emerging rule. These columns use today's cluster boundaries, which were not known then.
- **First seen.** The corpus is re-clustered as it stood in each year, and each niche is followed to the niche of that year holding most of its papers. First seen is the first year such a niche was flagged, with the share of that flagged niche this one made up. In the solid-electrolyte sample, aqueous zinc anodes were first seen in 2019, as a fifth of a flagged zinc niche that also held zinc–air work, and flagged on their own papers from 2021.

A date in the earliest year examined means by then. A second table lists niches flagged in an earlier year but not now, with their shape: fields that peaked or slowed. `first_flagged`, `last_flagged`, `first_seen`, and `first_seen_share` are columns of `niches.csv`, and `timeline.csv` has every niche and year.

**Emerging terms** ranks words and bigrams by growth z times their concentration in a few niches, so field vocabulary outranks phrasing that is spreading through every abstract. `--watch` adds yearly document counts for terms you name.

The report embeds seven figures from the `plots/` directory; `--no-plots` skips them:

- `areas.png`: each area's yearly share, with the growth window shaded and the fitted trend, curved when a shape was found;
- `area_growth.png`, `topic_growth.png`, and `niche_growth.png`: each cluster's yearly share change against its size, one tree level per figure, blue where the share is rising and red where it is falling. Areas and topics carry their 95% intervals. The niche figure leaves them out, as they would hide several hundred dots, rings the flagged niches, and names the ten top-ranked;
- `niches.png`: the shares of the twelve top-ranked niches, with lines at the years each was first seen and first flagged;
- `timeline.png`: the timeline table as a heatmap, with flags as dots and the first-seen year boxed;
- `watch_terms.png`: the share of each year's papers that use each `--watch` term.

## Check it with the backtest

The **backtest** repeats the niche analysis on the papers up to each of the last few cutoff years (`--backtest-cutoffs`), assigns the papers of the following `--horizon` years to the nearest niche, and reports the Spearman correlation between each signal at the cutoff and the niche's later change in share. On the 20,000-paper sample above (13,642 abstracts), the score correlated 0.27–0.39 with the change in share over the next three years, 0.31 pooled. Paper count alone correlated 0.12. From three of the four cutoffs, the five top-ranked niches grew their share while the rest shrank. Novelty correlated negatively (−0.21), so it is reported but not part of the score. The same query with `in:all` gave 0.12. A weak backtest says the niche ranking on that corpus should not be trusted. The field map's areas and topics each rest on more papers, so their statuses are less noisy.

## Limits

- The score measures momentum. A field is found once it is measurably growing (by default eight papers, three of them recent), not before. A niche that appeared in 2023 can be flagged with 2024 data at the earliest.
- It does not measure importance. A small, fast-growing niche ranks above a large field holding steady; use the field map for the latter.
- Results depend on the query and on abstract coverage, so record both alongside any conclusion.
- Clusters are fixed across years. The timeline follows today's niches back, but a field that split or merged shows only as the overlap of its papers, not as a lineage of its own.
- One clustering tree holds the pairwise distance matrix in memory, about 10 GB at the 50,000-paper limit. Sample larger corpora evenly across years. The backtest and timeline build one more tree for each year they examine.

Granularity is set by `--areas`, `--topic-size`, and `--niche-size`. Smaller topics separate fields that share vocabulary, such as lithium–sulfur cathodes and sulfide electrolytes.
