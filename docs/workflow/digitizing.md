# Recovering curves from figures

Papers often report a result only as a plotted curve: a mass loss followed over weeks, a conductivity against temperature. A recipe can ask for the figure that plots a record's result, as the `polymer_degradation` recipe's `Mass loss figure` field does, and `pmt digitize` then recovers that figure's curves as numbers with [plottracer](https://github.com/SMTG-Bham/plottracer).

plottracer reads each panel's axes from its tick labels and titles by OCR, and traces every curve, telling curves apart by colour, line style and markers. It does not read legends, so each curve is named by how it is drawn, such as `1f77b4-solid` (its colour in hexadecimal, then its style), not by the sample it belongs to. Matching curves to samples is left to the reader, helped by the figure caption, the preview images, and the records that cite the figure, which every output row lists.

## Installation

plottracer is a dependency of PaperMinerToolkit on Python 3.14 or later on Linux, the only platform it supports, and is installed with it, including the OCR extra that reads axis labels. On other Python versions or operating systems it is not installed and `pmt digitize` reports that it is unavailable. To add it to an existing Linux environment on Python 3.14:

```bash
pip install "plottracer[ocr]"
```

PaperMinerToolkit and plottracer depend on `opencv-python`. An environment that already has `opencv-python-headless` ends up with both builds; uninstall the headless one if OpenCV imports fail.

## Usage

Run it on a scraped or stored CSV, naming the column that holds each record's figure:

```bash
pmt digitize papers.db scraped_polymer_degradation.csv \
  --figure-field "Mass loss figure" \
  --only-missing "Mass loss curve" \
  --output mass_loss_curves.csv \
  --previews mass_loss_previews
```

`--only-missing` follows only the records whose value in that column is missing, here the records that name a mass-loss figure but report no curve values. Without it, every record that names a figure is followed, which also gives a check on the curves a model did transcribe.

| Option | Meaning |
| --- | --- |
| `--figure-field` | Column naming the figure that plots each record's result. Required. |
| `--only-missing` | Only follow records whose value in this column is missing. |
| `--output` | CSV to append the recovered points to. Default `digitized_figures.csv`. |
| `--previews` | Directory for a preview of each recovered figure: its curves drawn over a faded copy. |
| `--detect/--no-detect` | Detect and store figures from a paper's PDF when the corpus holds none for it. Default on. |
| `--limit` | Stop after this many figures, to try a few first. |
| `--force` | Recover figures already recovered. |
| `--json` | Print the run summary as JSON. |

## How figures are found

Each record's figure field is read for figure numbers and panels in the forms papers and models write them: `Figure 4`, `Fig. 4a`, `Figure 4(b)`, `Figs. 2 and 3`, `Figures 2-4`, `Figure 5a-c`, `Figure 7a, b`. The records citing one figure of one paper are gathered into a single request, so each figure is recovered once, with the union of the panels its records name; a record naming the whole figure asks for every panel.

The figure image comes from the corpus as the [layout image mode](scraping.md#choosing-which-images-the-model-sees) uses it: a structured figure downloaded with the article is preferred; a paper with no stored figures has them detected from its PDF and stored, so later runs, and image scraping, reuse them. Figures are matched by the number in their label, such as `Figure 3`, or failing that in their identifier, such as `f003`.

To prepare or inspect the figures first, run `pmt figures download` (publisher images from stored XML) or `pmt figures detect` (crops from stored PDFs), then `pmt figures list` or `pmt figures export` to check what was found; see the [figure commands](scraping.md#choosing-which-images-the-model-sees).

Supplementary figures (`Figure S4`, `Supplementary Figure 2`) are counted but not recovered: supporting-information files are not split into figures.

## Output

The output CSV has one row per recovered point:

| Column | Meaning |
| --- | --- |
| `Paper id`, `doi` | The paper. |
| `Figure id`, `Figure label`, `Figure source` | The stored figure that was read, and where it came from. |
| `Referenced as` | The distinct figure references that asked for it. |
| `Record rows`, `Record names` | Index labels and `Name` values of the citing records in the input CSV. |
| `Panel` | The panel letter read from the figure, `inset`, or the panel's position when no letter was read. |
| `Series`, `Series style` | The curve: its colour and style, such as `1f77b4-solid`, and its style alone (`solid`, `dashed`, `dotted`, `dashdot` or `markers`). |
| `X title`, `Y title` | The axis titles as read, including their units, or `x` and `y`. |
| `x`, `y` | The point, in the axes' own units. |

Rows are appended, so a resumed run adds to the same file. Use `--force` with a fresh output path to recover everything again into a new file.

## Checkpoints and failures

Each figure's outcome is stored in its corpus asset as `digitization_status` and `digitization_error`, separately from image-extraction progress, so a rerun skips figures already recovered and retries the rest. A figure fails, with plottracer's reason recorded, when its axes cannot be read: fewer than three tick labels in line on an axis, labels drawn as outlines, or a second y-axis on the right. A cited panel letter that plottracer cannot read keeps every panel of the figure, and the run says so.

Check the previews before relying on the numbers. Curves that touch or cross can swap where they meet, legends without a frame in a curve's colour are traced as curves, and bitmap figures carry the resolution they were rendered at. Series drawn in one colour and told apart only by marker shape, as in a black-and-white scatter plot of several samples, come back as one series of markers and pieces of line: plottracer separates them only when given each series' legend marker, which `pmt digitize` does not supply. See plottracer's [limitations](https://github.com/SMTG-Bham/plottracer#limitations).
