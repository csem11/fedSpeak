# FedSpeak style guide

How this project writes, reports numbers, draws charts and builds its page.
It records conventions the repo already follows, so that anything added later
reads as part of the same piece of work. Where the repo currently breaks one of
these rules, the break is listed under [Known deviations](#8-known-deviations)
rather than hidden.

One rule outranks all the others: **the data wins.** Never change a number,
round one differently, or drop an inconvenient run to fit this guide or a
narrative. Change the prose instead.

---

## 1. Framing and voice

### The frame is fixed

This is a **mechanism demo, not a policy tool.** Every page, write-up or sample
file that shows model output must make clear that:

- the models learn register and structure, not facts about the Fed;
- every date, name, number and vote they generate is fabricated by construction;
- nothing here models Fed decisions or the link between data and rates.

The README, the page's opening callout and each sample file already carry this.
New surfaces need it too.

### Voice

| Do | Don't |
|---|---|
| Lead with the finding, then the evidence | Build up to the conclusion |
| One idea per sentence | Chain clauses with semicolons when two sentences would do |
| Past tense for what was done, present for what the results show | Mix tenses within one claim |
| Describe behaviour: *writes*, *produces*, *assigns* | Claim comprehension: *understands*, *believes*, *decides* |
| Put the caveat next to the claim it limits | Save every caveat for a section at the end |
| Say when a conclusion changed and why | Silently overwrite an earlier result |

The chunk-size section is the model for the last row. It keeps the four-point
sweep, the seven-point U and the fair re-scoring visible in sequence, because
how the answer moved is part of the result.

### Headings

- **Results sections state the finding:** "Context length: a real optimum, not
  a plateau", "Scored fairly, the answer moves", "It did not transfer".
- **Utility sections stay plain:** Setup, Reproduce, Caveats.
- Never a heading like "Results" or "Analysis" on its own.

### Emphasis and punctuation

- **Bold one sentence per section at most,** the one carrying the conclusion.
  Never bold a whole paragraph.
- No em dashes or en dashes. Use a colon, a comma, or a new sentence.
- Ranges are written in words: "0.901 to 0.907", never "0.901–0.907" or "±".
- Avoid *significant* unless a statistical test was run. None have been.

---

## 2. Numbers

### Units

| Context | Unit | Written as |
|---|---|---|
| Character model, on its own | nats per character | prose: "0.698 nats per character"; labels: `nats/char` |
| Any comparison across tokenizers | bits per character | prose: "0.60 bits per character"; labels: `bits/char` |
| Word-piece models, on their own | nats per token | only when the tokenizer is fixed |

Bits per character is the only fair unit across a character model and a
word-piece model. Convert with:

```
bits/char = (nats per token / ln 2) / (characters per token)
```

Characters per token are 5.39 for SmolLM2 and 5.55 for Qwen3 on this corpus.

### Precision

The noise floor sets the precision. Re-running the same seed moves validation
loss by up to **0.005**, because the GPU kernels are not bit-for-bit
deterministic.

- Report losses to **at most three decimals**; chart labels round to two. The fourth decimal is noise.
- The only place four decimals appear is the noise-floor table, where the tiny
  differences are the point.
- Treat any difference under **0.01** as noise and say so.
- A claim that one setting beats another needs **more than one seed**, and the
  ranges across every run at each setting must not overlap.

### Ranges and seeds

- Report the full range across every run at a setting, including same-seed
  re-runs, not a mean with ±.
- Name the seeds. The default is 1337; the extra seeds are 1338 and 1339.

### Other quantities

| Quantity | Style | Example |
|---|---|---|
| Parameters | two significant figures past the unit | 10.65M, 1.74B, 17.4M |
| Tokens and characters | millions with a unit | 1.31M tokens, 36.8M characters |
| Wall time | minutes, one decimal, from an idle machine | 5.2 min |
| Loss labels | say where it was measured | "val loss @ 1000 iters" |

### Where numbers come from

- **Every number in prose must exist in a file under `results/`.** Prefer
  generating tables from the CSVs over typing them.
- **The reported numbers are the original sweep** in `results/experiments/`.
  Same-seed re-runs live in `results/experiments_rerun/` and are used only for
  timings and the noise floor. Extra seeds live in `results/experiments_seeds/`.
- `runs/` is scratch. Nothing in a write-up may cite it directly.

---

## 3. Terminology

| Use | Not | Meaning |
|---|---|---|
| chunk size | context length, block size (in prose) | length of each training window; `--block_size` in code |
| context | | how much history a single prediction can see |
| own-window loss | training loss, reported loss | each model scored in windows of its own chunk size |
| fair scoring, same targets | common-context loss | every model scored on identical characters with its full context |
| fine-tuned, fine-tuning | finetuned, fine tuned | `finetuned` is fine in code identifiers only |
| LoRA fine-tune | adapter training | the Qwen3 run: frozen weights, rank-16 adapters |
| full fine-tune | | every weight trains: the SmolLM2 run |
| untouched | base, vanilla, raw | a pretrained model before any Fed text |
| character model | char-level GPT, nanoGPT model | the 10.65M model trained from scratch |
| FOMC statement, FOMC minutes, Beige Book | Beige book, statements (alone, when ambiguous) | the three sources, always capitalised this way |
| SmolLM2-360M, Qwen3-1.7B | SmolLM, Qwen 1.7B | exact model names on first use in a section |
| word-piece | BPE (in reader-facing prose) | subword tokenization; BPE is fine in code comments |
| validation set | test set, holdout | documents from 20 September 2023 onward |

---

## 4. Generated text

Model output is the project's most persuasive and most misleading material.

- **Label it every time.** State the model, checkpoint, sampling settings and
  seed. Sample files open with a header saying the text is generated and that
  nothing in it is fact.
- **Don't cherry-pick.** Use the fixed seed, 1337. If a sample was chosen from
  several, say how.
- **Verify before you quote a claim about output.** "The district list appears
  112 identical times" was checked against `data/clean/` before it was written.
  Every factual statement about what the corpus does or doesn't contain needs
  the same check, and the count goes in the text.
- **Mark fabrication precisely.** The annotated sample uses four tags, and new
  annotations reuse them:

| Tag | Means |
|---|---|
| form | structure reproduced correctly |
| fabricated | right shape, false content |
| drift | locally fluent, globally meaningless |
| bug | a character-level slip |

On the page, generated text is set in the mono face on a `--paper-2` panel,
with the prompt in `--accent`.

---

## 5. Visual design

### Colour tokens

All colour comes from tokens on `:root`. Components never use a hex value
directly. The one exception is the navigation's scroll-fade mask, where black
only sets transparency and is never seen.

| Token | Light | Dark | Role |
|---|---|---|---|
| `--paper` | `#f6f5f0` | `#121615` | page ground |
| `--paper-2` | `#edece5` | `#1a1f1d` | panels, text documents |
| `--paper-3` | `#e3e2da` | `#232927` | window chrome, chips, status bars |
| `--ink` | `#151b18` | `#e9e8e0` | primary text |
| `--ink-2` | `#4d5753` | `#b4bab5` | secondary text, captions, axis labels |
| `--ink-3` | `#7a837e` | `#7f8783` | muted labels, section numbers |
| `--rule` | `#d9d8cf` | `#2c3330` | borders, gridlines |
| `--accent` | `#1d6b4f` | `#52b98c` | links, eyebrows, prompts, focus rings |
| `--accent-soft` | `#d8ebe2` | `#1d3a2e` | callout ground |

The neutrals lean slightly green toward the accent, a deliberate choice rather
than a default grey.

### Themes

The page supports three theme states: light, dark and the viewer's system
setting.

- Define the full light palette on bare `:root`.
- Redefine **only tokens** under `@media (prefers-color-scheme: dark)`, guarded
  by `:root:not([data-theme="light"])`, and again under
  `:root[data-theme="dark"]`.
- Style components through tokens only, never inside a theme block.
- Every new colour gets a light and a dark value. A colour that exists in only
  one theme is a bug.

### Typography

All three families are IBM Plex, loaded from Google Fonts, each with a system
fallback.

| Role | Face | Size | Notes |
|---|---|---|---|
| Page title | Plex Serif 500 | `clamp(2.6rem, 7vw, 4.6rem)` | letter-spacing −0.01em |
| Section heading | Plex Serif 500 | `clamp(1.7rem, 3.5vw, 2.3rem)` | `text-wrap: balance` |
| Subheading | Plex Serif 500 | 1.2rem | |
| Body | Plex Sans 400 | 16px, line-height 1.6 | max 68 characters wide |
| Captions | Plex Sans | 0.86rem, `--ink-2` | max 72 characters wide |
| Eyebrows | Plex Mono | 0.75rem, uppercase, 0.12em tracking | `--accent` |
| Data, code, labels, generated text | Plex Mono | 0.68 to 0.92rem | tabular figures for numbers |
| Stat tiles | Plex Mono 500 | 1.7rem | `tabular-nums` |

The mono face means "this is data or machine output". Don't use it for
ordinary prose.

### Layout

| Element | Width |
|---|---|
| Content column | max 920px |
| Navigation bar | max 1100px |
| Running text | max 68 characters |
| Side gutter | 16px at every width |

- Sections are numbered 00 to 06 because the page is a sequence. Don't number
  things that aren't.
- Lay out siblings with grid or flex and `gap`, not per-element margins.
- Only tables and code may scroll sideways, each in its own container. The page
  never does.
- In the navigation bar, only the section links scroll. The GitHub link and the
  theme toggle are pinned so they can never be pushed off-screen.

### Components

| Component | Use it for |
|---|---|
| Callout (`--accent-soft`, left rule) | the framing statement, one per page |
| Stat tiles | headline figures: the hero and the corpus shares |
| Document panel (mono, `--paper-2`) | generated or source text |
| Window (title bar, tabs, status bar) | the interactive playground only |
| Tag highlights | annotated generated text |
| Verdict pair | the usable / not-usable summary |

Not everything is a card. Borders and fills mark a separate object; don't wrap
ordinary prose in them.

### Accessibility

- Every interactive element shows a visible focus ring in `--accent`.
- Honour `prefers-reduced-motion`. The playground's typing effect turns off.
- Every chart has an `aria-label`. Colour never carries meaning alone: series
  have direct labels or a legend.

---

## 6. Charts

### Series colour

A fixed order, validated for colour-blind separation. Assign in order and
never cycle. Within one chart a colour means one thing. Across charts, the same
entity keeps the same colour where it appears: the character model is always
`--s2`, fine-tuned SmolLM2 always `--s1`, Qwen3 always `--s4`.

| Slot | Light | Dark | Current use |
|---|---|---|---|
| `--s1` | `#2a78d6` | `#3987e5` | first series; own-window loss; SmolLM2 fine-tuned |
| `--s2` | `#eb6834` | `#d95926` | second series; fair scoring; character model |
| `--s3` | `#1baf7a` | `#199e70` | third series; SmolLM2 untouched |
| `--s4` | `#4a3aa7` | `#9085e9` | fourth series; Qwen3-1.7B |

- **Four series at most per chart.** With more, plot one series, or show
  representative members: the chunk-size chart shows all seven sizes as a
  single curve and four of them over time.
- `--s3` is below 3:1 contrast on the light ground, so any chart using it needs
  visible labels or a legend. All current charts have both.
- The annotation tags (`--tag-*`) are a separate palette. Never use them for
  data series.

### Marks and labels

- Lines 2px. Markers radius 4, radius 5.5 when highlighted.
- Direct end labels in mono 11px with a `--paper` halo, spread apart so they
  never collide.
- Label only what matters: the minimum, an end point. Never a number on every
  point.
- **Clip data to the plot box.** A series that starts off-scale enters from the
  edge; it never draws over the prose above. Labels stay outside the clip so
  they can sit in the margin.
- Every chart responds to hover. Line charts show a crosshair and a tooltip,
  skipping points outside the visible range; bar charts show a tooltip per bar.

### Axes

- The y-axis title sits above the plot, clear of the top tick label.
- Axis titles carry the unit: "val loss (nats / char)".
- Gridlines are `--rule`, horizontal only. They should recede.
- Half-width charts use a narrower drawing width so 11px text stays legible.

### Legends and captions

- **Generate legends from the same data as the series.** Two hand-written
  legends silently fell out of sync with their charts when a fourth model was
  added. A legend is never typed out by hand.
- Captions state the takeaway, not a description of the axes. "Qwen3-1.7B starts
  where SmolLM2 finishes", not "Loss against tokens for two models".
- Whiskers span every run at a setting: all seeds and any same-seed re-run.
  The caption says so.

### Static figures

`plot_results.py` draws the PNGs used by the markdown write-ups. It uses the
same four series colours as the page. It uses different neutrals, which is a
known deviation below.

---

## 7. Files and process

### Source of truth

| Path | Status |
|---|---|
| `results/` | source of truth for every reported number, committed |
| `runs/`, `out/`, `finetune/out*/` | scratch and checkpoints, gitignored |
| `site/template.html` | the page source: edit this |
| `docs/index.html`, `site/artifact.html` | generated by `site/build.py`: never edit by hand |
| `data/clean/` | cleaned corpus, committed |
| `data/raw/`, `data/*.bin`, `data/ft*/` | reproducible, gitignored |

### Adding an experiment

1. Run it on an otherwise idle machine, and check nothing else is using the GPU.
2. Confirm it reached its final iteration. A run that crashed is reported by
   name, never silently skipped.
3. Copy its logs into the right folder under `results/`.
4. If it supports a comparison, run more seeds and report ranges.
5. Write it up from the CSV, not from memory.
6. Rebuild the page with `python3 site/build.py`, then look at it once in both
   themes before publishing.

### Commits

- The subject is an imperative summary of what changed.
- The body explains why, and states any result the change produced.
- Say so when a commit corrects an earlier claim, and name the claim.
- End with the co-author line when Claude contributed.
- Commits use the GitHub no-reply address. This repo's local git config
  already does.

### Privacy

- No personal email addresses in code, request headers or commit metadata.
  The scraper identifies itself with a link to the repository instead.
- No absolute home-directory paths in committed files.

---

## 8. Known deviations

Places the repo currently breaks this guide. Fix them when the file is next
touched for another reason, and remove the line when you do.

| Where | Deviation | Rule |
|---|---|---|
| `EXPERIMENTS.md`: the heading "Context length: a real optimum, not a plateau", a sentence in Setup and one in Caveats | "context length" where chunk size is meant | §3 terminology |
| `README.md`: the experiments summary paragraph | "context length" where chunk size is meant | §3 terminology |
| `site/template.html`, chunk-size chart x-axis | labelled "context length" | §3 terminology |
| `plot_results.py` | neutrals `#fcfcfb` / `#0b0b0b` / `#52514e` differ from the page's `--paper` / `--ink` / `--ink-2` | §5 colour tokens |
| Chart axis titles vs markdown tables | "nats / char" with spaces on the page, "nats/char" in tables | §2 units |
