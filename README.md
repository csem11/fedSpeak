# FedSpeak

A small GPT trained from scratch, character by character, on thirty years of
Federal Reserve prose: FOMC statements, FOMC minutes, and Beige Books. Built to
show *how next-token pretraining works*, and where it stops.

## What this demonstrates, and what it does not

**It demonstrates** that a language model trained only to predict the next
character from raw text will absorb the *register* of that text: the sentence
rhythms, the section structure, the hedged institutional vocabulary, the
formulaic openings and closings. Everything the model produces comes from that
single objective. There is no rule about what a central bank is, no notion of
inflation, and no labels of any kind.

**It does not demonstrate**, and must not be read as demonstrating, anything
about monetary policy:

- It does not model or reason about actual Fed decisions.
- It does not learn causal relationships between economic data and rate moves.
- The numbers, dates, and names it generates are plausible-looking *shapes*,
  not facts. They are wrong in the way a fluent parody is wrong.
- Nothing here is grounded, verified, or suitable for analysis of any kind.

This is a **mechanism demo, not a policy tool**. The interesting result is that
so much style survives such a stupid objective, and that so little substance
does.

## Pipeline

Four scripts, run in order. Dependencies are `requests`, `beautifulsoup4`, and
`torch` (`pip install -r requirements.txt`).

| Step | Script | What it does |
|---|---|---|
| 1 | `scrape_fed.py` | Collects every document from federalreserve.gov, cleans it, writes one text file per document and one concatenated corpus |
| 2 | `prepare.py` | Maps each distinct character to an integer, splits 90/10 chronologically, writes `train.bin` and `val.bin` |
| 3 | `train.py` | Trains the GPT from random initialisation; logs loss curves to CSV; saves the best checkpoint |
| 4 | `sample.py` | Generates text from the checkpoint |

```
python scrape_fed.py        # ~10 min first time (rate-limited); seconds afterwards from cache
python prepare.py
python train.py             # ~30 min on Apple Silicon (MPS); see --help for sizing knobs
python sample.py --prompt "The Committee decided to"
```

## The data

All three sources are official Board of Governors publications and are
scraped from federalreserve.gov's own archive pages, not from mirrors.

| Source | Years | Documents | Characters | Notes |
|---|---|---|---|---|
| FOMC statements | 1994-2026 | 246 | 0.6M | Sparse before 1999: the FOMC only issued statements when it changed policy |
| FOMC minutes | 1994-2026 | 261 | 12.0M | |
| Beige Books | Oct 1996-2026 | 240 | 24.3M | National summary plus all twelve district reports. Earlier issues exist only as PDF and are skipped |

Cleaning strips site navigation, press-release headers and footers, footnote
markers, and comment blocks; maps typographic quotes and dashes to ASCII; and
normalises whitespace. The result is 36.8M characters over an 86-character
vocabulary. `data/raw/` keeps every page exactly as downloaded so cleaning can
be redone without touching the network, and `data/clean/` keeps one file per
document so any single one can be inspected.

Two things about the mix are worth knowing. The Beige Book is two thirds of
the corpus, so the model's default voice leans toward district economic
narrative. And the corpus is concatenated in date order, so the validation set
is the most recent documents: the model is scored on prose written after
everything it trained on.

Reconnaissance notes: the Fed changed its URL scheme five times and its page
template four times since 1994, several older pages are ISO-8859-1 rather than
UTF-8, the Fed's own 2003 Beige Book index omits the September issue, and the
June 2008 minutes live at a URL that doesn't contain the word "minutes". The
scraper follows link *labels*, which have been stable, rather than URL
patterns, which have not.

## The model

`model.py` is Andrej Karpathy's nanoGPT model, unmodified apart from removing
the GPT-2 weight loader and an A100 utilisation estimate. `train.py` is a
simplified rewrite of nanoGPT's training loop with the distributed, logging,
and compilation machinery removed so the mechanism is visible in ~150 lines.

| | |
|---|---|
| Tokenisation | character-level, 86 symbols, no BPE |
| Architecture | 6 layers, 6 heads, 384-wide, context 256 chars |
| Parameters | 10.65M |
| Training | 2000 iterations x 64 x 256 chars = 33M chars, about one pass over the corpus |
| Optimiser | AdamW, lr 1e-3 with 100-iter warmup and cosine decay to 1e-4, weight decay 0.1 |
| Hardware | Apple M4 via MPS, ~800 ms/iteration, 29 minutes end to end |

These are nanoGPT's `shakespeare_char` settings with dropout turned off, since
a single pass over 33M characters has no overfitting risk.

## Results

Full logs are in `results/` (`eval_log.csv`, `step_log.csv`) and every sample
quoted below is in `results/samples.txt`, generated at temperature 0.8, top-k 40.

### Loss curve

Loss is cross-entropy in nats per character. 4.45 is what uniform guessing over
86 characters scores; the model starts there.

| iteration | train | val | elapsed |
|---|---|---|---|
| 0 | 4.575 | 4.575 | 0 min |
| 200 | 1.746 | 1.734 | 3 min |
| 400 | 1.080 | 1.091 | 6 min |
| 800 | 0.838 | 0.869 | 12 min |
| 1200 | 0.726 | 0.771 | 18 min |
| 1600 | 0.662 | 0.715 | 23 min |
| 2000 | 0.637 | 0.698 | 29 min |

Train and val move together the whole way, with a small gap opening after
iteration 1000. There is no overfitting at one epoch, and the model would keep
improving with more compute. The point of the demo is not the number; it is
what the number buys and what it does not.

### What the model sounds like as it learns

At iteration 400 (val loss 1.09), prompted with "The Committee decided to":

> The Committee decided to the Bank of the nance of accommodity of the year
> that sized by momewhat and on economic activity rates in decreased-months,
> noted slightly of contacts and cains, paper that contacts expected improved
> to modest in competity.

It has the vocabulary's *shape* before it has the vocabulary: "contacts",
"modest", "slightly", "economic activity" are real Beige Book words, and
"accommodity", "momewhat", "competity" are what Fed-speak looks like from
inside a character model that has only seen 6.5M characters.

At iteration 2000 (val loss 0.70), the same prompt:

> The Committee decided to consider the extent to which the conditions
> restraining inflation of the economic outlook.
>
> The Committee continued to assess the economic outlook the outlook in
> business conditions and will employ into the implications of System Open
> Market Account in advanced foreign currencies. [...]
>
> Committee Policy Action
> In their discussion of monetary policy for this meeting, meeting
> participants judged that the stance of monetary policy and composition and
> the implications for the Committee's objectives for maximum employment and
> price stability.

Prompted with "Minutes of the Federal Open Market Committee":

> Minutes of the Federal Open Market Committee
>
> January 1, 2100
>
> A joint meeting of the Federal Open Market Committee and the Board of
> Governors of the Federal Reserve System in April and May 38, 2002, at 9:00
> p.m.
>
> Present:
>
> Mr. Greenspan, Chairman
> Mr. Gramlich, Manager, System Open Market Account
> Ms. Oliner, Deputy Secretary
> Mr. Boehne, General Counsel
> Mr. Alvarez, General Counsel

Prompted with "Consumer Spending", a Beige Book section heading:

> Consumer Spending
> Retail sales continued to decline through September, and general retailers
> expected steady sales to stronger in the first half of the country, although
> they are foreseen as a strong pickup in spending [...]
>
> First District - Boston
>
> The First District economy continue to expand in the First District and the
> weakness in since our last report and high demand. Several contacts noted
> that high gas prices for professional and parts solid growth

### What it learned

- **Register.** The hedged, passive, committee voice: "participants judged
  that", "contacts noted that", "the Committee continued to assess".
- **Document structure.** A minutes document opens with a title, a date, a
  "joint meeting ... was held" sentence, and a "Present:" roster of
  "Mr./Ms. Surname, Role" lines. A Beige Book runs section headings followed
  by district headings. A prompt naming one source reliably produces that
  source's format, not another's.
- **Vocabulary and collocations.** "System Open Market Account", "agency
  mortgage-backed securities", "maximum employment and price stability",
  "reverse repurchase agreements", all spelled correctly, from a model that
  has never seen a word boundary as a unit.
- **Real names in real roles.** Greenspan is Chairman, Alvarez is General
  Counsel. These are memorised co-occurrences, not knowledge.

### What it did not learn

- **Anything true.** "January 1, 2100". "May 38, 2002, at 9:00 p.m." The
  dates have the right shape and impossible values, because the model learned
  the *format* of a date and nothing about calendars. Every number, vote, and
  rate in the samples is fabricated the same way.
- **Meaning across a sentence.** Clauses are locally fluent and globally
  empty: "will employ into the implications of System Open Market Account in
  advanced foreign currencies". Each five-word window is plausible; the
  sentence is not about anything.
- **Roles that stay consistent.** Two people are General Counsel in one
  roster. Nothing in the objective penalises that.
- **When to stop.** Prompted with "Federal Reserve Bank of Boston", it emits
  a dozen more "Federal Reserve Bank of ..." headings in a loop before
  escaping. Small models sampled at moderate temperature fall into
  repetition because the most likely continuation of a list is more list.
- **Anything about policy.** There is no representation here of why rates
  move, what inflation is, or what the Committee would do. Ask it what the
  Fed will decide and it will produce a sentence in the right voice that
  means nothing, which is precisely the failure mode this demo exists to make
  visible.

## Experiments and further material

[EXPERIMENTS.md](EXPERIMENTS.md) changes one knob at a time (context length,
model size, learning rate) and measures what moves the loss, then looks at
per-source difficulty, sampling temperature, and how the samples change along
the training run. Supporting material lives in `results/`:

| File | What it is |
|---|---|
| `results/figures/` | loss curve and ablation charts |
| `results/experiments/<run>/` | eval and step logs for every ablation run |
| `results/loss_by_source.csv` | validation loss of the final model on each source |
| `results/temperature_sweep.txt` | the same prompt sampled at five temperatures |
| `results/samples_by_iteration.txt` | samples from checkpoints along the main run |
| `results/annotated_sample.md` | one generated document with every fabrication marked |
| `results/scale_table.csv` | this run beside GPT-2, GPT-3 and Llama 3.1 in tokens and FLOPs |
| `results/corpus/` | documents per year, source shares, the 86-character vocabulary, a raw-to-clean example |
| `results/finetune/` | phase 2: fine-tune logs, base and fine-tuned samples, per-source loss for both |

## Phase 2: fine-tuning a pretrained model

[FINETUNE.md](FINETUNE.md) takes the opposite approach: instead of learning
English from nothing, it continues pretraining SmolLM2-360M on the same corpus
for one hour on a 16 GB laptop. The base model already beats the character
model on Fed text before seeing any of it (0.86 vs 1.00 bits per character),
fine-tuning brings it to 0.68, and the samples go from clause-level to
paragraph-level coherence while fabricating exactly as much. The scripts are
in `finetune/` and need two extra dependencies (`requirements-finetune.txt`).

## Attribution

Model code: [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy,
MIT licence. Training text: publications of the Board of Governors of the
Federal Reserve System, which are works of the United States government.
