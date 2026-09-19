# One generated document, annotated

Prompt: `Minutes of the Federal Open Market Committee`. Model: `out/ckpt.pt`
(10.65M params, 2000 iterations, val loss 0.698). Temperature 0.8, top-k 40,
seed 1337. The full text is in `samples.txt`; this is the first 900 characters
with every line commented.

Legend: **[form]** = structure the model reproduced correctly.
**[fabricated]** = right shape, false content. **[drift]** = locally fluent,
globally meaningless. **[bug]** = a character-level slip.

```
Minutes of the Federal Open Market Committee
```
**[form]** The prompt. Every real minutes document since 1994 opens this way.

```
January 1, 2100
```
**[form]** A date on its own line follows the title in every real document.
**[fabricated]** The year 2100 appears nowhere in the corpus. The model
learned "a month, a day, a comma, four digits starting with 19 or 20", and
"21" is a plausible continuation of "20"-something. It has no concept that a
year is a number.

```
A joint meeting of the Federal Open Market Committee and the Board of
Governors of the Federal Reserve System in April and May 38, 2002, at 9:00
p.m.
```
**[form]** The sentence template is exact: "A joint meeting of the Federal
Open Market Committee and the Board of Governors of the Federal Reserve
System was held in the offices of the Board of Governors ... on Tuesday,
January 27, 2026, at 10:00 a.m."
**[drift]** "was held in the offices ... on" collapsed into "in April and".
**[fabricated]** "May 38". Days run 1 to 31 in every one of the 261 real
documents; the model has never seen "38" after "May" but has seen many
two-digit numbers after month names. "9:00 p.m.": the corpus has 240 meetings
starting at 9:00 a.m. and none starting in the evening.

```
Present:

Mr. Greenspan, Chairman
```
**[form]** The roster block, exactly as formatted from 1994 to about 2010.
**[fabricated, but real]** Greenspan is on the roster of 97 of the 261 minutes in
the corpus (37%). "Mr. Greenspan," is followed by "Chairman" essentially every
time it appears at the start of a roster line, so this is memorised
co-occurrence, the one place where the model's output is accidentally true.
Note the roster style (Mr./Ms. Surname) is the pre-2011 format, consistent
with Greenspan, so the model also kept the *era* consistent for one line.

```
Mr. ki meeting on Secretary
```
**[bug]** A surname that failed to form. "Mr. " is followed by a capital
letter in every real line; here it sampled a lowercase "k" at temperature
0.8 and could not recover into a name.

```
Mr. Gillum, on August 18, 1997, until Procedure
Mr. Gramlich, Manager, System Open Market Account
```
**[fabricated]** In the corpus Gillum is always "Assistant Secretary" (74
roster lines) and Gramlich is always listed bare, as voting members are.
"Manager, System Open Market Account" is a real title that belongs to other
people. Roles are assigned by what sounds right after a comma, not by who
held them.

```
Mr. General CounselVrey Deputy Secretary
```
**[bug]** A title where a surname should be, then a fused token "Vrey". The
model tracks "we are in a roster line" but at 0.8 the character-level
sampling occasionally jumps to the wrong part of the template mid-line.

```
Ms. Oliner, Deputy Secretary
Mr. Boehne, General Counsel
Ms. Bridge, Deputy Secretary
Mr. Coyne, Assistant Secretary
Mr. Alvarez, General Counsel
```
**[form]** Five perfectly formed roster lines in a row with real FOMC
staff titles. Four surnames are real; "Bridge" appears on no roster in the
corpus and is an invented name that merely looks like one.
**[fabricated]** Two people hold General Counsel and two hold Deputy
Secretary. Nothing in the next-character objective penalises repeating a
role three lines later; a 256-character context can see the repetition but
was never trained to avoid it. Alvarez is "General Counsel" on all 43 of
his roster lines, so that line is accidentally right. Boehne is listed as a
voting member, and Oliner as "Senior Adviser, Division of Research and
Statistics", never Deputy Secretary.

```
Ms. Minehan, Senior Economist, Division of Monetary Affairs, Board of
Governors
```
**[form]** The long-form staff line, with division and institution, is the
correct format for that part of the roster.
**[fabricated]** Minehan appears 36 times as a bare voting member (she was
President of the Boston Fed), never as a staff economist.

## What to take from this

Every line has the right *shape* for its position in the document. Almost
nothing in it is *true*. That is the entire content of the demo: next-
character prediction over 37 million characters buys you the form of Fed
prose in full, and buys you nothing about the world the prose describes. A
reader who did not know the corpus could not tell the fabricated roster from
a real one without checking. A reader who wanted to know who was General
Counsel in 2002 would be misled with complete confidence.
