# Results, stage by stage

How every setting of the app was chosen: what I compared at each stage, the
numbers, and the decision. The stages ran in this order.

Every retrieval number comes from the files in `eval/`: one row per run in
`eval/experiments.csv`, and each run's question-by-question results in
`eval/runs/<run>/per_question.jsonl`. `python -m src.eval.compare <A> <B>
--k <k>` recomputes any comparison below, with the `k` of its table (add
`--k-b` when the two options send a different number of chunks, as in
512 × 5 against 256 × 8). The answer runs are in `eval/generation/runs/`.

## The starting setup

Each stage changed one setting and kept the rest as below; the winner then
became the starting point for the next stage.

| Setting | At the start |
| --- | --- |
| Chunking | `section_t512_o64` (512 tokens, respecting sections, overlap 64; see *Chunking*) |
| Embedding | `nomic-embed-text-v1.5` |
| Search | hybrid, `alpha` 0.5 |
| Chunks sent (`k`) | 5 |
| Chunks per section (cap) | 2, a safeguard from the design |

## How to read the tables

- **The questions.** One question per eligible section of the corpus, 1,771 in
  all, split once into a **development set of 1,240** (every decision below)
  and a **test set of 531**, kept aside until the very end.
- **The metric that decides.** `hit@k`.
  Next to it: **MRR** and **doc_hit** (the right *article* is among the chunks sent, even
  if not the right chunk), **hit@20** and the **context** the model reads,
  in tokens.
- **The rule, fixed before any result.** Two options are compared question by
  question (a *paired* comparison: every question counts, but only the ones
  where they disagree move the result). A new option replaces the current
  one only if it wins by **more than the 95% margin of error and by more than
  2 points**. Anything less is a **tie**, and a tie goes to the simpler option
  (when neither is simpler, the current one stays).
- In the comparison tables, **"only A" / "only B"** count the questions that
  one option found and the other did not.

---

## The question sets

**Retrieval pool.** OpenAI's `gpt-6-luna` wrote one question per eligible section
(1,775 sections: every section except those only about release, reception or
other artists' covers and, in articles about people, studios, equipment or
techniques, only those that name the Beatles or one of them), each with the exact sentence that answers it; a question
is kept only if that sentence exists word for word in its section. 1,771 were
kept. I then read a random sample of 100 by hand: **8 of 100 had a problem
(8%)** (other artists' records, ambiguous questions, answers going beyond
their quote). Problem questions stay in the pool: they add similar noise to every
option, which can hide small differences but is unlikely to flip a decision.

**Generation set.** 108 candidates written with the help of AI, reviewed by an
independent agent and then by me: 4 dropped, 18 fixed, **104 frozen** in
`eval/generation/set_v1.yaml`. The full record is in
`eval/generation/curation_v1.md`. It is used in the *Answers* section.

---

## Chunking: how to cut the articles

### The plan: four arms

Wikipedia articles come divided into titled sections (*Background*,
*Recording*, *Personnel*…), and each one is usually about a single subject.
That structure is worth keeping: a chunk cut inside one section never mixes
two subjects, its header can say which section it comes from, and each
citation in the app can link straight to that section of the article. The
question this stage starts with is whether respecting sections actually finds
more answers than cutting the whole article blindly, as in the course.

A grid of many sizes would mostly compare near-identical indexes: when each
section is cut on its own, a section that fits the size stays one chunk at any
size. How long the sections are decides how many each size cuts. The index holds 2,646 sections: what is left after normalisation removes the ones that are not about the music (references and notes, charts and certifications, track listings and personal life). 1,775 of them were eligible for a
retrieval question and 1,771 got one: for the other 4, the question generator
never quoted its answer word for word from the section, even after a retry,
so no question could be checked and they were left out.

This table shows how the sections are spread by length and, for each chunk size, which of them get cut:

| Section length | Sections | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- |
| up to 128 tokens | 24.9% | whole | whole | whole |
| 129–256 | 23.5% | whole | whole | whole |
| 257–512 | 29.2% | cut | whole | whole |
| 513–1024 | 19.8% | cut | cut | whole |
| over 1024 | 2.6% | cut | cut | cut |

So I chose four arms:

| Arm | What it tests |
| --- | --- |
| `blind_t512_o64` | Do sections help at all? Whole article cut blindly, as in the course |
| `section_t256_o32` | Cut sections a lot: splits about half of them |
| `section_t512_o64` | Cut sections a little: splits about 22%. **The baseline** |
| `section_t1024_o128` | Almost never cut a section: splits about 3% |

The three section arms share the same overlap ratio (12.5%), so they differ
in size only; `blind_t512` and `section_t512` differ only in how the text is
divided. A tie goes to the simpler arm, in the order 512 → 256 → 1024.

### Results (development set, k = 5)

| Arm | hit@5 | MRR@5 | doc_hit@5 | hit@20 | Context |
| --- | --- | --- | --- | --- | --- |
| `section_t512_o64` | **0.926** | 0.762 | 0.965 | 0.987 | 1,628 |
| `blind_t512_o64` | 0.894 | 0.735 | 0.939 | 0.981 | 2,296 |
| `section_t256_o32` | 0.923 | 0.766 | 0.971 | 0.982 | 989 |
| `section_t1024_o128` | 0.922 | 0.762 | 0.961 | 0.987 | 2,050 |

| Comparison (B against A = `section_t512_o64`) | B − A | Margin | Only A / only B | Verdict |
| --- | --- | --- | --- | --- |
| `blind_t512_o64` | −3.2 | ±1.5 | 68 / 28 | **512 wins** |
| `section_t256_o32` | −0.2 | ±1.1 | 26 / 23 | Tie |
| `section_t1024_o128` | −0.4 | ±0.7 | 12 / 7 | Tie |

**Conclusion.** Respecting sections helps: cutting blindly lost by 3.2
points. The three sizes tie, so 512 stays (the simpler choice by the rule).
But 256 reached the same score **with 40% less context** (989 tokens against
1,628). That is a cost argument, not a simplicity one, so I left it open for
the *How many chunks* stage, where sizes are compared with a similar number of tokens.

### Follow-up: is the overlap needed?

| Comparison | B − A | Margin | Only A / only B | Verdict |
| --- | --- | --- | --- | --- |
| `section_t512_o0` against `section_t512_o64` | −0.5 | ±0.5 | 9 / 3 | Tie |

**Conclusion.** Overlap changes almost nothing, because it only matters when
an answer falls exactly on a cut. Tie, so overlap 0 stays (no repeated text):
**`section_t512_o0`**.

*Not run, by the rules fixed in advance: the midpoints (384, 768), which only
run when two neighbouring sizes do not tie; and `blind_t256`, planned for the
case where 256 wins (256 won later, at the *How many chunks* stage; see there why it
was skipped).*

---

## Embedding model

The local, free `nomic-embed-text-v1.5` against OpenAI's
`text-embedding-3-small`, both in hybrid search (`alpha` 0.5, which decides)
and in dense search alone (`alpha` 1, to see the embedder without the keyword
half).

| Embedder | Search | hit@5 | MRR@5 | doc_hit@5 | hit@20 |
| --- | --- | --- | --- | --- | --- |
| nomic | hybrid | **0.921** | 0.758 | 0.964 | 0.986 |
| OpenAI | hybrid | 0.920 | 0.742 | 0.964 | 0.984 |
| nomic | dense only | 0.897 | 0.692 | 0.940 | 0.970 |
| OpenAI | dense only | 0.857 | 0.663 | 0.927 | 0.960 |

| Comparison | B − A | Margin | Only A / only B | Verdict |
| --- | --- | --- | --- | --- |
| OpenAI hybrid (B) against nomic hybrid (A) | −0.1 | ±1.2 | 30 / 29 | Tie |
| OpenAI dense (B) against nomic dense (A) | −4.0 | ±1.8 | 93 / 44 | nomic wins |
| nomic dense (B) against nomic hybrid (A) | −2.4 | ±1.5 | 58 / 28 | Hybrid wins |

**Conclusion.** In hybrid search the two embedders tie, so the free local one
stays: **nomic**. On its own nomic is even 4 points better. And hybrid search
beats dense alone by 2.4 points, which confirms the keyword half earns its
place.

---

## Contextual retrieval: planned, not run

I excluded it by reasoning, before spending on it:

- every chunk already carries a header with its song, album and section;
- 256-token chunks (more pieces of each section) tied with 1,024-token ones
  (almost whole sections), so a fragment losing its context costs nothing
  measurable;
- 98.6% of the answers are already among the top 20, so there is little left
  for better chunk text to find. What is left is ranking: the answer is
  usually there but sometimes below the cut, which is the job of a reranker
  (next section).

---

## Reranker

I tried two small local models: MiniLM (`Xenova/ms-marco-MiniLM-L-6-v2`)
and Jina turbo (`jinaai/jina-reranker-v1-turbo-en`). Both run locally, so
they add no cost for the user and no new key.

- **Quick trial:** MiniLM on 100 random development questions. No gain
  (hit@5 0.93 → 0.92, MRR 0.780 → 0.766), and each search went from about
  0.3 s to 2.4 s.
- **Best-case test:** a reranker can only fix questions whose answer is
  already in the top 20 but below the cut. To win it had to rescue at least
  25 of them (2 points of 1,240). Run only on those 81 questions, MiniLM
  rescued 50 and Jina turbo 47: enough to justify a full run. Jina was also
  slower and ranked worse (lower MRR), so only MiniLM got a full run.
  These two checks save no files by design, so they are the only numbers
  here that cannot be recomputed from the repository.
- **Full run with MiniLM:**

| | hit@5 | MRR@5 | Search time |
| --- | --- | --- | --- |
| No reranker | 0.921 | 0.758 | 327 ms* |
| MiniLM | 0.934 | 0.781 | 2,111 ms |

\* Measured during the reranker trial; that run's row in `experiments.csv` has no
search time.

| Comparison | B − A | Margin | Only A / only B | Verdict |
| --- | --- | --- | --- | --- |
| MiniLM (B) against none (A) | +1.3 | ±1.4 | 34 / 50 | Tie |

**Conclusion.** It rescued 50 questions but lost 34 that were already found:
+1.3, a tie, at six times the search time. **No reranker** (the code stays,
switched off).

*Note: the reranker was checked again on the final setup: 34 questions
could be rescued and 25 were needed; MiniLM rescued 19 and Jina turbo 20.
Still out.*

---

## Alpha: the weight of keywords against meaning

`alpha` 0 is sparse only (BM25 keywords), 1 is dense only (embeddings).
Since a different `alpha` could break the size tie of the chunking stage, all
three sizes ran with five values: 15 runs (256 with overlap 32, 512 with
overlap 0 and 1024 with overlap 128: the arms in use at the time).

### hit@5 (development set)

| Size | α 0 | α 0.25 | α 0.5 | α 0.75 | α 1 |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.794 | 0.854 | **0.923** | 0.921 | 0.887 |
| 512 | 0.816 | 0.867 | **0.921** | 0.919 | 0.897 |
| 1024 | 0.815 | 0.863 | 0.922 | **0.923** | 0.904 |

### Each value against 0.5, at the same size

| Size | α 0 | α 0.25 | α 0.75 | α 1 |
| --- | --- | --- | --- | --- |
| 256 | −13.0 ±2.0, loses | −6.9 ±1.5, loses | **−0.2 ±1.2, tie** | −3.6 ±1.6, loses |
| 512 | −10.5 ±1.9, loses | −5.4 ±1.5, loses | **−0.2 ±1.1, tie** | −2.4 ±1.5, loses |
| 1024 | −10.6 ±1.9, loses | −5.9 ±1.5, loses | **+0.1 ±1.2, tie** | **−1.8 ±1.5, tie** |

The best cells tie with each other too: 1024 at 0.75 against 512 at 0.5 is
+0.2 ±1.2, and 256 at 0.5 against 512 at 0.5 is +0.2 ±1.1.

**Conclusion.** BM25 alone loses by 10 to 13 points at every size; 0.5 and
0.75 tie everywhere (at 1024, dense only ties too), so **`alpha` 0.5** stays (the one already in use), and
the sizes still tie, so **512 stays for now**.

---

## How many chunks, and how many per section

### First: how many chunks, at 512

No new runs were needed: every run stores the rank of the right chunk for
each question, so a single run gives hit at any k.

| Chunks (k) | hit@k | Against the step before |
| --- | --- | --- |
| 5 | 0.921 | |
| **7** | **0.951** | +3.0, wins |
| 8 | 0.957 | +0.6, tie |
| 10 | 0.970 | +1.9 over 7, tie |

The rule was applied step by step, as fixed before this stage: start from
the fewest chunks and move up only while the next step wins. 7 beats 5,
but neither 8 nor 10 beats 7, so 7 stays: fewer chunks, less text for the
model. In the end it mattered little: the final setup is 256-token chunks × 8,
which ties with 512 × 7 while sending 31% less text.

*Note: hit@8 was added later, for the 256 vs 512 comparison.*

### Then: the cap per section, at 512 with 7 chunks

The cap stops overlapping pieces of one section from filling the list.

| Cap | hit@7 | Context |
| --- | --- | --- |
| 1 | 0.944 | 2,316 |
| **2** | **0.951** | 2,312 |
| 3 | 0.950 | 2,313 |
| None | 0.950 | 2,313 |

At 512 the cap barely matters: cap 1 against 2 is −0.7 ±0.7, cap 3 is
−0.1 ±0.3. Both ties.

No cap ties with cap 2 as well (−0.1). Cap 2 stays because it was already in
place as a safeguard from the design: it stops one long section from filling
the list and pushing out chunks from other sources. Here a tie keeps the
current option rather than the simplest one.

### Then: the size question left open in the chunking stage

After the chunking stage showed 256 needing 40% less context at k = 5, and
before this stage ran, I planned to compare 256 and 512 **with a similar
number of tokens** (so this comparison was not fixed before any result).
512-token chunks × 5 and 256-token chunks × 8 both give the model about 1,600
tokens:

| Option | hit | Context |
| --- | --- | --- |
| 512 × 5 | 0.921 | 1,622 |
| 512 × 7 | 0.951 | 2,312 |
| **256 × 8** (overlap 32 at this point) | **0.956** | 1,592 |

| Comparison | B − A | Margin | Only A / only B | Verdict |
| --- | --- | --- | --- | --- |
| 256 × 8 (B) against 512 × 5 (A) | +3.5 | ±1.3 | 15 / 59 | **256 × 8 wins** |
| 256 × 8 (B) against 512 × 7 (A) | +0.6 | ±1.0 | 18 / 25 | Tie, with 31% less text |

**Conclusion.** With a similar number of tokens, smaller chunks and more of them
find 3.5 points more answers. That was the comparison planned in the chunking
stage, so it decides; against 512 × 7 it ties while sending 31% less text.
**256-token chunks, 8 per question.**

*Note: the chunking stage planned `blind_t256` for the case where 256 won.
It was not run, for three reasons. Respecting sections lets each citation in
the chatbot link to the right section of the article. Blind cutting had
already lost to cutting by section at 512 (−3.2 points), and I saw no reason
to expect a different result at 256. Lastly, a blind chunk can span several
sections, so its header has no `Section`,
which could make the keyword search less precise.*

### Finally: overlap and cap again, at the new size

| Comparison (at 256 × 8) | B − A | Margin | Only A / only B | Verdict |
| --- | --- | --- | --- | --- |
| Overlap 0 (B) against overlap 32 (A) | −0.3 | ±0.7 | 13 / 9 | Tie → overlap 0 |

| Cap (256, overlap 0, 8 chunks) | hit@8 | Against cap 2 |
| --- | --- | --- |
| 1 | 0.913 | −4.0 ±1.3, **loses** |
| **2** | **0.953** | |
| 3 | 0.953 | +0.0 ±0.2, tie |
| None | 0.952 | −0.1 ±0.3, tie |

**Conclusion.** With smaller chunks a cap of 1 now hurts (−4.0): many answers
need two neighbouring pieces of the same section. **Cap 2** stays.

### A last check: does any earlier tie flip at the new size?

Every tie was crossed with the new size using runs that already existed.
For example, `alpha` 0.75 against 0.5 at 256 with 8 chunks (overlap 32,
before the last overlap check): 0.948 against 0.956, −0.8 ±0.9, still a tie.

Only OpenAI embeddings at 256 were not re-checked, because the result could
hardly change the decision: at 512 the two embedders tied in hybrid search
(−0.1) and OpenAI lost by 4 points on its own (dense only). Nothing in the
other stages suggested the chunk size favours one embedder over the other,
and even a small OpenAI gain would still be a tie, which goes to the free
local model.

---

## The test set

The final retrieval setup run on the 531 questions kept aside:

| Set | n | hit@8 | MRR@8 | doc_hit@8 | hit@20 | Context |
| --- | --- | --- | --- | --- | --- | --- |
| Development | 1,240 | 0.953 | 0.769 | 0.981 | 0.981 | 1,589 |
| **Test** | **531** | **0.959 ±1.7** | **0.775** | **0.985** | **0.987** | **1,594** |

**Conclusion.** The test score is ≈0.6 points *above* development, well inside
the margin: no sign of overfitting to the development set.

---

## Answers: the generation set

Retrieval tells whether the right chunk arrived; this stage tells what the
model did with it. Each of the 104 questions got an answer from the whole app,
graded by hand: **correct** and **faithful**. Questions that must be declined
pass only with the exact decline sentence.

### Prompt, version 1 → 2

After the first full run the prompt was changed once (the decline list now
overrides the "answer the part you can" rule, and lyrics are declined with the
same sentence). The effect was small: it declined 24 of the 31 questions it
should decline instead of 23 (it fixed three and broke two), and it still
declined only one answerable question it should have answered. I stopped
tuning there, to avoid fitting the prompt to these 104 questions. That first
run is not in the repository, so its numbers cannot be checked and only the
final run is reported in full.

### Final run

| | Result |
| --- | --- |
| **Pass all checks** | **88 / 104 (85% ±7)** |
| Correct (answerable) | 71 / 73 |
| Faithful (answerable) | 66 / 73 (90% ±7) |
| Declined when it should | 24 / 31 (77% ±15) |
| Answer chunk among the 8 sent | 64 / 73 |

*8 of the 9 answers without their chunk were still correct and faithful,
from another chunk that had the same fact.*

| Kind of question | Pass all checks |
| --- | --- |
| Credits (who played, sang, engineered) | 15 / 15 |
| Adversarial (instructions hidden in the question; 1 answerable, 2 to decline) | 3 / 3 |
| Unanswerable (counts, "the most...", lists) | 17 / 19 |
| Details the model is unlikely to know | 33 / 38 |
| Famous facts | 12 / 14 |
| Follow-ups ("and who played on it?") | 3 / 5 |
| Out of scope (lyrics, solo careers, opinions) | 5 / 10 |

### Does retrieval help? The model with no sources

The same 73 answerable questions, asked to the model alone with no search.

| | Correct |
| --- | --- |
| Model alone | 62 / 73 |
| The app | 71 / 73 |

The app fixed 10 of the 11 the model got wrong on its own and broke 1 it
got right. **+12.3 points ±8.5**, a clear win by the paired rule.

### One search or two?

A second search can only help an answer whose first search missed the right
chunk. Only **1** failed answer was like that.

The table below is not a run: it shows the best case for two searches,
worked out by hand. Suppose a second search rescues some of the failed
answers and breaks none. Over the 104 questions, each rescued answer adds 1/104 ≈ 1 point, and
the margin comes from the same paired formula as everywhere else (each
question scores +1 if rescued, 0 otherwise). The margin grows with the gain
because every rescued answer is one more question where the two runs
disagree. "Wins?" applies the tie rule: the gain must beat both the margin
and 2 points.

| Rescued answers | Gain | Margin | Wins? |
| --- | --- | --- | --- |
| 1 | 1.0 | ±1.9 | No |
| 2 | 1.9 | ±2.7 | No |
| 3 | 2.9 | ±3.2 | No |
| 4 | 3.8 | ±3.7 | Yes, just |

**Conclusion.** Even the best case needs 4 rescued answers to win, and only
1 failed answer had missed its chunk, so a second search could not win.
**One search stays.**

### What the 16 failures were

The **pass rate** says how often the app fails; this section says why, which
is what points to the next improvements. Only one failure came from the search
missing its chunk: the rest are the model's (not declining, or adding a
detail from memory), and three are partly the test's own (refusals in other
words, a badly written question).

- **7 did not decline when they should.** Two declined to quote lyrics, but in
  their own words rather than with the exact sentence; one question turned out
  to have an answer in the corpus after all (my mistake writing it; it stays
  in the frozen set and still counts as a failure); four gave a true,
  cited partial answer instead of declining.
- **7 were correct but added a detail from memory** that the cited chunk
  does not contain.
- **2 were wrong:** one because the search missed the chunk, one because the
  answer left out half of what was asked.

---

## The final setup

| Setting | Value |
| --- | --- |
| Chunking | 256 tokens, respecting sections, no overlap |
| Embedding | `nomic-embed-text-v1.5` |
| Search | Hybrid, `alpha` 0.5 |
| Reranker | None |
| Chunks sent (`k`) | 8 |
| Chunks per section (cap) | 2 |
| Searches per question | 1 |
| **Retrieval, test set** | **hit@8 0.959 ±1.7** |
| **Answers, generation set** | **85% pass all checks** |
