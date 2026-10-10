# Beatles Song Detective — Engineering Brief

> **If you are reviewing the project:** this is the brief the AI assistant
> worked from, so it is written as instructions to whoever writes the code.
> The README has the summary and the results. The most useful parts here are
> **4. Architecture** (the layers and the three rules that hold them
> together), **5. Data flow** (the corpus, chunking and the evaluation
> design) and **6. Known traps** (bugs that produced no error and were found
> only by running code).
>
> Contents: 1. What this is · 2. Scope · 3. Operating constraints ·
> 4. Architecture · 5. Data flow · 6. Known traps · 7. Code conventions ·
> 8. What not to do

Read this before writing any code. It carries the design, the reasons behind
it, and the traps that were found by running code rather than by reasoning
about it. Do not re-open a settled decision unless new evidence contradicts it;
if you believe one is wrong, say so and explain why before changing anything.

---

## 1. What this is

A RAG application about **the Beatles' songs**: their particular
details, the techniques behind their sound, their curiosities, and the stories
of how they were written and recorded — sessions, equipment, engineers, and who
was in the room. It answers in the language of the question, from English
Wikipedia, and **every factual sentence carries a citation**.

**The audience is curious fans, not engineers.** People who want to know how a
sound was made or where a song came from, without studio jargon. The song is the
centre of the product: albums, people, studios, equipment and techniques matter
mostly as they are asked about *through* a song ("how did they get the vocal
sound on *Tomorrow Never Knows*?").

The chatbot is not really the deliverable. The deliverable is the **evaluation
discipline**: every design choice — chunking, embeddings, search settings,
orchestration — is decided by measurement rather than intuition, and the
results are published alongside the code.

Readability is a requirement, not a preference. This codebase is meant to be read
and learned from, so a clear thirty-line function beats a clever five-line one.

---

## 2. Scope

### In scope

| Query type | Example |
| --- | --- |
| Recording technique | How did they get the vocal sound on *Tomorrow Never Knows*? |
| Credits / who played what | Who played the flugelhorn on *Penny Lane*? |
| Origin and composition | What inspired *Strawberry Fields Forever*? |
| Authorship and disputes | Who really wrote *In My Life*? |
| Equipment and studios | What was special about the console at Abbey Road? |
| Sessions and chronology | What happened during the *Get Back* sessions? |
| Album context | Why did *Revolver* sound so different? |
| Studio people | What did Geoff Emerick change when he took over? |
| **No answer** | → "The sources here don't say." |

### Out of scope: counts, lists, extremes

A vector search answers "what happened" questions, not "how many" or "which is
the most" ones: counts, complete lists, date ranges, extremes, and
cross-referencing by technique or instrument. Answering those needs a SQL-like
catalogue and a router, which this project does not build.

**These questions must be declined, not attempted**, and the distinction is
subtler than it looks. "Which technique did Emerick use most?" is thematically
in scope, but it is an *extreme*: no single passage states the answer, and
retrieval hands the model eight passages out of thousands. An answer asserting
"most" from those eight is unsupported by its own context, which under this
project's primary metric is a **failure even when the answer happens to be
right**. The passage-level rewording ("what did Emerick change when he took
over?") is the one that works.

The generation set therefore carries a deliberate population of superlative and
counting questions whose correct behaviour is "the sources here don't say".

### Explicitly out of scope

Song lyrics (copyright — never quote more than a few words), solo careers, chart
rankings and subjective opinions, deep harmonic analysis, modern DAW advice, and
microphone placement or exact equipment settings, which the corpus does not
contain.

**There is exactly one answering mode**, and nothing is asserted without a
citation: no ungrounded extras such as modern studio equivalents.

---

## 3. Operating constraints

- The application runs as a **public Hugging Face Space**, using the CPU only.
  Free Spaces now get ZeroGPU hardware only, which will not start without one
  `@spaces.GPU` function: app.py declares an empty one that is never called.
- **No API keys in the repository.** The user pastes their own OpenAI key into a
  single UI field, held per session and never logged.
- **Nothing expensive runs on the user's key.** Corpus embedding and
  evaluation-set generation run offline, on the developer's key. At request time
  the user's key pays only for the answer.
- Trying the app end to end must cost the user well under half a dollar.

---

## 4. Architecture

Six layers. **Dependencies point downward only.** Nothing in a lower layer knows
that a higher one exists.

```
6. INTERFACES     app.py (Gradio)        src/eval/      ← never import each other
5. AGENT          agent.py  tools.py  llm.py
4. RETRIEVAL      search.py                             ★ THE BOUNDARY
3. INDEX          chunking  nodes  embeddings  build
2. INGEST         seeds  fetch  normalize  audit
1. MODELS         schemas.py (Pydantic)
0. CONFIG         config.py  (config.yaml)
```

`publish.py` lives in `src/index/` but is a command-line step run after the
evaluation, not part of the layer: it checks the copy by searching both
stores, so it is the one index module that imports `search.py`, and nothing
imports it (a test makes sure the app never loads it).

### Three rules that hold the design together

1. **`search.py` is the only boundary that matters.** Its signature never
   changes: `search(query, filters, top_k) -> list[Chunk]`. The agent, the app
   and the evaluation all call it. Whatever is inside — LlamaIndex, Qdrant,
   hybrid fusion — is invisible to them. This is what makes it possible to
   change the embedding model without touching anything else.

2. **`llm.py` is the only module that imports `openai`.** It translates between
   the project's own `Msg` / `ToolCall` dataclasses and the SDK. Conversation
   history lives in the project's format, never in raw SDK objects.

3. **Configuration is loaded at entry points and passed down explicitly.**
   `app.py`, `build.py` and `run_eval.py` load `config.yaml`; every function
   below receives the values it needs as parameters. No module reads globals from
   deep inside. This is what makes an experiment "change a YAML" instead of
   "hunt for constants". `config.py` validates cross-references on load: a
   topic, doc type or embedding named in the settings must exist, and the
   retrieval candidate pools must reach the largest `k`. `build.py` and
   `run_eval.py` also take `--strategy` and `--embedding` (and run_eval
   `--alpha`, `--reranker`, `--final-top-k` and `--max-per-section`): they
   override the file for one run and go through the same validation, while
   the file keeps the values the app uses.

### Repository layout

```
beatles-rag/
├─ app.py                    # Space entry point (Gradio)
├─ config.yaml               # EVERY constant lives here
├─ requirements.txt          # Space: minimal
├─ requirements-dev.txt      # + corpus collection, experiments, tests
├─ README.md                 # Space front-matter + documentation
├─ results.md                # every stage: the tables, the rule, the decision
├─ LICENSE                   # MIT for the code; Wikipedia text is CC BY-SA 4.0
├─ src/
│  ├─ config.py              # loads config.yaml into a typed object
│  ├─ schemas.py             # Pydantic: Document, Section, Chunk, EvalQuestion
│  ├─ ingest/
│  │  ├─ seeds.yaml          # the corpus: every article, its doc_type, its rules
│  │  ├─ seeds.py            # reads seeds.yaml for fetch and normalize
│  │  ├─ fetch.py            # Wikipedia API → data/raw/ (only what is missing)
│  │  ├─ normalize.py        # raw → data/docs/
│  │  └─ audit.py            # the report to read once after normalizing
│  ├─ index/
│  │  ├─ chunking.py         # article/section Documents + SentenceSplitter
│  │  ├─ embeddings.py       # the nomic wrapper and the OpenAI variant
│  │  ├─ nodes.py            # Chunk <-> LlamaIndex node, both directions
│  │  ├─ collections.py      # collection names and how one is opened (no ingest imports)
│  │  ├─ build.py            # one Qdrant collection per strategy
│  │  └─ publish.py          # copies the winning collection to qdrant_store (command-line step)
│  ├─ rag/
│  │  ├─ search.py           # THE stable interface
│  │  ├─ rerank.py           # optional cross-encoder, local (fastembed)
│  │  ├─ tools.py            # the search tool, numbered sources, citation links
│  │  ├─ agent.py            # the loop
│  │  ├─ llm.py              # the only module importing openai
│  │  └─ prompt.md
│  ├─ app/
│  │  ├─ startup.py          # finds the store: HF dataset repo or data/qdrant_store
│  │  ├─ chat.py             # one chat turn: key, events, friendly errors
│  │  ├─ render.py           # answer as escaped text + links built by code
│  │  ├─ ui.py               # the Gradio layout; every class beatle_rag_*
│  │  ├─ style.css           # the page's look, editable
│  │  ├─ beatlessongdetective.png  # the title image
│  │  ├─ fonts/              # Inter (OFL), embedded in the page
│  │  └─ deploy.py           # uploads the code to the Space (python -m src.app.deploy)
│  └─ eval/
│     ├─ generate.py         # writes the pool (one question per eligible section)
│     ├─ split.py            # dev/test, stratified by doc_type
│     ├─ audit_sample.py     # random sample for the hand audit, and its report
│     ├─ question_file.py    # question files: the pool (JSONL), the set (YAML)
│     ├─ metrics.py          # hit@k, MRR, paired margins, the tie rule
│     ├─ run_eval.py         # scores one configuration on the pool
│     ├─ compare.py          # two runs, question by question, with the tie rule (--k, --k-b)
│     ├─ recoverable.py      # best-case check before a full reranker run
│     ├─ run_generation.py   # the agent on the generation set -> answers to grade
│     ├─ generation_checks.py  # what code checks in an answer; pass/fail
│     ├─ generation_report.py  # a graded run, compared by the tie rule
│     └─ closed_book_prompt.md
├─ eval/
│  ├─ generation/
│  │  ├─ set_v1.yaml         # the frozen set run_generation.py reads
│  │  ├─ curation_v1.md      # how the 108 candidates became these 104
│  │  └─ runs/               # answers.yaml of the final and closed-book runs, graded by hand
│  ├─ pool/
│  │  ├─ eval_v1.jsonl       # the retrieval pool
│  │  └─ audit_eval_v1.yaml  # the audited sample, with verdicts
│  ├─ runs/                  # per_question.jsonl of every run
│  └─ experiments.csv        # one row per run, with the settings that define it
├─ tests/                    # pytest: pure functions, plus checks on built data
└─ data/                     # gitignored — published as a HF Dataset repo
   ├─ raw/  docs/  chunks/
   ├─ qdrant_experiments/    # all trial collections
   └─ qdrant_store/          # only the winner — this is what ships
```

---

## 5. Data flow

```
seeds.yaml → fetch → raw/*.json → normalize → docs/*.json → audit (read once)
                                                  ↓
                               chunking → nodes → embeddings → qdrant_store/
                                                                    ↓
question → agent → tools → search() ───────────────────────────────┘
                              ↓
                        chunks → llm → answer with citations
```

Three data layers, each with one job:

| Layer | Path | Contains |
| --- | --- | --- |
| Raw | `raw/wikipedia/{pageid}.json` | API response, untouched. Downloaded once, never again |
| Document | `docs/{doc_id}.json` | One article, normalised, **containing no chunks** |
| Chunks | `chunks/{corpus_version}/{strategy_id}.jsonl` | One per line, per strategy |

The document layer holds **no chunking decisions**. That is precisely why any
strategy can be rebuilt without re-downloading anything, and why one frozen
evaluation set can compare all of them.

### The corpus (`corpus_v1`)

**`seeds.yaml` is the corpus.** Every article is listed there by its canonical
Wikipedia title, grouped by the `doc_type` it gets; nothing is resolved from
categories, so a later fetch can never add a page nobody chose. `fetch.py`
downloads only the seeds that have no raw file yet, so every revision already
downloaded stays frozen. `normalize.py` looks each raw file up in `seeds.yaml`
by title, which is why doc_types and per-article rules live there and nowhere
else, and why a doc_type can change without downloading anything again.

326 English Wikipedia articles:

| `doc_type` | What | Rule |
| --- | --- | --- |
| `song` | The songs the Beatles recorded: their own, and the covers they released | For a cover, only the sections about their version |
| `album` | The studio albums, *Let It Be... Naked*, the three *Anthology* sets | |
| `person` | Early members and stand-ins, producers, engineers, session players | The four Beatles: only their musical sections |
| `studio` | Where they recorded | |
| `technique` | A process or effect (double tracking, varispeed, distortion) | |
| `equipment` | A device (instruments, amplifiers, the EMI consoles, the DI unit) | |
| `overview` | Articles spanning the catalogue, and key events (Hamburg, the Decca audition, India, the rooftop concert, the *Let It Be* film) | |

Per-article rules in `seeds.yaml`, used by only a few articles:

- **`keep_sections`**: keep only the sections whose path contains one of the
  listed strings. Used by the covers (`[Introduction, Beatles]`), the four
  Beatles (their musical sections: personal material sits inside their
  chronological sections, so dropping headings would not catch it) and "The
  Beatles' recording sessions" (the narrated sections on the early recordings;
  the rest is a session log). Each string must match at least one section.
- **Songs are listed under their album**, which is their `Album` header and
  filter value: the first of the corpus's album articles on which the Beatles
  released the song (original albums as first issued, *Magical Mystery Tour*
  as the 11-track LP; an *Anthology* set only for songs first released there
  in 1995–96), or `Single/Outtake`. `load_seeds()` refuses an album group
  with no album article, so every value has one.
- **`name`**: the name shown in the header when the article title differs from
  the Beatles' title ("Maggie May (folk song)" → "Maggie Mae").

### Identifiers every document carries

- **`pageid`** — stable Wikipedia page id; the source of `doc_id`.
- **`revid`** and **`revised_at`** — the exact revision downloaded and its
  date. They freeze the corpus and back the provenance of every citation.
  `revid` must never be null.

### Normalisation rules

- **Templates that carry text are replaced by their text**, not deleted:
  sharps and flats (`{{music|sharp}}` → ♯), time signatures
  (`{{music|time|5|4}}` → 5/4), quotation boxes, and wrappers such as
  `{{nowrap}}`. Every other template (citations, layout) is deleted. Deleting
  them all would lose exactly what this project is about: "chorus (B, F, E)"
  instead of F♯, and "McCartney said:" followed by nothing.
- **Footnotes (`<ref>` tags), category links and images are removed.**
  `strip_code()` keeps a footnote's text, which would land in the middle of a
  sentence ("(1923 – 2008)Thedeadrockstarsclub.com – accessed…").
- **The gaps deleted templates leave are closed**: empty parentheses and runs
  of spaces. What remains is counted by the audit.
- **Headings are reduced to plain text**: links, HTML anchors and italic/bold
  markup removed (`''Love'' remix` → `Love remix`). The cleaned title is what is
  stored, given topics, matched by `keep_sections` and shown.
- **The text before the first heading is the section "Introduction".**
- **Two drop lists, each heading dropped with all its subsections.**
  `wikipedia.private_sections` (personal life, family, deaths, illness...) is
  always dropped: personal data the app has no reason to surface.
  `wikipedia.drop_sections` (references and the like, charts, certifications,
  track listings, other artists) is dropped unless the heading also names an
  in-scope topic: "Cultural references" stays, "Charts and certifications"
  goes.
- **Every heading rule matches whole words, ignoring case and apostrophe
  style** (`names_any`): "notes" is in "Notes and references" but not in
  "Denotes", "musician" is not in "Musicianship". The drop lists, the topic
  keywords and `keep_sections` all use it; exact titles are compared the same
  way (`comparable`), as whole headings.
- **"(s)" or "(es)" at the end of a word makes that ending optional**:
  `origin(s)` is "origin" or "origins", so each concept is listed once.
  Irregular plurals are written in full, and `influence` (reception) and
  `influences` (writing) stay separate on purpose. Any other parenthesis in a
  drop list or keyword is refused when the config loads, since it would
  silently match nothing.
- **A section with no text is dropped** (a heading followed only by
  subsections); its title survives in its subsections' `path`. Short sections
  stay as they are; the audit reports how many there are.
- **When a section is nested deeper than level 2, its `path` carries the full
  chain** (`Recording > Take 7`). It disambiguates generic headings in a chunk
  header, and it is what topics and `keep_sections` read.
- **A section gets a list of `topics`** (`config.yaml`, `topics`): every
  topic whose keywords its heading names, so "Recording and release" is both
  `recording` and `release`. An exact title ("Music", "Process") gives its one
  topic. There is no order among topics.
- **A heading with no signal of its own inherits from its nearest ancestor
  that has one**: "Take 7" under "Recording" is `recording`. The text before
  the first heading is `introduction`; a heading with no signal anywhere in its
  path is `general`.
- **The topics** follow what fans ask about: `writing` (origin, inspiration,
  lyrics, composition), `recording` (sessions, production, overdubs, demos),
  `personnel`, `musicianship` (playing style, instruments, gear),
  `live_performance`, `media` (films, videos, artwork, cultural references),
  `biography`; and three `out_of_scope` ones the evaluation pool leaves out:
  `release`, `reception`, `cover_versions`.
- **Topics label; nothing filters on them yet.** They decide which sections get
  an evaluation question and are reported by the audit. Chunks carry them as a
  list payload, so a future filter ("only recording") matches any element; the
  agent does not filter on them.

### Chunking grid

- **One splitter for every arm: LlamaIndex's stock `SentenceSplitter`.** Arms
  differ in exactly two things: `unit` (what one LlamaIndex `Document` is) and
  size/overlap. The splitter never cuts across two Documents, so "respecting
  sections" is not a different chunker; it is a different Document.
  `unit: article` (one Document per article) is the naive baseline taught in
  the course. `unit: section` (one Document per section) makes it impossible
  for a chunk to mix sections.
- **The grid has four arms**, each answering one question the eval can
  resolve: `blind_t512_o64` (do sections help at all?), `section_t256_o32`
  (splits about half the sections), `section_t512_o64` (a whole section for
  most; the default), `section_t1024_o128` (almost always one chunk per
  section). Section arms share a 12.5% overlap ratio, so they differ in size
  only. `blind_t512_o64` and `section_t512_o64` differ only in `unit`. The
  share of sections each size splits is printed by the audit.
- **Tie rule, fixed before seeing results:** an arm wins only if its gap beats
  **both** the paired 95% margin of error **and**
  `eval.min_meaningful_gap_points` (2 points). Anything less is a tie, and a tie
  goes to the simpler arm, in this order: 512 → 256 → 1024.
- **Follow-up arms, only if the result calls for them:** when two neighbouring
  sizes do not tie, their midpoint (768 or 384); `blind_t256_o32` only if 256
  wins (256 won at `top_k`; skipped since respecting sections is better in several ways: see results.md); the winning size with overlap 0 (kept if it ties).
- **Arms are compared at equal top-k**, which is what the app does: the LLM
  always receives `final_top_k` chunks, and at this model's price the extra
  tokens of bigger chunks cost well under a tenth of a cent per question. Every
  results row reports the average context tokens next to hit@k, so the
  trade-off is visible. The final size was decided later, comparing 256 and
  512 at a similar number of tokens (the `top_k` stage), a comparison added
  after the chunking results, before that stage ran.
- **The header is the Document's metadata.** Each Document carries the
  `header_fields` of its `doc_type` (e.g. `Song`, `Album`, `Section`) as
  metadata, and LlamaIndex prepends it to the text it embeds
  (`MetadataMode.EMBED`). `doc_id` and `char_start` are in metadata too, but
  excluded from the embedded and LLM text. `Section` is the section's `path`;
  a song's `Album` is `Document.album`, the group it is listed under in
  `seeds.yaml` (the schema refuses a song without one). The article-unit arm has no `Section` field: one blind chunk
  can span several sections. Its payload records the section where the chunk
  starts, and `max_per_section` uses that section.
- **Two side effects of the stock splitter, accepted on purpose.**
  (a) LlamaIndex subtracts the metadata length from `chunk_size`, so a "512"
  chunk carries ~480 tokens of text. (b) Splitting is greedy, so a long
  section can end in a small tail chunk (600 → 512 + 88). Both are small, and
  both affect every section arm equally.
- **The one piece of custom code that cannot go:** in a section-unit arm,
  `node.start_char_idx` is relative to the section, so a transformation adds
  the section's `char_start` (trap 2). The evaluation compares document-level
  offsets.
- **A retrieval hit is measured by coverage, not containment:** a chunk hits
  when it covers at least half (`metrics.HIT_MIN_COVERAGE`) of the evidence
  span, by character offsets. Under containment, a quote longer than the
  overlap and cut by a boundary would count as a miss in every chunk,
  penalising the small arm for a metric artefact.

### Evaluation: two separate sets, two separate questions

**1. The retrieval pool** answers "did the right passage reach the LLM?". It
decides chunking, the embedding model, the reranker, `alpha`,
`final_top_k` and `max_per_section`.

- **One question per eligible section** (`generate.py`), so every eligible
  section is covered by construction. Two rules decide eligibility: the section
  is not `release`, `reception` or `cover_versions` (chart and opinion material
  the project does not answer; it stays in the index), and in an article whose
  subject is not the Beatles (a person, studio, piece of equipment or
  technique) the section names the band or one of them. Checked automatically
  by `validate_candidate()`.
- **Metrics, all on what `search()` returns** (`run_eval.py`, one search per
  question with top_k = the largest k): **hit@k** for every `k_values` entry
  (= recall@k, since each question has exactly one evidence quote) and
  **MRR@k**, one row per k as in the course; and **doc_hit@k**, whether any chunk comes from the
  right article, which separates "wrong song" failures (fixed by `alpha` or
  the header) from "right song, wrong passage" ones (fixed by the chunk
  size). **hit@`final_top_k` decides**; hit@20 is not a value the
  app would use, but a large gap between it and hit@`final_top_k` is the signal that
  justifies testing a reranker. Every run also reports the average context tokens.
- **Every run is saved**: its per-question ranks in
  `eval/runs/<run>/per_question.jsonl` (what `compare.py` reads) and one row in
  `eval/experiments.csv` with the settings that define it (stage, strategy,
  embedding, `alpha`, `final_top_k`, `max_per_section`, split). The rows of one
  stage are that stage's comparison table.
- **Every question, every run, always the same questions.** Retrieval is local
  and free, so there is nothing to save by sampling, and random subsets would
  add noise between configurations.
- **Split into dev and test** with `split_seed` and `dev_ratio`, stratified by
  `doc_type`. Decisions use dev only; test is run once, at the end.
- **Reported overall, per `doc_type` and per topic.** The pool follows the
  corpus, with no quota per type or topic: songs, and writing, recording and
  personnel sections, are most of it, because they are most of what users ask
  about. **Decisions use the overall number.** The other rows are an alarm: a
  configuration that wins overall but sinks a type or topic with enough
  questions is looked at before it is accepted. A question's topics come from
  its `section_id` at scoring time, and a section with two topics counts in
  both rows. Rows with few dev questions (musicianship, biography,
  live_performance: roughly 20–40) are shown as indicative only: too noisy to
  tune on.
- **A single share shows "±" only with at least 5 of each outcome**
  (`metrics.share_text`): "66/73 (90% ±7)", but "71/73 (97%)", never a
  false "±0".
- **Every margin of error is 95%** (1.96 standard errors, `metrics.Z_95`),
  and a comparison between two configurations uses the paired margin: both
  answer the same questions, so only the questions where they disagree carry
  noise.
- **Known limitation:** when the same fact appears in two articles (a song's
  and its album's), retrieving it from the other one counts as a miss. It adds
  similar noise to every configuration: it can hide small differences, but is
  unlikely to flip a decision.

**2. The generation set** answers "what did the LLM do with the passage?", which
the pool cannot see: whether the answer is right, whether each citation points
at the right chunk, whether it declines with "the sources here don't say" when
it should, and whether it leans on what the model already knows about the
Beatles instead of the sources.

- **About 100 questions, chosen and reviewed by hand** (±7 points at 100; only
  answers that differ between two runs are graded twice): some unanswerable, some
  credits, and some chosen on purpose for details the model is unlikely to
  know from memory (which engineer, which amplifier, who played an overdub),
  since the subject is famous and most general questions it answers unaided.
  Five are follow-ups: a first question builds the chat history and only the
  second ("and who played the brass on it?") is graded, which tests that the
  search query is rewritten to stand on its own. Three are adversarial:
  instructions hidden in the question, which the agent must not obey.
- **Run at the end**, on the winning configuration, and for the orchestration
  decision (`max_steps`: 1 against 2, settled by a pre-check without a
  2-step run).
- **Graded by hand, not by an LLM judge**, in each run's `answers.yaml`.
  Answerable questions get two grades: `correct` (says what the expected
  answer says, contradicts nothing) and `faithful` (every sentence is
  supported by the passage it cites, which covers both "citations point at
  the right chunk" and "relies only on the sources"). What code can check, it
  checks (`generation_checks.py`): a question that must decline passes only
  on the exact decline phrase, with no hand grade; cited numbers must exist;
  no links; and whether the evidence passage was retrieved and cited.
  **One pass/fail per question** (answerable: correct, faithful, citations
  exist, no link) is what two runs are compared on, by the usual paired
  rule. A judge would have to be calibrated against hand grades anyway, and
  it knows the Beatles too, so it would tend to pass the very failure this
  set exists to catch: a true claim the chunks do not support.
- **`closed_book_correct` is also graded by hand**, in the closed-book run's
  own `answers.yaml` (not in the set): the model answers each question with
  no retrieval, and the answer is compared with the question's own `answer`
  and `evidence`. True if it has the key fact and contradicts nothing; false
  if it is wrong, misses the key fact or says it does not know; not asked for
  questions the sources cannot answer. A case that is not clear-cut gets its
  reason in `note`, so later grading stays consistent. The report gives the share it got right unaided and the
  pass rate on the questions it got wrong; the gain published in the README is
  the paired comparison of `correct` against `closed_book_correct` over all
  answerable questions.

### Experiment order

Index-time decisions first, query-time decisions last, one stage at a time,
each on the dev split, each decided by the tie rule. The stages, in the
order they ran (names as in the `stage` column of `eval/experiments.csv`;
`max_steps` was decided on the generation set, so it has no row there):

| Stage | Decides | Result |
| --- | --- | --- |
| `chunking` | Chunking strategy | `section_t512_o0`; replaced by `section_t256_o0` at `top_k` |
| `embedding` | Embedding model | nomic (`text-embedding-3-small` tied) |
| `reranker` | Cross-encoder reranker | none (code kept, off) |
| `alpha` | Dense/sparse weight | 0.5 |
| `top_k` | `final_top_k` / `max_per_section` | 8 / 2 |
| `max_steps` | Searches per question (generation set) | 1 |

- The held-out test split is run once, on the final configuration. Do not
  inspect its individual failures: read the number and nothing else; looking
  at them turns the test split into a second dev split.

### Agent and app

- **One search per question** (`max_steps: 1`, tool call
  forced), with the last 3 chat turns in view (`history_turns`: questions and
  answers only, without their [n] or search results) so a follow-up ("and who
  engineered it?") gets a complete search query. One prompt
  (`src/rag/prompt.md`) serves every `max_steps`: the loop decides when
  searching stops.
- **The LLM always writes the search query in English**, whatever the
  question's language: the corpus and both retrievers are English. **The
  answer is in the question's language**, and the decline phrase with it.
- **The answer is plain text with numbered citations** (`[1]`, `[2]`), one
  number per source chunk, not a JSON object: text can be streamed as it is
  written, and the app turns each number into a link to the exact Wikipedia
  revision. Structured Outputs stay where code reads the reply (question
  generation).
- **No filters in the search tool.** A wrong filter hides the answer, and
  nothing in the evaluation measures whether the LLM picks filters well.
  `search()` still accepts `filters` (doc_type, topics), used only by its
  command-line check; no filter was convincing enough to put in the app.
- **The answer prompt forbids quoting more than a few words of a lyric**, and
  the static part of the prompt (instructions and tool schemas) must reach the
  1,024 tokens OpenAI needs to cache it (trap 11).
- **Prompt attacks are handled mostly by design, not by the prompt.** The
  user pays with their own key, the prompt is public, the only tool reads,
  and history is per session. What remains: the prompt treats passages as
  data and keeps its role whatever a message asks; links are built by code
  from the sources, never written by the model; the app shows answers as
  text, not HTML. No input classifier: another call on the user's key for
  little protection.
- **The Space loads the embedding models at startup**, into a fixed cache
  folder, so the first visitor does not wait for a download.
- **The app (`src/app/`)**: the user pastes their key, used for one question
  and never stored or logged. Answers are escaped text; the only HTML is
  written by `render.py` (citation links, sources, time and cost). API errors
  arrive as llm.py's own exceptions and become chat messages.

---

## 6. Known traps

Each of these was found by running code. They produce no error message; they
quietly invalidate results.

1. **Qdrant point ids must be UUIDs or unsigned integers.** A text-based
   `id_func` returning `"wp_1__s2__c0"` raises
   `ValueError: Point id ... is not a valid UUID`. Use a deterministic
   `uuid.uuid5(NAMESPACE, chunk_id)` and keep the readable `chunk_id` in
   metadata, excluded from the embedded text.

2. **`node.start_char_idx` is relative to the section Document**, not to the
   article. A transformation step must add the section's `char_start` to obtain
   document-level offsets, which is what the evaluation compares against.

3. **The embedding model requires task prefixes and fastembed does not add
   them.** Its `query_embed()` and `passage_embed()` call `embed()` unchanged.
   Wrap the embedding class so the prefixes are applied in exactly one place.
   Sanity check: `get_query_embedding(x)` must differ from
   `get_text_embedding(x)`.

4. **Prefixes apply to the dense vector only.** The sparse encoder receives text
   through `node.get_content(MetadataMode.EMBED)` inside `QdrantVectorStore`,
   bypassing the embedding class. That is correct — BM25 must not index
   `search_document:`.

5. **Always pass `encoding="utf-8"` to every `open()`.** The corpus is full of
   en-dashes (`Lennon–McCartney`), typographic quotes, ♯ and ♭. On
   Windows this is the single most likely bug in the project, and it fails
   quietly.

6. **`FilterOperator.ANY` is not supported by Qdrant** — it raises
   `NotImplementedError`. Use `FilterOperator.IN`, which maps to `MatchAny` and
   works on list-valued payload fields.

7. **Payload indexes have no effect in local Qdrant.** Filtering still works, as
   a linear scan, which is fine at this corpus size.

8. **Local Qdrant locks its storage folder.** Pass only the synchronous client.
   On the Space, copy the downloaded store to a writable path before opening it,
   and never publish the `.lock` file.

9. **tiktoken downloads its BPE vocabulary on first use.** Set
   `TIKTOKEN_CACHE_DIR` so it is cached.

10. **Overlapping chunks can fill the top-k with the same section**, wasting the
    LLM's context and starving multi-source questions. Cap chunks per section
    on the final ranked list, after fusion.

11. **Prompt caching breaks when the static prefix changes.** Tool definitions
    and output schemas are part of that prefix, and it must reach 1,024 tokens
    to be cached at all. **Conversation history goes after the static
    instructions**, together with the retrieved chunks — placed before them, it
    changes the prefix on every turn and nothing is ever cached. Check
    `cached_tokens > 0` on the first real run.

12. **`revid` comes back null unless requested explicitly:** `prop=revisions`
    with `rvprop=ids|content|timestamp` (without `ids` there is no `revid`) and
    `rvslots=main`. `fetch.py` refuses a page without one.

13. **fastembed truncates input to the tokenizer's configured maximum,
    silently.** The limit comes from each model's tokenizer config, which can
    sit below the model's real capacity. If the embedder truncated below the
    largest chunk, the 1,024 arm would be embedded from its first half and the
    evaluation would blame the chunk size. **Measured:** nomic accepts over
    8,000 tokens — every arm is embedded whole.

14. **A plain title can fetch the wrong page, with no error.** "Norman Smith",
    "Tambura" or "Maggie May" return a disambiguation page or another article.
    `fetch.py` refuses disambiguation pages and redirects (a seed must be the
    canonical title), and the covers' `keep_sections: [Introduction, Beatles]`
    fails loudly on an article about another song.

15. **`strip_code()` deletes every template, including the ones that carry
    text:** sharps, flats, time signatures and quotation boxes. `normalize.py`
    replaces those by their text first; the audit lists the templates still
    deleted, most frequent first, to read once.

---

## 7. Code conventions

- **Everything in English**: code, comments, docstrings, documentation.
- **Type hints on every signature.** No exceptions.
- **Docstrings explain *why*, not *what*.**
  Good: `"""Adds the section offset so chunk positions are document-level, which
  is what the evaluation compares against."""`
  Bad: `"""Adds an offset."""`
- **Functions under about thirty lines, one responsibility each.**
- **No clever one-liners.** An explicit loop beats a three-level nested
  comprehension.
- **Pure functions wherever possible**, especially in ingest and index: data in,
  data out, no side effects. Those are the ones that can be tested with three
  sample articles.
- **One file, one responsibility.** If `normalize.py` starts downloading things,
  something has leaked.
- **`pathlib` everywhere**, never string paths with separators.
- **Fail loudly.** A missing article title, a failed assertion or a song album
  with no album article must raise. Silent skips poison the evaluation.
- **General rules, not exceptions.** A rule that applies to one article is data
  in `seeds.yaml`, not an `if` in the code.

### Build-time assertions

These catch bugs that produce no error but invalidate every measurement:

1. **Fetch:** no raw document has a null `revid`, and none is a disambiguation
   page or a redirect.
2. `full_text[char_start:char_end] == chunk_text` for every chunk.
3. **Normalise:** every raw file is listed in `seeds.yaml` and every seed has a
   raw file; every `keep_sections` entry matches a section.
4. Every evaluation evidence quote exists verbatim in **its section** (looked up
   by `section_id`), not merely somewhere in the document.
5. No chunk has an empty header field, per its `doc_type`'s `header_fields`
   (without `Section` for the article-unit strategy).
6. `get_query_embedding(x) != get_text_embedding(x)`, proving prefixes apply.
7. The embedder's effective truncation length is at least the largest chunk in
   WordPiece tokens (trap 13).

---

## 8. What not to do

- Do not add an answering mode or an ungrounded second answer block.
- Do not add a second narrative source. There is no good openly licensed one, and
  it would reintroduce mixed trust levels across the corpus.
- Do not use the Batch API. The savings are negligible at this volume and the
  turnaround can be a full day.
- Not in this version: a provider or model selector in the UI, or a claim of
  multi-provider support. `base_url` is configurable, but only one provider is
  tested and evaluated; another model would need its own evaluation run.
- Do not cache embeddings, of chunks or of questions. Rebuilding a collection
  costs minutes of CPU, an eval run a few more; a stale cache gives wrong
  numbers silently.
- Do not compare metrics across different corpus versions or evaluation
  versions.
- Do not inspect individual failures in the held-out test split. Read the number
  and nothing else; looking at them turns the test split into a second dev split.
