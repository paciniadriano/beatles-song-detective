---
title: Beatles Song Detective
emoji: 🎸
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: How the Beatles' songs were made, with cited sources
datasets:
  - adrianopacini/beatles-song-detective-index
models:
  - nomic-ai/nomic-embed-text-v1.5
  - Qdrant/bm25
---

# Beatles Song Detective

Discover how The Beatles' songs were created: curiosities, stories, people
involved, technical equipment and much more. Ask a question in any language;
the answer is built from English Wikipedia articles, and every claim is cited with a
link to the exact revision of the article it came from.

**Try it:** [Beatles Song Detective on Hugging Face Spaces](https://huggingface.co/spaces/adrianopacini/beatles-song-detective)

**Code:** [paciniadriano/beatles-song-detective on GitHub](https://github.com/paciniadriano/beatles-song-detective)

> **Required: a password.** Type it in the box at the top of the page; ask
> me for it. The app answers with my OpenAI key, so each user is limited to
> 10 questions every 5 minutes (a counter per IP address, reset 5 minutes after
> the first question). A question costs about $0.0002–0.0005, an estimate
> is shown under each answer.
>
> No key or password is stored in this repository: they are secrets of the
> Space (`OPENAI_API_KEY`, `APP_PASSWORD` and an admin password
> `APP_UNLIMITED_PASSWORD`, which has no limit). The steps that cost money
> (writing the evaluation questions and running the evaluations) ran offline,
> on my own key.
>
> Questions asked on the Space are saved, with their answers and searches, to
> a private dataset, to improve the app. No IP address, password or other
> personal data is saved. A local run saves nothing. The owner reads them with
> `python -m src.app.show_questions` (a list, and `data/questions.csv`).

## Why I chose this topic for my RAG system

I wanted to build something for musicians. I'm a musician and I thought of a lot of ideas but they were all very difficult to achieve in a reasonable amount of time.
I'm also a Beatles fan so the idea started shaping itself in my mind and then it all made sense. I also wanted to create something artistic and not so common.

## How it works

You ask a question related to The Beatles' songs.
You can write a simple and direct question or you can write something like: "How was this song written?" pointing to a prior turn in which you mention that song.

The model streams the answer based on the data it has, citing each claim as [1], [n]… 
If there is no data to answer the question, it should politely say that the sources don't cover it.

The corpus is 326 English Wikipedia articles (songs, albums, people, studios,
equipment and techniques) frozen at fixed revisions in September 2026. 

The search index is published as a dataset:
[adrianopacini/beatles-song-detective-index](https://huggingface.co/datasets/adrianopacini/beatles-song-detective-index).

## Optional features that this project has

1. Domain-specific application: The Beatles' songs from Wikipedia
2. Evaluation: code, question sets and published results (scripts in `src/eval/`,
   datasets and runs in `eval/`, every stage in [`results.md`](results.md))
3. Hybrid search: dense + BM25
4. Streaming responses: it can be seen in the chatbot
5. Function calling: the model calls a "search" tool
6. Structured JSON outputs: used to create the evaluation dataset. The question
   generator (`src/eval/generate.py`) asks the model for a fixed JSON schema
   (question, answer and the exact quote that supports it), so every question
   can be checked against its source section automatically.
7. Prompt caching: the instructions and the `search` tool definition (about
   1,300 tokens) are identical on every call, and the messages are ordered so
   they always come first: instructions → earlier turns → question → search
   results. OpenAI asks for a minimum of 1,024 tokens to "activate" the cache.

### Tried, but left out of the app

The code for two more features is still in the repository, switched off.
I tried them, but neither earned its place in the app:

- **Reranker** (`src/rag/rerank.py`). I tried two models, MiniLM
  (`Xenova/ms-marco-MiniLM-L-6-v2`) and Jina turbo
  (`jinaai/jina-reranker-v1-turbo-en`). MiniLM, the one that got a full run, tied (see *How I chose the configuration* below) and made each
  search about six times slower, so it stays off (`reranking.active: null` in
  `config.yaml`).
- **Metadata filtering** (the `filters` parameter of `search()`). Every chunk
  carries its article type (song, album, person...) and its topics, and the
  search can filter on them. I never turned it on in the app: a wrong filter
  hides the chunk that has the answer, and no automatic filter convinced me
  it was worth that risk.
  The only option that made sense was letting the user pick an album in the
  UI, which felt impractical for the way people ask about songs.

## Results - The short version

On questions used to test the app, the search finds the right chunk 96% of the time and on a set of specially curated questions 85% of the answers pass all checks (correct, faithful to the sources, or declined when they should be).

The full story, stage by stage, with every table and the reasoning behind each
decision, is in [`results.md`](results.md).

### How I measured this

**Did the search find the right chunk?** I generated one question for every
eligible section of the corpus (1,771 questions kept out of 1,775 sections), each with the exact sentence
that answers it. I decided to split them into two groups: 1,240 questions to make decisions with (dev), and 531 kept aside and run only once at the very end (test). 
On those 531:
- the chunk with the answer is among the 8 sent to the model for **95.9% (±1.7)** of questions;
- that chunk is ranked first for 66% of questions and in the top 3 for 87% (MRR 0.775).

**Did the model use it well?** I wrote 104 questions with the help of AI and then curated and graded every answer myself to analyze the correctness and the faithfulness. 73 can be answered from the corpus; 31 can't (things the articles don't cover, counts, song lyrics, and a couple of prompt-injection attempts), and those should get a polite "the sources here don't say".

| | Result |
| --- | --- |
| Pass all checks | **88 of 104 (85% ±7)** |
| Correct (answerable questions) | 71 of 73 |
| Faithful: every sentence backed by the chunk it cites | 66 of 73 |
| Declined when it should | 24 of 31 |

**Does the retrieval actually help?** The model already knows a lot about
the Beatles, so I asked it the same 73 questions with no search and no
sources at all, just the model on its own. It
got 62 right on its own. With the app it got 71: it fixed 10 of the 11 it had
missed and broke 1 it already knew (the search missed its chunk). That's
+12 points, beyond the margin of error (±8.5). Both sides are compared on
correctness only, since faithfulness can't be graded without sources; the gain
and its margin come from the `correct` and `closed_book_correct` grades, with
the paired formula below.

### How I chose the configuration

A new option replaced the current one only if it won by more than **the
margin of error of a paired comparison (a 95% margin)** and by **more than 2
percentage points** (the smallest gain worth a change); otherwise it was a
tie, and the simpler option stayed. The comparison is paired because both
runs answer the same questions.

```
Margin formula:

mean   = sum of d / n
sd     = √( sum of (d − mean)² / (n − 1) )     (standard deviation)
margin = 1.96 × sd / √n
```

where n is the number of questions and d is each question's score: +1 if
only the new option got it right, −1 if only the current one did, 0 if both
agree.

This table summarizes what you see in [`results.md`](results.md).

| Decision | What I compared | What happened |
| --- | --- | --- |
| Chunking | Cutting articles blindly vs. respecting sections | Respecting sections won by 3.2 points. Sizes of 256, 512 and 1,024 tokens tied, so **512**, the default, stayed for now |
| Overlap | 512 tokens with and without overlap, and again at 256 | Tie both times, so **no overlap** |
| Embedding model | `nomic-embed-text-v1.5` (local, free) vs. OpenAI `text-embedding-3-small` | Tie (92.1% vs 92.0%), so **nomic** stayed |
| Hybrid search | Dense vectors only vs. Dense + BM25 keywords | **Hybrid** won by 2.4 points |
| Reranker | Adding a cross-encoder after the search (MiniLM and Jina turbo) | +1.3, a tie, so **no reranker**. Re-checked at the end with the winning configuration: still not enough |
| `alpha` | Dense/Sparse (score = alpha × dense + (1 − alpha) × sparse) balance from 0 (sparse only) to 1 (dense only): 0, 0.25, 0.5, 0.75 and 1, at all three sizes | 0.5 and 0.75 tied. **0.5** stayed |
| How many chunks | 512-token chunks × 5 vs. 256-token chunks × 8 | Smaller chunks, more of them, similar number of tokens: +3.5 points. Against 512 × 7 it tied, with 31% less text. So **256** won. |
| Chunks per section | At most 1, 2, 3 or no limit | 1 lost by 4 points; **2** stayed |

**Using an agent.** The user's question does not go straight to the search.
The app is a small agent: the model first rewrites the question as a search
query in English, and then answers from the chunks it gets back. The
rewrite in English mattered to me because I'm from Argentina, a
Spanish-speaking country, and I wanted to be able to ask in Spanish. The
model answers in the language of the question. The agent also helps with
follow-up questions such as "and who played the brass on it?": it uses the
last 3 turns of the chat to rewrite the query.

I checked how many searches the agent should be allowed per question: one,
or a second one if the first was not enough. A second search can only help
when the first one missed the right chunk. In the answer set the first
search missed it for 9 answers, but only one of them failed because of it, and
a second search would have needed to rescue at least 4 failed answers to make
a measurable difference. So the agent makes **one search** per question.

  The final configuration achieved is:
  - Chunking: 256 tokens/0 overlap containing a header with specific section context 
  - Local embedding: nomic-embed-text-v1.5
  - Hybrid-search (dense and sparse fused equally: alpha 0.5)
  - Top k chunks: 8
  - Max amount of chunks per section: 2
  - No reranker
  - One search per question

> Every search run is in `eval/experiments.csv`, with its question-by-question results in `eval/runs/`; the graded answer runs are in `eval/generation/runs/`.

### Limitations: what these numbers don't tell you

- **The search questions are probably easier than real ones.** A language
  model wrote each question while looking at the section that answers it, so
  questions tend to reuse the section's words, which helps the keyword half of
  the search. People asking in their own words will be different, and this may
  have tilted the hybrid-search and `alpha` decisions towards keywords.
- **Every grade was decided by one person: me.** The questions were written
  with the help of AI and checked by a separate review agent, and review
  agents proposed a grade for each answer, but I confirmed or changed every
  grade myself, with no second human grader. With 104 questions the margin is
  ±7 points, so the 85% is a rough level, not a precise score.
- **The prompt was adjusted once on the same 104 questions.** After the first
  run I changed it so it would decline more reliably, and stopped there. The
  final 85% is measured on questions the prompt was tuned against, so it may
  be slightly optimistic.
- **The comparison without sources is tilted on purpose.** Over half of the
  answerable questions (38 of 73) ask for details the model is unlikely to know, which is where retrieval helps most. On famous facts the gap would be smaller.
- **Everything was tested in English.** The app answers in the language of
  the question and always searches in English, but I never measured how good
  the answers are in other languages.
- **A fact that appears in two articles counts as a miss** if the search
  finds it in the "wrong" one (a song's article instead of its album's). This
  makes the search numbers a bit worse. It adds similar noise to every
  option: it can hide small differences, but is unlikely to flip a decision.
- **The results belong to one model.** All answers were measured with
  `gpt-6-luna` at temperature 0, which does not make the answers fully
  deterministic, and the final answer set was run once. Another model would need its
  own run of the answer set.
- **The app knows only what those Wikipedia articles say.** When they don't
  cover something, it is built to reply "the sources here don't say", though
  it doesn't always manage it (see the results table above).
- **Only Wikipedia.** A richer corpus would include books, interviews and
  session notes, but the index is published in a public repository, and
  only openly licensed text can go there: Wikipedia's CC BY-SA allows it,
  books and interviews don't. Lyrics were out for the same reason. Mixing
  in a second kind of source would also mix levels of trust, while the app
  cites every chunk the same way. The cost is detail: Wikipedia rarely
  says where a microphone was placed or how a device was set, which is
  exactly what some fans would ask.

## How this system was built

I designed the architecture of the whole project from the beginning based on what I learned in the course and added my previous knowledge and my research on each necessary element of a RAG system (LLM, embedding models, rerankers, vector database and specific libraries such as LlamaIndex, fastembed, etc.).

An AI coding assistant helped me write the code. I went through every stage myself, checking the code, modifying the necessary things, reading the results of each evaluation and making each decision based on it.

[`CLAUDE.md`](CLAUDE.md) is the engineering brief I used: the design, the reasons behind each choice, and the bugs/traps that were found by running code. It was not written once at the start: I kept updating it as the project moved forward, adding each decision and each trap as it came up, so it describes the app as it ended up.

## Lessons Learned

The main focus was to get the correct corpus, as clean and complete as I could possibly get it. I spent more than half of the time of the whole project achieving this. I tried a lot of things and at last I came to the idea of having the list of the articles in a YAML file where I could control the fetching of the correct articles, the sections and the grouping of things, such as doc types and topics. The way in which Wikipedia has everything curated and well separated in sections with their own title was really useful. It helped me a lot to get such good numbers in my retrieval because there was very little noise in my data.

The other thing I wanted to mention is the importance of the evaluation process. My project was centered on the data and on the evaluation. Each evaluation stage was the guide for every change I made, that's one of the main lessons I learned from the course.

## For future considerations

I had to stop developing ideas to be able to deliver the app to get a certification, and a lot of things were left out.

### Improvements to what is already there

  - Improve the system prompt to achieve a pass rate above 85%
  - A new set of generation questions, split in two: one part to tune the prompt on and one run only once to report the result, as the dev/test split does for the search. Today the 85% is measured on the same questions the prompt was adjusted against
  - A more flexible check for declining. Today an answer passes only with the exact sentence "the sources here don't say", so a few correct refusals written in other words (for example, declining to quote lyrics) count as failures
  - Better faithfulness, one of the two places where the app fails most (66 of 73): enforce or verify a citation on every sentence (the prompt asks for one), and check that the cited chunk really holds what the sentence claims
  - Remove "(1962–1970)" from the system prompt (`src/rag/prompt.md`, and the same line in `src/eval/closed_book_prompt.md`). I noticed this at the end, while gathering all the info to write this README file. The corpus also has the later songs (*Free as a Bird*, *Real Love*, *Now and Then*), so the range doesn't match. Changing it now would mean the app no longer runs the prompt that was evaluated, and I would have to re-evaluate the 104 answers.

### New ideas I haven't implemented (yet)

  - A SQL database to resolve queries such as "How many songs did John write?" or "In how many songs does George play bass?", adding a new tool for it
  - Possibility to change the model between Luna and a local model. I wanted to test how a tiny local model performs against a big one. 
  I would have loved to have the system running without paying a single cent and with an excellent performance.
  - Save the real user's questions to improve the system in the future
  - Filters in the UI for the user to interact with, if a practical form comes up
  - Create a Trivia mode to play

## Running and reproducing it

### Running it locally

The search index is not in the repository (`data/qdrant_store` is published as a
dataset), so the quickest way is to let the app download it at startup:

```bash
pip install -r requirements.txt
DATASET_REPO=adrianopacini/beatles-song-detective-index python app.py
```

Locally no password is needed (the box can stay empty); the answers use the
OpenAI key in `.env`. If you build the index
yourself (the steps below up to `python -m src.index.publish`), run
`python app.py` without `DATASET_REPO`: the app opens
`data/qdrant_store`.

### Reproducing the corpus, index and evaluation

Every step that built the corpus, the index and the evaluation is a script in
`src/`. They need `requirements-dev.txt` and a `.env` file (see `.env.example`):
`fetch` uses `WIKIPEDIA_CONTACT_EMAIL`, and `generate` and `run_generation`
call OpenAI with `OPENAI_API_KEY`.

`fetch` downloads the current revision of each article, and the raw files
of `corpus_v1` are not published, so a rebuild today gets slightly different
text. The method can be reproduced, but not `corpus_v1` exactly: a retrieval
question whose quote has changed on Wikipedia would fail the evidence check.

```bash
python -m src.ingest.fetch         # fetch articles listed in src/ingest/seeds.yaml
python -m src.ingest.normalize     # clean them into sections
python -m src.index.chunking       # cut the sections into chunks
python -m src.index.build          # embed and index in Qdrant
python -m src.index.publish        # copy the chosen index to data/qdrant_store
python -m src.eval.generate        # write the retrieval questions
python -m src.eval.split           # split them into dev and test
python -m src.eval.run_eval        # score a search configuration
python -m src.eval.run_generation  # answer the generation set
python -m src.eval.run_generation --closed-book  # the same questions, with no sources
python -m src.eval.generation_report RUN_ID # summarise a graded run
```

`RUN_ID` is the name of a folder in `eval/generation/runs/`: `run_generation`
creates one per run, named after the date, time and configuration, and prints it when it starts. The answer-set table in this README comes from the command below (the
closed-book comparison, +12 ±8.5, was computed from the same two runs with the
paired formula):

```bash
python -m src.eval.generation_report 2026-09-30_1830_section_t256_o0_k8_steps1 --closed-book 2026-09-30_1557_closed_book
```

## Licences

- **Code:** MIT (see `LICENSE`).
- **Text:** from English Wikipedia, under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This covers
  the published index and the question files in `eval/`.
- **Models:** `nomic-ai/nomic-embed-text-v1.5` (Apache 2.0) for dense
  vectors, `Qdrant/bm25` for sparse ones; answers by OpenAI's `gpt-6-luna` with the owner's key.
- **Font:** [Inter](https://rsms.me/inter/), under the SIL Open Font
  License 1.1; the file and its licence are in `src/app/fonts/`.
- **Title image:** set in [Bootle](https://www.fontspace.com/bootle-font-f2373)
  by Northern Fonts, a freeware font licensed for personal and commercial use. I lengthened the T in an image editor, in the style of the Beatles' logo, as a homage. It is not the band's logo, and no album covers or photos are used.
  This is a fan project, not affiliated with the Beatles or Apple Corps.

## Thank you!

I really enjoyed the course. It gave me a lot of tools and a new way of thinking: the criteria to know when an LLM is worth using in a system and how to use it. It also cleared up a lot of doubts I had about the strange and fascinating world of AI. This is just the beginning for me, thanks!