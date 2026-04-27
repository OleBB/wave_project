Ask the *first* agent to produce **structured, source‑anchored “maps”**, not essays.

The goal: later agents can **reload context + structure** with **minimal hallucinated interpretation**. That means:

- highly **extractive** (close to the text),
- **explicit about where each piece came from**,
- **machine‑friendly structure** (tables, bullet lists, JSON‑like).

Below are concrete summary types and prompt templates you can use.

---

## 1. Per‑paper cards (Zotero items)

Have the first agent create one **card per reference**, using only the paper + your notes, no outside guessing.

**Format (example):**

```markdown
# PAPER_CARD
id: smith2020_deeplearn
zotero_key: ABC123
title: ...
authors: ...
year: 2020
tags: [theory_X, method_Y, dataset_Z]

claims:
  - c1:
      text: "Main claim in the paper, as stated."
      type: main
      support: [e1, e2]
      page_refs: [p3, p4]

evidence:
  - e1:
      text: "Core experiment/result supporting c1."
      page_refs: [p5]
  - e2:
      text: "Analytical or theoretical argument for c1."
      page_refs: [p6-p7]

assumptions:
  - a1:
      text: "Key assumption, quoted or very close paraphrase."
      page_refs: [p2]

limitations_author:
  - l1:
      text: "Limitations explicitly acknowledged by the authors."
      page_refs: [p8]

limitations_inferred:
  - l2:
      text: "Limitations that follow directly/logically from descriptions."
      page_refs: [p8]
      note: "My inference; grounded in the text."

relevance_to_thesis:
  - r1:
      text: "Which part of my thesis this paper is relevant for."
      thesis_sections: [2.3, 4.1]
```

**Prompt template:**

> You are not allowed to introduce any information that is not present in the paper or in the notes I provide.
> 
> Read this paper/notes and write a `PAPER_CARD` in the exact schema below.
> 
> - Be *maximally extractive*: favor short quotes or very close paraphrases.
> - Every claim, assumption, or limitation must be traceable to the text (include page or section refs).
> - If you infer something (not directly stated), mark it `limitations_inferred` and justify it with explicit references.
> - Do **not** discuss whether the paper is good or bad; no evaluation, only representation.
> 
> Use this schema:
> ```markdown
> # PAPER_CARD
> id: ...
> zotero_key: ...
> title: ...
> ...
> ```
> 
> Here is the material:
> [paste PDF text / summary / Zotero note]

Later agents can just search or filter over `claims`, `assumptions`, `relevance_to_thesis`, etc.

---

## 2. Literature map (thematic, cross‑paper)

Once you’ve got paper cards, ask for **maps**, but still structured and traceable.

**Format:**

```markdown
# LITERATURE_MAP
topic: "My core topic"
last_updated: 2026-04-25

subtopics:
  - name: "Theory of X"
    core_papers: [smith2020_deeplearn, lee2018_theoryX]
    summary:
      text: "Purely integrative summary, built from PAPER_CARD.claims."
      source_cards: [smith2020_deeplearn.c1, lee2018_theoryX.c2]
    disagreements:
      - d1:
          issue: "Definition of X"
          positions:
            - paper_id: smith2020_deeplearn
              claim_id: c2
            - paper_id: lee2018_theoryX
              claim_id: c3
    open_questions:
      - q1:
          text: "Question that emerges from gaps/limitations."
          grounded_in:
            - paper_id: smith2020_deeplearn
              limitation_id: l1
            - paper_id: lee2018_theoryX
              limitation_id: l2
```

**Prompt template:**

> You have access to several `PAPER_CARD`s in the format below: [paste a few or representative ones].
> 
> Build a `LITERATURE_MAP`:
> 
> - You may only use information present in the `claims`, `assumptions`, and `limitations` fields.
> - When you synthesize (e.g., “most papers say X”), always list which cards/claim IDs this is based on.
> - No personal evaluation (“this is important”, “this is weak”). Stick to what is stated or trivially implied.
> - Represent:
>   - subtopics,
>   - where papers agree/disagree,
>   - questions that logically follow from stated limitations.
> 
> Output strictly in this schema:
> ```markdown
> # LITERATURE_MAP
> topic: ...
> ...
> ```

Later, another agent can jump in by loading `PAPER_CARD`s + `LITERATURE_MAP` and *see* where evidence and disagreements are.

---

## 3. Thesis argument map (for your draft)

Have the first agent convert your **thesis draft** into an **argument graph**: claims, support, dependencies, aligned with sections.

**Format:**

```markdown
# THESIS_ARGUMENT_MAP
version: "draft_2026-05-01"
research_questions:
  - rq1:
      text: "..."
      sections: [1.2, 4.1]

claims:
  - t1:
      text: "Main thesis claim."
      sections: [1.3, 5.1]
      supported_by:
        - evidence_id: e_thesis_1
        - literature_claims:
            - paper_id: smith2020_deeplearn
              claim_id: c1

evidence_internal:
  - e_thesis_1:
      type: experiment
      sections: [4.2]
      description: "What was done, extractive from the thesis."
      linked_code_paths: ["experiments/run_experiment_x.py"]

definitions:
  - d1:
      term: "Key concept"
      definition: "Exact or near‑exact quote from the thesis."
      first_defined_section: 2.1
      later_uses_sections: [2.3, 3.1, 4.2]
```

**Prompt template:**

> You are turning my thesis draft into a structured argument map.
> 
> Constraints:
> - Do **not** add new claims not in my draft. You can rephrase for clarity, but not change content.
> - For every claim, indicate the sections where it appears.
> - For each high‑level claim, list which internal evidence and which `PAPER_CARD` claims support it.
> - Mark any claim that lacks direct support in my text as `unsupported`.
> 
> Use this schema:
> ```markdown
> # THESIS_ARGUMENT_MAP
> version: ...
> research_questions: ...
> claims: ...
> evidence_internal: ...
> definitions: ...
> ```
> 
> Here is the thesis text:
> [paste chapter or whole draft]

Later, a new agent can answer questions like “What backs up t1?” without rereading the whole thesis.

---

## 4. Glossary + notation index

To avoid each agent inventing its own terminology, make a **global glossary**.

**Format:**

```markdown
# GLOSSARY
terms:
  - term: "latent representation"
    thesis_definition:
      text: "Exact quote or very close paraphrase of the definition in my thesis."
      section: 2.1
    literature_variants:
      - paper_id: smith2020_deeplearn
        claim_id: c2   # where it's defined
    notes:
      - "Later used loosely in section 4.3 (semantic drift)."
```

**Prompt template:**

> Extract a `GLOSSARY` from:
> - my thesis draft, and
> - the following `PAPER_CARD`s.
> 
> Rules:
> - For each term, include the *exact* or near‑exact thesis definition with section number.
> - List variants in the literature by linking to `PAPER_CARD` claim IDs; no new definitions.
> - Flag inconsistencies (“X is defined differently in thesis vs paper Y”) but do not resolve them; just describe.
> 
> Output strictly in this schema:
> ```markdown
> # GLOSSARY
> terms:
>   - term: ...
>     thesis_definition: ...
>     literature_variants: ...
>     notes: ...
> ```

---

## 5. General style constraints to minimize “agent thoughts”

Whatever summary you request, include global constraints like:

> - Be **extractive‑first**:
>   - Prefer short quotes or tight paraphrases.
>   - For each important statement, include a reference: section, page, or `PAPER_CARD` ID.
> - No evaluation:
>   - Do not say “important”, “novel”, “strong”, “weak”, etc.
>   - Do not judge quality of methods or results.
> - No external knowledge:
>   - Only use information explicitly present in the material I provide.
>   - If something is a necessary logical consequence, label it `inferred` and show the supporting source statements.
> - No “suggestions” or “opinions” unless explicitly asked in a separate section.
> - Output in the exact schema I provide (so a later agent can parse it programmatically).

You can even add a “provenance” field:

```markdown
provenance:
  built_from:
    thesis_sections: [1.1-1.4, 2.1-2.3]
    paper_cards: [smith2020_deeplearn, lee2018_theoryX]
  timestamp: 2026-04-25
```

---

## 6. How a later agent will use this

A later agent can:

- Load `PAPER_CARD`s to quickly see claims/assumptions/limitations and jump to the PDFs if needed.
- Load `LITERATURE_MAP` to understand the landscape and where disagreements are.
- Load `THESIS_ARGUMENT_MAP` to:
  - check alignment between research questions, methods, and claims,
  - see which claims lack support,
  - propose specific revisions (“add evidence for t3”).
- Use the `GLOSSARY` to keep terminology consistent and avoid re‑defining things.

If you tell me what you’re writing in (LaTeX, Word, Obsidian, etc.), I can suggest a minimal set of files (e.g. one `.md` per schema) and a concrete workflow for generating/updating them with your agent.