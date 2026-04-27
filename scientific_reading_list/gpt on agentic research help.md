You can think of your future “thesis agent” as a very fast, very consistent research assistant that sits across three main assets:

- your **thesis draft** (structure, arguments, writing style),
- your **codebase** (methods, experiments, analysis),
- your **Zotero library** (papers, PDFs, notes, tags).

Here are concrete categories of work it can offload, beyond fixing code.

---

## 1. Literature & theory work

### a) Mapping and organizing the literature
- **Thematic clustering** of your Zotero library:
  - “Group my papers into themes: theory A, theory B, methods, applications.”
  - “Show me which papers use method X vs method Y.”
- **Gap spotting** (at least at a rough level):
  - “Given my draft and my Zotero collection, what obvious subtopics I haven’t cited or discussed?”
- **Reference triage**:
  - “From these 60 papers in Zotero, which 10 are most central to section 2.3 of my thesis?”

### b) Summarizing and comparing papers
- Short structured summaries of individual papers:
  - Problem, method, data, key results, limitations.
- **Comparative matrices**:
  - “Summarize how papers A, B, C define concept X and how they differ.”
  - “Compare the assumptions of models in these 5 ML papers.”
- **Tracing definitions and debates**:
  - “Explain how the notion of $\textit{[your key concept]}$ evolved across these 7 papers, in timeline order.”

### c) Linking theory to your work
- “Given chapter 2 and these 15 papers, which theories directly justify my choice of method?”
- “Which theoretical criticisms of approach X (from my Zotero library) apply to my method section?”
- “What are the strongest theoretical arguments *against* my chosen framing?”

---

## 2. Thesis-structure and argumentation

### a) Global structure and coherence
- Check for **argument flow**:
  - “Read my introduction and conclusion. Are my research questions, methods, and claims aligned?”
  - “Do chapters 3 and 4 actually answer the research questions stated in chapter 1?”
- Suggest **restructuring**:
  - “Propose 2–3 alternative outlines of the thesis that improve logical order and reduce redundancy.”

### b) Local argument strengthening
- For each section:
  - “What claims here are unsupported by citations, according to my Zotero library?”
  - “Where is the reasoning weak or hand-wavy? Suggest more precise formulations or needed evidence.”
- Generate **counterarguments**:
  - “Play devil’s advocate: what are the strongest objections to the argument in section 4.2?”

### c) Consistency of terms and notation
- Check that you use key terms and symbols consistently:
  - “Scan the thesis draft: list all distinct notations for the same quantity and propose a unified scheme.”
  - “Identify places where I subtly change the meaning of ‘robustness’ / ‘validity’ / ‘generalization’.”

---

## 3. Methods and experimental design (conceptual level)

### a) Clarifying and justifying methodology
- “Given my research questions, suggest appropriate methods and explain tradeoffs.”
- “Help me write a rigorous justification for why I use method X instead of Y, grounded in my Zotero papers.”
- “Check whether my experimental design risks common biases (e.g. data leakage, p-hacking, confounding).”

### b) Designing ablations and sanity checks
- Suggest **robustness checks**:
  - “Given my model and dataset, what ablation studies would strengthen my claims?”
  - “What simple baselines should I include to make my results more convincing?”
- Critique your plan:
  - “Read my methodology section and list potential threats to validity (internal/external).”

---

## 4. Code–theory integration

You already use it to fix code; you can go further:

### a) Code as a test of understanding
- “Explain in plain language what this function does and how it connects to section 3.2 of my thesis.”
- “Is my implementation of algorithm from Paper X faithful to the original? Where do I deviate?”

### b) Producing method descriptions from code
- Turn code into a **methods write-up**:
  - “Given this training script and model definition, draft the ‘Model’ and ‘Training’ subsections.”
- Ensure **consistency between results and text**:
  - “Cross-check that all reported hyperparameters and dataset splits in chapter 4 match the code.”

---

## 5. Results analysis and interpretation

### a) First-pass interpretation
- “Here are my experimental results (tables/figures). What main patterns and anomalies do you see?”
- “Which results directly support or contradict my hypotheses in section 1.3?”

### b) Alternative explanations and caveats
- “Suggest plausible alternative explanations for the performance gap in Table 3.”
- “Help me phrase limitations of my empirical results clearly and honestly.”

### c) Visuals and reporting
- “Propose clearer ways to visualize these results to highlight the key takeaway.”
- “Check whether my reporting of metrics respects good practice (mean, variance, confidence intervals, etc.).”

---

## 6. Writing, editing, and style (research-grade, not just grammar)

### a) Clarity and precision
- Improve wording for **technical clarity**:
  - “Rephrase this paragraph to be more precise while keeping the same claims.”
- **Compression**:
  - “Shorten this 400-word section to 250 words without losing key technical content.”

### b) Genre alignment
- “Make this section sound more like a methods section in [target journal/conference].”
- “Turn this informal explanation into a formal theorem-style statement, then a proof sketch.”  
  (You should still carefully check the math.)

### c) Cross-document consistency
- “Ensure key terms are defined exactly once, in chapter 1 or 2, and used consistently afterward.”
- “Align notation between the thesis draft and the notation used in my main reference paper.”

---

## 7. Citation management and formatting

Since you’re using Zotero:

- **Citation suggestions**:
  - “For this paragraph (pasted here), which items in my Zotero library are likely relevant citations?”
- **Deduplication and consistency**:
  - “Find likely duplicate entries in my Zotero library and propose a canonical entry.”
- **Bibliography sanity checking**:
  - “Check if all in-text citations have corresponding bibliography entries and vice versa.”
- **Style compliance**:
  - “Format these references according to [APA, IEEE, Chicago, etc.].”

---

## 8. Planning and project management

- Break down your work:
  - “Given my current draft and TODO comments, suggest a 4-week plan to reach a complete first draft.”
- Track and update:
  - “Create a checklist of open problems and missing analyses mentioned in comments across the thesis and code.”

---

## 9. What it should *not* fully replace

To stay rigorous and original, keep these mostly human-led:

- **Core insight and research questions**: the conceptual jump should be yours.
- **Final judgment of evidence**: the agent can suggest, but you decide what counts as convincing.
- **Critical reading of key papers**: use summaries as a guide, not a substitute.
- **Formal proofs/derivations**: an agent can sketch, but you must verify and often refine.

---

## If you want, I can help you design the setup

If you tell me:
- your field,
- your thesis topic,
- what tools you’re comfortable with (e.g. Obsidian, Overleaf, VS Code, Zotero plugins),

I can outline a concrete workflow like:

- how to connect the agent to Zotero (via exports/API),
- how to give it structured access to your thesis (e.g. per-chapter),
- how to expose your codebase sensibly (e.g. via a repo index),
- and example “prompt templates” for the kinds of tasks above that you’ll use repeatedly.