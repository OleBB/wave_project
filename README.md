## Solo Workflow

**1. Always work on branches, never directly on main**
```bash
git checkout -b exp/descriptive-name
```

**2. Commit often with clear messages**
```bash
git add .
git commit -m "exp: tried random forest, F1=0.68"
```

**3. When the experiment works, merge to main**
```bash
git checkout main
git pull  # habit, even solo
git merge exp/descriptive-name
git push
```

**4. Delete the branch**
```bash
git branch -d exp/descriptive-name
```

## Why This Works

- **Main is always stable** - you can always return to working code
- **Branches are cheap experiments** - try things fearlessly
- **Clear history** - you can see what you tried and when
- **No merge conflicts** - you're the only one working

## Simplified Rules

- Name branches `exp/what-im-trying` for experiments
- If something works well, tag it: `git tag v0.1-baseline`
- Push to GitHub occasionally as backup: `git push origin exp/whatever`

## The One Safety Net

Before trying something risky:
```bash
git commit -am "safety: everything working before I break it"
```

Now you can always get back with `git log` and `git checkout <commit-sha>`.

---

## Pytest quick commands

- Run one file:
  !pytest -q test_sandkasse.py

- Run all tests in current folder:
  !pytest -q

- Run a single test function:
  !pytest -q test_sandkasse.py -k test_svaret

- Verbose output:
  !pytest -vv

- Stop at first failure:
  !pytest -x

- Show print() output:
  !pytest -s

- Re-run only failures from last run:
  !pytest --last-failed

- List available fixtures:
  !pytest --fixtures

- Show full help:
  !pytest -h

with open("tests/pytest_cheatsheet.md", "w", encoding="utf-8") as f:
    f.write(text)
print("Saved to tests/pytest_cheatsheet.md")


## Pytest:

Totally fair question. If you already do manual checks and have error handling, why bother with pytest?

Purpose of pytest in a data science project
- Prevent regressions automatically: What you validated today can silently break next week. Tests re-run the same checks on every change without you having to remember them.
- Codify assumptions as executable documentation: Tests make your data/algorithm contracts explicit (e.g., timestamps are monotonic, sampling is 1 Hz, peak distance ≥ N).
- Enable safe refactoring: When you optimize or restructure code, tests tell you instantly if behavior changed.
- Catch edge cases you didn’t exercise manually: Parametrized tests and property-based tests try lots of inputs quickly.
- Support collaboration and CI: Teammates and CI run tests on their machines/PRs so issues don’t slip into main.
- Reproducibility over time: With fixed seeds and “golden” baselines (metrics or small images), tests ensure future runs match expectations within tolerances.
- Faster feedback loop: A test suite runs in seconds. Manual plots and eyeballing take minutes and are easy to skip.

How tests differ from runtime error handling
- Error handling deals with “what to do when something goes wrong at runtime” (e.g., raise or fallback). Tests check that “things still behave as intended” during development.
- Example:
  - Error handling: If data has duplicates, raise ValueError.
  - Tests: Ensure the function raises ValueError given a tiny synthetic DataFrame with duplicates. That guards the contract and ensures your checks remain in place.

Concrete scenarios where pytest adds value beyond manual checks

- Visual vs. metric regression
  - You eyeball plots now, but a future change subtly shifts scaling or filters. A test that asserts summary metrics (mean, RMS, peak counts, dominant frequency) or image similarity within tolerance will catch it immediately.

- Pipeline drift
  - You change a resampling method or filter order. A test that runs a tiny synthetic signal through the pipeline and asserts the expected spectrum/peaks prevents unnoticed drift.

- Data quality gates
  - Tests on tiny sample inputs ensure: monotonic timestamps, max gap ≤ X seconds, NaN fraction ≤ Y%, amplitude range within bounds. These run automatically in CI, not just when you remember to check.

- Refactoring safety
  - You move logic from notebooks to modules, rename functions, or vectorize code. Tests verify outputs still match known baselines.

Minimal, high-impact test set to start with
- A few “smoke” tests that run your pipeline end-to-end on toy data.
- One or two golden metrics tests (e.g., known peak count and times on a synthetic waveform).
- A schema/quality test for input data.
- One fail-fast test that ensures your error handling triggers for bad inputs.



# REQUIREMENTS

conda env export --from-history
name: draumeriket
channels:
  - defaults
dependencies:
  - python=3.11
  - spyder=6.1.0
  - notebook
  - spyder-notebook
  - scipy
  - pytest
  - numpy
  - matplotlib
  - sympy
  - pandas
  - plotly
  - seaborn
  - pyarrow
  - tabulate
  - spyder-unittest
prefix: /opt/anaconda3/envs/draumeriket
