

## Git Cheat Sheet for a Solo Data Science Workflow

 Start an experiment branch
```bash
git checkout -b exp/try-xgb-baseline
```

 Stage and commit frequently
```bash
git add .
git commit -m "exp: add initial xgb with default params"
```

 Push branch and create PR on GitHub
```bash
git push -u origin exp/try-xgb-baseline
```

 Update local main before merging
```bash
git checkout main
git pull
```

 Squash-merge experiment into main
```bash
git merge --squash exp/try-xgb-baseline
git commit -m "feat: xgb baseline with tuned params and reproducible seed"
git push
```

 Tag a milestone
```bash
git tag -a v0.4-xgb -m "XGB baseline with F1=0.71 on stratified split"
git push origin v0.4-xgb
```

 Stash work-in-progress quickly
```bash
git stash push -m "WIP: trying SMOTE variations"
git stash apply
```

 Optional: create an annotated branch for a new feature or fix
```bash
 Feature branch
git checkout -b feature/add-scaling-pipeline

 Bug fix branch
git checkout -b fix/leak-in-validation
```

 Optional: keep a clean history by squashing locally
```bash
git rebase -i main
 mark commits you want to squash as 's'
```

 Optional: revert a bad commit safely (shared history)
```bash
git revert <commit-sha>
```

 Optional: find a regression with bisect
```bash
git bisect start
git bisect bad             #current commit is bad
git bisect good v0.3-baseline
 test each step, then:
git bisect good            #or
git bisect bad
git bisect reset
```

 Optional: work with two branches side-by-side
```bash
git worktree add ../repo-exp exp/try-new-split
 Now you have two working directories: repo/ and repo-exp/
```
"""


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
