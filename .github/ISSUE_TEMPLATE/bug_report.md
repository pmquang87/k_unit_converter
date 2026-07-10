---
name: Bug report
about: Something converted wrong, or a deck won't convert
title: ""
labels: bug
assignees: ""
---

**What happened**
A clear description of the problem.

**The offending keyword block**
Paste the smallest `*KEYWORD ... *END` snippet that reproduces it (a single
keyword and its cards is usually enough):

```
$ paste the keyword block here
```

**`kunit check --json` output**
Run this on the deck (or the snippet) and paste the result:

```
$ kunit check <deck>.k --json
```

**Command you ran**
e.g. `kunit convert deck.k --to ton-mm-s`

**Expected vs actual**
What you expected to happen, and what happened instead (include any error
message or the relevant lines of the converted output).

**Environment**
- kunit version (`kunit --version`):
- Python version:
- OS:
