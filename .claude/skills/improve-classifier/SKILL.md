---
name: improve-classifier
description: Improves the review classifier with one measured change. Runs the eval, reads the confusion matrix, proposes a single prompt or taxonomy edit, shows the diff before changing anything, re-runs the eval and reports before/after numbers. Use when the user wants better classification accuracy or asks why categories get mixed up.
argument-hint: "[focus, e.g. controls_ux vs cheating_hackers]"
disable-model-invocation: true
---

Make one evidence-based change to the classifier. Optional focus: `$ARGUMENTS`.

## Ground rules

- **Tune on `dev`, report `test`.** Read only dev-split examples when looking for causes. Use test numbers to report the result, never to choose a change.
- **One change per run**, so the before/after numbers say what caused them.
- **Never edit the prompt or taxonomy without first showing the diff and getting the user's OK.**
- The prompt version is a hash of the system prompt and tool schema, so any edit to `config/taxonomy.yaml` or `src/review_intel/llm/prompts.py` creates a new version. The eval then re-classifies the 100 labeled reviews (about $0.02). Say so before running it.

## Steps

1. **Baseline** (free when the prompt hasn't changed, because predictions are cached):
   ```bash
   uv run review-intel eval --judge 0
   ```
   Note test and dev accuracy, macro-F1, sentiment accuracy, weak categories and the top confusions.

2. **Find the cause on dev.** Load the dev labels from `evals/labeled_reviews.jsonl` (`"split": "dev"`) and the current predictions from `evals/predictions/<prompt_version>__<model>.jsonl`, joined on `key`. List the dev reviews behind the biggest confusion, or the one in the focus: text, true label, predicted label. Name the pattern, e.g. "the game's own auto-aim is read as cheating".

3. **Propose one change.** Usually a sharper definition or a better example in `config/taxonomy.yaml`. Edit `_SYSTEM_TEMPLATE` in `src/review_intel/llm/prompts.py` only when the rule isn't about one category. Write the edit, show `git diff`, and explain which confusion it targets and what could get worse. **Wait for the user's OK.**

4. **After the OK:**
   ```bash
   uv run pytest -q tests/test_prompts.py tests/test_taxonomy_and_schemas.py
   uv run review-intel eval --judge 0
   ```

5. **Report before and after** in a small table: test accuracy, test macro-F1, dev accuracy, dev macro-F1, sentiment accuracy, and F1 for the targeted categories. Add one line on whether to keep the change. Judge by dev. If dev got worse or didn't change, recommend reverting with `git checkout -- <file>` and ask first.

6. **If kept**, offer to:
   - refresh the README table (`uv run review-intel eval --write-readme`; the judge adds about $0.06)
   - re-classify the stored reviews under the new version (`uv run review-intel classify --app <slug> --dry-run --limit 5000` for the cost first; use `--batch`)
   - commit as `feat(llm): ...` with the before/after numbers in the body

   Do each only on the user's yes.

With 35 test reviews, one review moves accuracy by about 3 points. The eval prints 95% bootstrap intervals: if the new score sits inside the old interval, call it noise unless dev agrees.
