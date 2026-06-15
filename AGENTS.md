# Development Agent Notes

This repository contains Sentinelle Validation System, a reusable validation-design tool for tabular prediction workflows.

## Principles

- Keep the core validation logic independent from any specific agent or LLM.
- Preserve the CLI, Python package, and local web interface.
- Treat `.claude/skills/validation_design/SKILL.md` as an optional adapter, not a runtime dependency.
- Do not commit datasets, generated artifacts, cache folders, or local environment files.
- Keep user-facing copy concise and product-oriented.

## Local checks

```bash
python -m pip install -r requirements-web.txt
cd frontend
npm install
npm run build
cd ..
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

CLI smoke test:

```bash
python scripts/validation_design_agent.py \
  --demo \
  --target target \
  --output-dir artifacts/validation_design_demo \
  --run-cv-benchmark \
  --model-name random_forest \
  --n-splits 3
```
