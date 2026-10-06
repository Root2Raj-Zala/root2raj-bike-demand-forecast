# Reproduce and inspect

Tested locally with Python 3.12 and the pinned top-level dependencies in `requirements.txt`. No accounts, cloud keys or paid services are needed. The source download uses HTTPS and stays in `data/`; generated local databases and row-level files are ignored by Git.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python forecast.py
python -m unittest discover -s tests -v
```

The first command downloads the real source if it is missing. Read `data/SOURCE.md` before reuse. Outputs include measured results, validation checks and published aggregate evidence. Data/schema assertions fail explicitly if the upstream source changes. The workflow definition repeats the run and tests in GitHub Actions; a workflow file alone is not proof of a passing cloud run.

## Scope of execution

This is a working local reference implementation, not a deployed enterprise service. Source files, transformations, quality policy and limitations are reviewable. Distributed orchestration, permissions, monitoring infrastructure and operational SLAs are not claimed.
