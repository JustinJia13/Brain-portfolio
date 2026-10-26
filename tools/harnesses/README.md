# Manual harnesses

One-off diagnostic scripts, each with its own `argparse` entrypoint. They are **not**
unit tests despite their original `test_*.py` names -- they contain no test functions and
most call the live WorldQuant BRAIN API, so they are not collectable by `pytest`.

The real unit tests are in `alpha_metric/tests/` and run offline:

```bash
python -m pytest alpha_metric/tests/ -q
```
