# Tests

The packaged suite was checked against the corrected pipeline: 79 passed and two expected failures. The expected failures concern GAN output scale and non-deterministic training.

From the project root, install the project dependencies and run:

```bash
.venv/bin/python -m pytest -q tests
```
