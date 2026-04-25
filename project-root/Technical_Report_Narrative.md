# Technical Report: Development Narrative

## Phishing URL Detection System — Challenges & Solutions

---

### 1. Data Loading: Bypassing the Hugging Face Script Block

The original implementation called `load_dataset("ealvaradob/phishing-dataset")` directly. On the first run, the `datasets` library raised a hard error:

```
Dataset scripts are no longer supported, but found phishing-dataset.py
```

Hugging Face deprecated dynamic dataset loading scripts in version 2.x+ for security reasons. The repository still contained a legacy `phishing-dataset.py` loader, which the updated library refused to execute. The naive workaround of passing `trust_remote_code=True` was also rejected, as support for that flag had been simultaneously removed.

**Solution:** We inspected the repository's file manifest using `huggingface_hub.list_repo_files()`, which revealed the raw data was available as standalone JSON files (`urls.json`, `combined_full.json`, etc.). By switching to the built-in JSON loader with a direct `hf://` path:

```python
load_dataset("json", data_files="hf://datasets/ealvaradob/phishing-dataset/urls.json")
```

...we bypassed the script entirely while still satisfying the assignment requirement to load from Hugging Face using the `datasets` library.

---

### 2. The 'Google Bug': Feature Distribution Mismatch

After the first successful training run, the demo flagged `https://www.google.com` as phishing — consistently and confidently. The prediction was technically correct given what the model had learned, but for entirely wrong reasons.

A statistical inspection of the processed dataset revealed the cause:

| Label | Mean `has_https` |
|---|---|
| Legitimate (0) | 0.06 |
| Phishing (1) | 0.30 |

In the `urls.json` dataset, legitimate URLs had been stored without any protocol prefix (e.g., `billsportsmaps.com/?p=1206`), while phishing URLs frequently retained their `http://` or occasionally `https://` prefix. As a result, the model learned that the presence of `https://` was a mild *phishing indicator*, the direct inverse of its real-world meaning.

A second issue compounded this: the mean URL length was statistically identical across both classes (~47 characters), but `https://www.google.com` is only 22 characters after normalization — placing it in a region of feature space the model associated with short, suspicious URLs.

**Solution:** A `normalize_url()` function was introduced in `url_features.py`. It records the `has_https` flag from the original URL *before* stripping the protocol prefix, then strips `http://` or `https://` before computing `url_length` and `dot_count`. This aligns inference-time feature computation with the format of the training data, making the distribution consistent.

After retraining with the normalized features, `https://www.google.com` was correctly classified as Legitimate.

---

### 3. Testing & Code Coverage: Achieving >80% Coverage on Windows

The rubric required a test suite with 60-70% code coverage. Initially, running `pytest --cov=src` resulted in 0% coverage and a warning:

```
CoverageWarning: Module src was never imported. (module-not-imported)
```

The root issue was twofold:
1. Windows path resolution was failing to locate the `src` module nested inside `project-root/` when run from the parent workspace.
2. Reaching >60% coverage on a machine learning pipeline is difficult without executing the entire slow training process or downloading gigabytes of datasets during the test suite.

**Solution:** 
First, we solved the module resolution by dynamically appending the `project-root` to `sys.path` within `test_basic.py`. Second, to achieve high coverage rapidly, we heavily utilized `unittest.mock.patch` and `tmp_path`:
*   We mocked the Hugging Face `load_dataset` network call to instantly test the success/failure branches of `data_loader.py` without downloading the 800,000-row dataset.
*   We created a 2-line dummy CSV file to test `process_dataset` in `url_features.py` instantly.
*   We mocked `sys.argv` and `joblib.load` to simulate a user running the CLI `demo.py` script.

By abstracting away the heavy I/O, we pushed our code coverage to an excellent **84%** in under 5 seconds of test execution time.
