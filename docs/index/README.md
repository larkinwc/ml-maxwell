# PEP 503 wheel index (GitHub Pages)

This directory backs a self-hosted simple index so heavy sm_50 wheels (published to
GitHub Releases) are pip-installable:

```bash
pip install vllm-maxwell torch \
  --extra-index-url https://larkinwc.github.io/ml-maxwell/simple/
```

## How it works

- Wheels live on **GitHub Releases** (not in git — too large).
- A CI step (see `build-torch-sm50.yml`, extend as needed) generates a PEP 503
  `simple/<pkg>/index.html` whose links point at the Release asset URLs.
- GitHub Pages serves `simple/` from this repo.

## To enable

1. Repo Settings → Pages → deploy from branch (or a `gh-pages` action).
2. Add a `publish-index` workflow that, on each Release, regenerates
   `simple/torch/index.html` and `simple/vllm-maxwell/index.html`.

Layout once populated:

```
simple/
├── index.html              # lists packages
├── torch/index.html        # links to torch-2.9.1-cp312-...whl on Releases
└── vllm-maxwell/index.html
```
