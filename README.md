# QuantumPath: hybrid quantum vision for cancer diagnostics

Hackathon Track 8 prototype. A classical CNN reads medical images; a small quantum layer
(QSVC kernel, VQC or QCNN) classifies compact features. Research use only, not a medical device.

## Run the benchmark
    pip install -r requirements.txt
    python quantum_diagnostics.py --n-train 600 --epochs 5

This writes `results.json` and `gradcam.png`. Put both next to `index.html`.

## Website (GitHub Pages)
1. Push these files to a GitHub repo: `index.html`, `quantum_diagnostics.py`, `requirements.txt`, `README.md`, `results.json`, `gradcam.png`.
2. Repo Settings > Pages > Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
3. Your site appears at `https://<your-username>.github.io/<repo-name>/`.

The dashboard loads `results.json` automatically on Pages. Locally, use the file picker.
