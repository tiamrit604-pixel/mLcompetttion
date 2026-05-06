# Bolted-Flange Looseness Detection — Streamlit Application

**Author:** Amrit Tiwari · Graduate Student · Mechanical Engineering · Cullen College of Engineering · University of Houston

A web application that classifies bolted-flange torque (0 / 25 / 50 ft-lb) from a single percussion-induced acoustic recording. Built for the **Spring 2026 Machine Learning Final Project Competition**.

---

## 1. What This Application Does

The app wraps the entire ML pipeline from the original Jupyter notebook into a polished, three-section web interface:

| Section | Purpose |
|---|---|
| ① **Train Your Own Model** | Upload labelled audio, reproduce the full pipeline (feature extraction → WFAN → PCA → 9 models → leave-one-flange-out validation), inspect every confusion matrix, and download trained models. |
| ② **Competition Prediction** | Use the pre-trained leave-one-flange-out bundles to predict torque on a new recording. Supports both audio file upload and live browser microphone. Per-flange accuracy-weighted ensemble for robust voting. |
| ③ **Signal Processing & Analysis** | Side-by-side feature plots (waveform, PSD, MFCC, STFT, log-Mel, envelope) across the 0/25/50 ft-lb classes — the visual story of *why* the classifier works. |

---

## 2. Folder Structure

After unzipping, your project folder should look exactly like this:

```
flange_app/
│
├── app.py                              ← Run this with: streamlit run app.py
├── build_models.py                     ← Run ONCE before launching the app
├── requirements.txt                    ← pip install -r requirements.txt
├── packages.txt                        ← For Streamlit Community Cloud
├── README.md                           ← This file
│
├── pages/                              ← Streamlit auto-discovers these
│   ├── 1_Train_Your_Own_Model.py
│   ├── 2_Competition_Prediction.py
│   └── 3_Signal_Analysis.py
│
├── utils/                              ← All the ML pipeline code
│   ├── constants.py                    ← Single source of truth for constants
│   ├── audio_io.py                     ← .m4a loading + peak segmentation
│   ├── features.py                     ← 221-d flat / CNN image / LSTM seq
│   ├── wfan.py                         ← Within-Flange-Area Normalisation
│   ├── models.py                       ← BPNN / CNN / BiLSTM definitions
│   ├── pipeline.py                     ← Full pipeline orchestration
│   ├── predict.py                      ← Ensemble prediction logic
│   ├── persistence.py                  ← Save/load trained bundles
│   └── plots.py                        ← All matplotlib / seaborn helpers
│
├── artifacts/                          ← Created by build_models.py
│   ├── ind_bundles.pkl                 ← Per-flange classical models + scalers + PCA
│   ├── BPNN_F1.pth ... BiLSTM_F4.pth   ← 12 deep model state_dicts
│   ├── wfan_means_flat.pkl             ← Competition WFAN means (flat features)
│   ├── wfan_means_2d.pkl               ← Competition WFAN means (CNN images)
│   ├── wfan_means_seq.pkl              ← Competition WFAN means (LSTM sequences)
│   ├── flange_top_models.pkl           ← Top-N model per flange
│   ├── flange_weights.pkl              ← Accuracy-weighted ensemble weights
│   ├── fold_results.pkl                ← LOFO accuracies (for the UI tables)
│   ├── dep_results.pkl                 ← Dependent-test accuracies
│   ├── bpnn_input_dims.pkl             ← Per-fold PCA output dimension
│   └── metadata.pkl                    ← Misc metadata shown in the sidebar
│
├── assets/                             ← Drop optional images here
│   ├── uh_logo.png                     ← University of Houston logo (optional)
│   ├── flange_setup.png                ← Test rig photo (optional)
│   └── hammer_recording.png            ← Data-acquisition photo (optional)
│
└── sample_data/                        ← Optional: drop sample .m4a files here
                                          for demos (not auto-loaded)
```

---

## 3. Prerequisites

### 3.1 Python
Python **3.10 – 3.12** is recommended. Check yours with `python --version`.

### 3.2 ffmpeg (system dependency)
Decoding `.m4a` files requires **ffmpeg** to be installed at the operating-system level. This is *not* a pip package — install it via your OS's package manager:

#### Windows
1. Download a static build from <https://www.gyan.dev/ffmpeg/builds/> (pick *“release essentials”*).
2. Unzip to e.g. `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your PATH (System Properties → Environment Variables → Path → Edit → New).
4. Open a fresh terminal and verify: `ffmpeg -version`.

Alternative (easier): if you have *Chocolatey*: `choco install ffmpeg`.

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Debian/Ubuntu)
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
```

---

## 4. Installation

```bash
# 1. Open a terminal in the flange_app/ folder
cd flange_app

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### Important note on PyTorch
The `requirements.txt` installs the **CPU build** of PyTorch, which is fine for prediction and even for training on small datasets. If you have an NVIDIA GPU and want training to run in a few seconds instead of a few minutes, follow the official instructions at <https://pytorch.org/get-started/locally/> *before* the `pip install -r requirements.txt` step.

### Important note on Streamlit version
Live browser recording (Section 2) requires **Streamlit ≥ 1.31** for the native `st.audio_input` widget. If you have an older version pinned, either:
- Run `pip install --upgrade streamlit` to get the current release, or
- Let the app fall back automatically to `streamlit-mic-recorder` (already in `requirements.txt`).

If neither is available the live-recording panel will tell you exactly which command to run, and the file-upload mode will keep working regardless.

---

## 5. Quick-Start (with your existing data)

```bash
# Step 1 — train all leave-one-flange-out models ONCE
#   This takes ~5–20 min on CPU, ~1–3 min on GPU.
#   Replace the path below with the folder containing your 48 labelled .m4a files.
python build_models.py --data /path/to/your/iphone_recordings

# Step 2 — launch the app
streamlit run app.py
```

A browser tab will open at <http://localhost:8501>.

### File-naming convention (for both training and analysis)
Every input file must be named `<torque>ftlb<Fx><Ay>.<ext>`:

```
0ftlbF1A1.m4a       25ftlbF1A1.m4a       50ftlbF1A1.m4a
0ftlbF1A2.m4a       25ftlbF1A2.m4a       50ftlbF1A2.m4a
...
0ftlbF4A4.m4a       25ftlbF4A4.m4a       50ftlbF4A4.m4a
```
(48 files total: 3 torques × 4 flanges × 4 areas.)

---

## 6. The Three Sections — Detailed Usage

### Section ①: Train Your Own Model

This page reproduces the entire notebook pipeline from inside the browser:

1. Adjust epochs / patience / top-N if needed (defaults are sensible).
2. Toggle "Include deep-learning models" on or off (off = classical only, ~5x faster).
3. Click *Browse files* and multi-select your labelled audio files.
4. Click **🚀 Run Full Pipeline**.
5. Watch the four phases run:
   - Phase 1: load + segment + extract features
   - Phase 2: dependent test (70/30 stratified split — 9 models)
   - Phase 3: independent test (leave-one-flange-out — 9 models × 4 folds)
   - Phase 4: WFAN means + per-flange weights
6. The results dashboard appears with four tabs:
   - **Dependent Test (70/30)** — bar chart + binary-mode bar chart + numeric table
   - **Independent Test (LOFO)** — grouped bar chart per flange + per-flange drill-down
   - **Confusion Matrices** — selectable fold/model + binary toggle
   - **Save / Download** — save artifacts to `artifacts/` (used by Section ②) or download as a zip

### Section ②: Competition Prediction

This is the page you'll actually demo. It loads the pre-trained `artifacts/` bundle and supports two input modes:

**A. Upload audio file (recommended)**
1. Pick the **Flange ID** (F1–F4) and **Striking Area** (A1–A4) — required for WFAN.
2. Upload an `.m4a` (iPhone) or `.wav` recording.
3. Click **🔍 Run prediction**.
4. See: headline torque card, accuracy-weighted probability bar, per-model votes, waveform with peaks, and the first 4 strike segments.

**B. Live browser recording**
1. Pick the flange and area.
2. Click the microphone button, hit the flange a few times, click stop.
3. Click **🔍 Run prediction**.
4. Same output as upload mode.

⚠️ **Browser microphone disclaimer**: the training data was recorded with an iPhone at native 48 kHz. Browser microphones use whatever the operating system gives them — different sample rates, different gain, different frequency response. For maximum accuracy at the actual competition, **upload an iPhone `.m4a` rather than rely on live browser recording**.

**Batch prediction**: the bottom expander accepts many files at once (named like `F1A1.m4a`, `F2A2.m4a`, …). It groups by flange, runs the pipeline on each, and prints one row per flange — exactly the format the competition asks for.

### Section ③: Signal Processing & Analysis

1. Upload labelled recordings (any number, any subset of torques).
2. The app shows a per-class hit count.
3. Pick which feature views to render (multi-select).
4. Adjust the *Samples per class* slider.
5. Each chosen view renders as a one-row-per-sample, one-column-per-class grid, with a "what to look for" caption explaining the physics.

This is the page that makes the science visible to a poster visitor.

---

## 7. Deploying to Streamlit Community Cloud (backup)

The local app is the primary demo. The cloud copy is a backup in case projector / Wi-Fi misbehaves.

### Steps:
1. **Create a new GitHub repository** (public or private) and push the entire `flange_app/` folder to it. The artifacts folder *can* be pushed (it's reasonably small for this project), but you can also choose to upload it via the Streamlit Cloud secret-files mechanism.
2. Go to <https://share.streamlit.io> and click **New app**.
3. Pick the repo, branch, and the entry file (`app.py`).
4. **Advanced settings → Python version**: pick 3.11.
5. Click **Deploy**.

Streamlit Cloud automatically:
- Reads `requirements.txt` and `pip install`s everything.
- Reads `packages.txt` and installs `ffmpeg` and `libsndfile1` at the system level.

Your app will be live at `https://<your-username>-<repo>-app.streamlit.app`.

### Cloud-specific tips
- The free tier provides ~1 GB of RAM and 1 CPU. Loading the pre-trained artifacts is fine; running Section ① (full re-training) on cloud will be much slower than locally — don't rely on that.
- If the `artifacts/` folder is too large for git (>100 MB), use **Git LFS** or upload the trained models via the Streamlit secrets mechanism and download them in `app.py` on first launch.
- Add this to `.streamlit/config.toml` (optional polish):
  ```toml
  [theme]
  primaryColor = "#C8102E"
  backgroundColor = "#FFFFFF"
  secondaryBackgroundColor = "#F9FAFB"
  textColor = "#111827"
  font = "sans serif"
  ```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ffmpeg-not-found` error when uploading `.m4a` | ffmpeg not on PATH | See §3.2. Open a *new* terminal after install. |
| `No trained artifacts found` on Section ② | `build_models.py` never ran | Run `python build_models.py --data <folder>` once. |
| Section ② says "no peaks detected" | Recording too quiet / clipped | Reduce hammer hit, increase mic gain, or use closer microphone. |
| "Skipped X files (bad name or load error)" in Section ① | Filenames don't match convention | Rename to `<torque>ftlb<Fx><Ay>.m4a`. |
| Browser mic produces strange results | Sample-rate mismatch with training data | Use file upload of iPhone recordings instead — see §6 disclaimer. |
| Streamlit Cloud build fails on `librosa` | Missing system audio libs | Make sure `packages.txt` contains `libsndfile1`. |
| Training appears stuck | Deep learning models on CPU | This is expected — see progress messages. Or toggle "Include deep-learning models" off. |
| `RuntimeError: CUDA out of memory` | GPU too small | Reduce `BATCH_SIZE` in `utils/constants.py`, or set `DEVICE = torch.device('cpu')`. |

---

## 9. Citation

If this work informs your own research, please cite:

> Tiwari, Amrit. *Bolted-Flange Looseness Detection from Percussion-Induced Acoustic Signals*. University of Houston Spring 2026 Machine Learning Final Project Competition, 2026.

---

## 10. Acknowledgements

- **Midstream Integrity Services (MIS)** — industrial context and motivation
- **Smart Materials and Structures Laboratory (SMSL)** — test rig and acquisition equipment
- **Artificial Intelligence Lab for Monitoring and Inspection (AILMI)** — guidance and review
- **University of Houston, Cullen College of Engineering** — institutional support

---

*Built with ❤️ at the University of Houston, May 2026.*
