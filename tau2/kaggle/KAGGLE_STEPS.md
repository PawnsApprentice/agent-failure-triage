# Scoring Laya on a Kaggle GPU

You need two files from this folder:
- `laya-tau2.zip`: the data plus the project's scoring code.
- `laya_tau2.ipynb`: the notebook that runs it.

Kaggle's free GPU runs the job in roughly 15-30 minutes. It took hours on the laptop.

## One-time setup

1. Create a free account at https://www.kaggle.com (Register, top right).
2. Verify your phone number: click your profile picture (top right), then **Settings**, then
   **Phone Verification**.
   - Kaggle won't give you a GPU or internet access until this is done.
   - The notebook needs internet to install `laya` and download the model.

## 1. Upload the data as a Kaggle Dataset

1. Click **+ Create** (top left), then **New Dataset**.
2. Drag `laya-tau2.zip` into the upload box and wait for it to finish (2.5 MB).
3. Set the title to `laya-tau2`, leave visibility **Private**, and click **Create**.

Kaggle normally unpacks the zip by itself. If it doesn't, the notebook unpacks it.

## 2. Create the notebook

1. Click **+ Create**, then **New Notebook**.
2. In the notebook's top menu, choose **File**, then **Import Notebook**, and upload `laya_tau2.ipynb`.
3. In the right-hand panel, find **Input** and click **+ Add Input**.
   - Choose the **Datasets** tab, then **Your Datasets**.
   - Click the **+** next to `laya-tau2`.
4. Still in the right-hand panel, open **Session options** (on some layouts this is the
   **Settings** menu):
   - **Accelerator**: GPU T4 x2.
   - **Internet**: On.

## 3. Run it

Run it as a committed version rather than live in your browser. That way it keeps running on
Kaggle's servers even if your internet drops.

1. Click **Save Version** (top right).
2. Choose **Save & Run All (Commit)** and click **Save**.
3. Click the version number (or the bell icon) to watch progress. The log shows:
   - `running on cuda (Tesla T4)`. If it says `NONE` or `cpu`, the accelerator setting didn't
     apply: stop, fix it, and save again.
   - `Step 0 on this device: accuracy 0.7xx`. **If this step fails, the run stops by design.** Send
     me the log.
   - Lines for W1, W2 and W3 on calibration, then the chosen wording, then the test scoring.
   - Finally, `wrote /kaggle/working/laya_predictions.csv`.

## 4. Download the results

1. Open the finished version and go to its **Output** tab.
2. Click **Download** (or **Download All**). You get a zip.
3. Unzip it into `tau2/kaggle/output/` in this repo, so that
   `tau2/kaggle/output/laya_predictions.csv` exists.

## 5. Back on the laptop

From the repo root:

```
.venv/bin/python tau2/experiment.py import-kaggle
.venv/bin/python tau2/experiment.py report
```

`report` prints the full table for both TF-IDF variants and Laya (shipped and refit
temperature): AUROC, recall at a 10% review budget, ECE, Brier and latency, all with 95% CIs,
overall and per domain. It also prints the GPU Step 0 check and the CPU-vs-GPU agreement.
Laya's latency in that table is the Kaggle T4; TF-IDF's is this laptop's CPU.

## If something goes wrong

- **`pip install` or the model download fails:** Internet is off, or the phone isn't verified.
- **"laya-tau2 dataset not found":** the dataset wasn't attached (step 2.3).
- **Run is very slow (minutes per item):** no GPU. Check the accelerator setting.
- **Weekly GPU quota:** Kaggle gives about 30 GPU-hours a week; this run uses well under one.

## Rebuilding the zip

Only needed if the code or split changes: `.venv/bin/python tau2/kaggle/build_package.py`.
