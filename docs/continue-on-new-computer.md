# Continue This Project On Another Computer

Yes, you can continue modifying and tuning this project on another computer if the repository and data are synchronized.

## What Must Be In The Repository

The repository should contain:

- the six G-code files under `Gcode/`
- analysis notes under `docs/`
- any future scripts, notebooks, or model files used for parsing, feature extraction, evaluation, and training

Avoid relying on absolute local paths such as:

```text
C:/Users/Administrator/Desktop/some-local-folder/Gcode
```

Inside project code and scripts, prefer repository-relative paths:

```text
Gcode/3_wave_whole_123678.mpf
```

## Workflow On This Computer

After changes are ready:

```powershell
git status
git add Gcode docs
git commit -m "Add G-code classification analysis notes"
git push origin main
```

Only add other files such as `README.md` or `styles.css` if they are intentionally part of this project.

## Setup On A New Computer

1. Install Git.

2. Clone the repository:

```powershell
git clone https://github.com/dgq34777/Gcode.git
cd Gcode
```

3. Check that the G-code files exist:

```powershell
Get-ChildItem Gcode
```

4. Install Python if you will run parsing or model experiments.

5. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

6. Install future dependencies once a `requirements.txt` exists:

```powershell
pip install -r requirements.txt
```

For the current exploratory scripts, useful packages are:

```powershell
pip install numpy scikit-learn
```

## Daily Sync Workflow

Before starting work on a new computer:

```powershell
git pull origin main
```

After finishing work:

```powershell
git status
git add <changed-files>
git commit -m "Describe the change"
git push origin main
```

On the other computer, run `git pull origin main` before continuing.

## Recommended Next Repository Additions

To make this project easier to continue anywhere, add:

- `scripts/parse_gcode.py`: parse MPF files into points and labels
- `scripts/evaluate_rules.py`: run the current rule-based classifier and print metrics
- `requirements.txt`: Python dependencies
- `data/` or `outputs/`: optional generated feature tables or experiment reports

Keep generated large files out of git unless they are needed for reproducibility.
