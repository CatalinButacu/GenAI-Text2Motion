# -*- coding: utf-8 -*-
"""Build the Colab training notebook for MotionSSM on HumanML3D.

v2 improvements:
- Drive data cache: copies motion_data.zip from Drive (~2 min) instead
  of re-downloading from HuggingFace (~60 min) on subsequent runs.
- --num-workers 0 to avoid Colab fork() issues.
- --resume latest so interrupted training can continue from last checkpoint.
"""
import json, uuid

OUT = "D:/Facultate/dissertation/scripts/cloud/colab/train_colab.ipynb"

cellsSrc = [

#  0. GPU check
r"""#  0. GPU check + logging setup
import subprocess, sys, os, pathlib, datetime

LOG_FILE = pathlib.Path('/content/train_log.txt')
LOG_FILE.write_text('')

def log(m):
    line = datetime.datetime.now().strftime('%H:%M:%S') + '  ' + m
    print(line, flush=True)
    with open(str(LOG_FILE), 'a') as f:
        f.write(line + '\n')

log('=== Cell 0: GPU check ===')
r = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                    '--format=csv,noheader'],
                   capture_output=True, text=True)
if r.returncode != 0:
    log('WARNING: nvidia-smi failed --no GPU?  ' + r.stderr[:120])
else:
    log('GPU: ' + r.stdout.strip())

import torch
log('PyTorch: ' + torch.__version__)
log('CUDA available: ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    log('GPU: ' + torch.cuda.get_device_name(0) +
        '  sm_' + str(cap[0]) + str(cap[1]) +
        '  ' + str(round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)) + ' GB')
else:
    log('FATAL: No GPU. Runtime -> Change runtime type -> T4 GPU')
    raise RuntimeError('No GPU available')

import numpy, tqdm, huggingface_hub
log('numpy ' + numpy.__version__ + '  tqdm ' + tqdm.__version__ +
    '  hf_hub ' + huggingface_hub.__version__)
log('Cell 0 DONE')""",

#  1. Mount Drive
r"""#  1. Mount Google Drive
from google.colab import drive
try:
    drive.mount('/content/drive')
except ValueError:
    drive.mount('/content/drive', force_remount=True)

# Two Drive directories we use:
#   DRIVE_DATA  - cached raw data (survives between runs, saves 60 min/run)
#   DRIVE_CKPT  - trained checkpoints output
DRIVE_DATA = pathlib.Path('/content/drive/MyDrive/dissertation_data/humanml3d')
DRIVE_CKPT = pathlib.Path('/content/drive/MyDrive/dissertation/checkpoints')
DRIVE_DATA.mkdir(parents=True, exist_ok=True)
DRIVE_CKPT.mkdir(parents=True, exist_ok=True)
log('Drive mounted.')
log('  data cache  : ' + str(DRIVE_DATA))
log('  checkpoints : ' + str(DRIVE_CKPT))
log('Cell 1 DONE')""",

#  2. Clone repo
r"""#  2. Clone dissertation repo
log('=== Cell 2: git clone ===')
REPO_URL = 'https://github.com/CatalinButacu/GenAI-Flickr.git'
REPO_DIR = '/content/repo'

if not os.path.exists(REPO_DIR):
    r = subprocess.run(['git', 'clone', '--depth=1', REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log('FATAL: ' + r.stderr[:400])
        raise RuntimeError('git clone failed')
    log('Cloned OK')
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=False)
    log('Pulled.')

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

required = [
    'scripts/training/train_motion_ssm.py',
    'src/modules/motion/training/trainer.py',
    'src/data/humanml3d_loader.py',
    'src/modules/motion/nn_models.py',
    'src/modules/motion/ssm/mamba.py',
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    log('FATAL missing files: ' + str(missing))
    raise RuntimeError('Missing: ' + str(missing))
log('All required files present. CWD: ' + os.getcwd())
log('Cell 2 DONE')""",

#  3. Data (Drive cache first, HuggingFace fallback)
r"""#  3. HumanML3D data --Drive cache then HuggingFace fallback
# Strategy: first run downloads from HuggingFace and saves zip to Drive.
#           Subsequent runs copy from Drive (~2 min vs ~60 min).
log('=== Cell 3: HumanML3D data ===')
import zipfile, shutil, time
from huggingface_hub import hf_hub_download

os.environ['HF_HOME'] = '/content/hf_cache'
HF_REPO  = 'lxxiao/272-dim-HumanML3D'
DATA_DIR = pathlib.Path('data/humanml3d')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def hf_get(fn):
    log('  dl from HuggingFace: ' + fn)
    return hf_hub_download(HF_REPO, fn, repo_type='dataset', resume_download=True)

def get_or_cache(fname, drive_dir):
    # Return local path: from Drive if cached, else download from HuggingFace + save to Drive.
    drive_path = drive_dir / fname
    if drive_path.exists():
        local = pathlib.Path('/content') / fname
        log('  copying from Drive: ' + fname + ' -> ' + str(local))
        t0 = time.time()
        shutil.copy(str(drive_path), str(local))
        log('  copied in ' + str(round(time.time() - t0)) + 's')
        return str(local)
    else:
        local = hf_get(fname)
        log('  saving to Drive cache: ' + fname)
        shutil.copy(local, str(drive_path))
        return local

#  texts
if not (DATA_DIR / 'texts').exists():
    zp = get_or_cache('texts.zip', DRIVE_DATA)
    tmp = DATA_DIR / '_texts_tmp'
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)

    texts_found = None
    if (tmp / 'texts').exists():
        texts_found = tmp / 'texts'
    else:
        for cand in tmp.iterdir():
            if cand.is_dir() and (cand / 'texts').exists():
                texts_found = cand / 'texts'
                break
        if texts_found is None:
            txt_files = list(tmp.glob('*.txt'))
            if txt_files:
                texts_found = tmp
            else:
                log('  WARNING: no .txt files found in texts.zip!')

    if texts_found is not None:
        dest = DATA_DIR / 'texts'
        if texts_found == tmp:
            dest.mkdir(exist_ok=True)
            for f in tmp.glob('*.txt'):
                shutil.move(str(f), str(dest / f.name))
        else:
            shutil.move(str(texts_found), str(dest))
    shutil.rmtree(str(tmp), ignore_errors=True)
    n_texts = len(list((DATA_DIR / 'texts').glob('*.txt'))) if (DATA_DIR / 'texts').exists() else 0
    log('  texts: ' + str(n_texts) + ' files')
else:
    log('  texts: already present (' + str(len(list((DATA_DIR / 'texts').glob('*.txt')))) + ' files)')

#  splits
split_dir = DATA_DIR / 'split'
split_dir.mkdir(exist_ok=True)
for sp in ['train', 'val', 'test']:
    dest = split_dir / (sp + '.txt')
    if not dest.exists():
        drive_split = DRIVE_DATA / 'split' / (sp + '.txt')
        if drive_split.exists():
            shutil.copy(str(drive_split), str(dest))
        else:
            shutil.copy(hf_get('split/' + sp + '.txt'), dest)
            (DRIVE_DATA / 'split').mkdir(exist_ok=True)
            shutil.copy(str(dest), str(drive_split))
    ids = [l.strip() for l in dest.read_text().splitlines() if l.strip()]
    log('  split/' + sp + '.txt: ' + str(len(ids)) + ' IDs')

#  mean/std
ms = DATA_DIR / 'mean_std'
ms.mkdir(exist_ok=True)
(DRIVE_DATA / 'mean_std').mkdir(exist_ok=True)
for npy in ['Mean.npy', 'Std.npy']:
    if not (ms / npy).exists():
        drive_npy = DRIVE_DATA / 'mean_std' / npy
        if drive_npy.exists():
            shutil.copy(str(drive_npy), str(ms / npy))
        else:
            shutil.copy(hf_get('mean_std/' + npy), ms / npy)
            shutil.copy(str(ms / npy), str(drive_npy))
log('  mean/std OK')

log('Cell 3 DONE')""",

#  4. Verify data
r"""#  4. Verify dataset before training
log('=== Cell 4: data verification ===')
texts_dir = DATA_DIR / 'texts'
split_dir  = DATA_DIR / 'split'

n_texts = len(list(texts_dir.glob('*.txt'))) if texts_dir.exists() else 0
log('texts/ : ' + str(n_texts) + ' annotation files')

if n_texts > 0:
    sample = sorted(texts_dir.glob('*.txt'))[:2]
    for p in sample:
        log('  ' + p.name + ' -> ' + p.read_text()[:80].replace('\n', ' | '))

train_ids = []
if (split_dir / 'train.txt').exists():
    train_ids = [l.strip() for l in (split_dir / 'train.txt').read_text().splitlines() if l.strip()]
    log('split/train.txt: ' + str(len(train_ids)) + ' IDs  first 5: ' + str(train_ids[:5]))

# Only need texts to be present; AMASS clips are on the training machine.
if n_texts == 0:
    log('FATAL: texts/ empty -- run download cell or check Drive cache.')
    raise RuntimeError('No annotation files.')
if not train_ids:
    log('FATAL: split/train.txt missing -- re-run download cell.')
    raise RuntimeError('Missing split.')

has_texts = sum(1 for cid in train_ids[:500] if (texts_dir / (cid + '.txt')).exists())
log('Text coverage (500 IDs): ' + str(has_texts) + '/500')
if has_texts == 0:
    log('FATAL: 0 annotation matches -- clip ID format mismatch?')
    raise RuntimeError('Text cross-match failed.')

log('Data OK')
log('Cell 4 DONE')""",

#  5. WandB (offline by default; local logs copied to Drive in Cell 7)
r"""#  5. WandB (offline by default)
log('=== Cell 5: WandB ===')
# Offline mode writes run data under each run's wandb/ folder locally. No account
# or network needed. Local logs are copied to Drive in Cell 7.
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'
log('WandB offline.')
log('Cell 5 DONE')""",

#  6a. Train RVQ tokenizer (prerequisite for MotionSSM)
r"""#  6a. Train RVQ tokenizer
# MotionSSM requires a pre-trained RVQ tokenizer. Train it first, then MotionSSM
# reuses the frozen tokenizer in Cell 6b.
log('=== Cell 6a: RVQ tokenizer ===')
RVQ_CKPT_DIR = 'checkpoints/rvq_tokenizer'
rvq_best = pathlib.Path(RVQ_CKPT_DIR) / 'best_model.pt'
drive_rvq = DRIVE_CKPT / 'rvq_tokenizer' / 'best_model.pt'

if rvq_best.exists():
    log('  RVQ already trained locally: ' + str(rvq_best))
elif drive_rvq.exists():
    log('  Copying RVQ checkpoint from Drive: ' + str(drive_rvq))
    rvq_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(drive_rvq), str(rvq_best))
else:
    cmd_rvq = [
        sys.executable, 'scripts/training/train_rvq_tokenizer.py',
        '--epochs',          '50',
        '--batch-size',      '64',
        '--device',          'cuda',
        '--num-workers',     '0',
    ]
    log('CMD: ' + ' '.join(cmd_rvq))
    r = subprocess.run(cmd_rvq)
    if r.returncode != 0:
        log('FATAL: RVQ training failed')
        raise RuntimeError('RVQ training exited ' + str(r.returncode))
    drive_rvq.parent.mkdir(parents=True, exist_ok=True)
    if rvq_best.exists():
        shutil.copy(str(rvq_best), str(drive_rvq))
        log('  RVQ checkpoint cached to Drive: ' + str(drive_rvq))
log('Cell 6a DONE')""",

#  6b. Train MotionSSM
r"""#  6b. Train MotionSSM (uses frozen RVQ tokenizer from Cell 6a)
log('=== Cell 6b: MotionSSM training ===')
CKPT_DIR = 'checkpoints/motion_ssm_hml3d'   # <- change per account

# Resume from latest checkpoint if one exists (survives session restarts)
resume_args = []
import glob as _glob
existing = sorted(_glob.glob(CKPT_DIR + '/*/checkpoint_epoch*.pt'))
if existing:
    log('  Resuming from: ' + existing[-1])
    resume_args = ['--resume', 'latest']
else:
    log('  Starting fresh.')

cmd = [
    sys.executable, 'scripts/training/train_motion_ssm.py',
    '--data-source',     'humanml3d',
    '--epochs',          '30',
    '--batch-size',      '64',
    '--lr',              '1e-4',
    '--device',          'cuda',
    '--d-model',         '256',
    '--n-layers',        '4',
    '--checkpoint-dir',  CKPT_DIR,
    '--num-workers',     '0',       # 0 avoids Colab fork() issues
    '--seed',            '42',
    '--use-sbert',
    '--bidirectional',
    '--use-film',
    '--rvq-checkpoint',  str(rvq_best),
] + resume_args

log('CMD: ' + ' '.join(cmd))
result = subprocess.run(cmd)
log('Train exit code: ' + str(result.returncode))
if result.returncode != 0:
    log('FATAL: training failed --check cell output above for the actual error')
    raise RuntimeError('Training exited ' + str(result.returncode))
log('Cell 6b DONE')""",

#  7. Save to Drive
r"""#  7. Copy checkpoints + wandb logs to Google Drive
log('=== Cell 7: save to Drive ===')
import shutil as _sh

CKPT_DIR = 'checkpoints/motion_ssm_hml3d'   # <- must match Cell 6b
ckpt_src = pathlib.Path(CKPT_DIR)
dest_root = DRIVE_CKPT / pathlib.Path(CKPT_DIR).name
dest_root.mkdir(parents=True, exist_ok=True)

# Per-run subdirectories -- each training invocation writes to checkpoints/<base>/<run_id>/
n_copied = 0
for run_dir in sorted(ckpt_src.iterdir()) if ckpt_src.exists() else []:
    if not run_dir.is_dir():
        continue
    dest = dest_root / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for f in run_dir.rglob('*'):
        if f.is_file():
            rel = f.relative_to(run_dir)
            tgt = dest / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            _sh.copy(f, tgt)
            n_copied += 1
    log('Synced run dir -> ' + str(dest))

if n_copied == 0:
    log('WARNING: no files found under ' + str(ckpt_src))
_sh.copy(str(LOG_FILE), dest_root / '_colab_log.txt')
log('All saved to: ' + str(dest_root))
log('=== ALL DONE ===')""",
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "name": "MotionSSM Training v2 - HumanML3D",
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": []
}

for i, src in enumerate(cellsSrc):
    nb["cells"].append({
        "cell_type": "code",
        "id": "col" + str(i) + "-" + uuid.uuid4().hex[:6],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src
    })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Written", len(nb["cells"]), "cells to", OUT)
