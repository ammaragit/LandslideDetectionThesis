@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM run_l4s.bat — Landslide4Sense Cross-Dataset Validation
REM Runs all 5 models on Landslide4Sense for cross-dataset comparison
REM
REM STEP 1 — DEBUG (verify pipeline with 150 samples):
REM   In l4s_config.yml: max_samples: 150, num_epochs: 10, save_interval: 2
REM   Then: run_l4s.bat
REM
REM STEP 2 — FULL RUN:
REM   In l4s_config.yml: max_samples: null, num_epochs: 50, save_interval: 5
REM   Then: run_l4s.bat
REM ═══════════════════════════════════════════════════════════════════════════

echo ============================================================
echo Landslide4Sense Cross-Dataset Validation
echo ============================================================

call conda activate haefnet

REM Step 0: Preprocess h5 files to npz
echo.
echo [0] Preprocessing Landslide4Sense h5 files...
python preprocess_l4s.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in preprocessing & pause & exit /b 1 )

REM Step 0b: Generate splits
echo.
echo [0b] Generating train/val/test splits...
python l4s_splits.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in split generation & pause & exit /b 1 )

REM Step 1: CM1
echo.
echo [1/5] Training CM1 on Landslide4Sense...
python train.py --config configs/l4s_config.yml --model cm1
if %ERRORLEVEL% NEQ 0 ( echo ERROR: CM1 failed & pause & exit /b 1 )

REM Step 2: CM2
echo.
echo [2/5] Training CM2 on Landslide4Sense...
python train.py --config configs/l4s_config.yml --model cm2
if %ERRORLEVEL% NEQ 0 ( echo ERROR: CM2 failed & pause & exit /b 1 )

REM Step 3: M1
echo.
echo [3/5] Training M1 on Landslide4Sense...
python train.py --config configs/l4s_config.yml --model m1
if %ERRORLEVEL% NEQ 0 ( echo ERROR: M1 failed & pause & exit /b 1 )

REM Step 4: M2
echo.
echo [4/5] Training M2 on Landslide4Sense...
python train.py --config configs/l4s_config.yml --model m2
if %ERRORLEVEL% NEQ 0 ( echo ERROR: M2 failed & pause & exit /b 1 )

REM Step 5: M3
echo.
echo [5/5] Training M3 on Landslide4Sense...
python train.py --config configs/l4s_config.yml --model m3
if %ERRORLEVEL% NEQ 0 ( echo ERROR: M3 failed & pause & exit /b 1 )

REM Step 6: Evaluate all
echo.
echo [6/6] Generating L4S evaluation figures...
python evaluate_all.py --config configs/l4s_config.yml
echo Evaluation complete.

echo.
echo ============================================================
echo ALL L4S RUNS COMPLETE
echo Results in: checkpoints/l4s/
echo Figures in: figures/
echo ============================================================
pause
