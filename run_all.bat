@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM run_all.bat — trains all 5 models sequentially then evaluates
REM
REM STEP 1 — DEBUG RUN (verify pipeline with 150 samples):
REM   Make sure in exp_haefnet.yml:
REM     max_samples: 150
REM     num_epochs: 10
REM     save_interval: 2
REM   Then run: run_all.bat
REM
REM STEP 2 — FULL RUN (overnight, all data):
REM   Change in exp_haefnet.yml:
REM     max_samples: null
REM     num_epochs: 50
REM     save_interval: 5
REM   Then run: run_all.bat
REM ═══════════════════════════════════════════════════════════════════════════

echo ============================================================
echo Starting full training pipeline
echo Models: CM1 ^> CM2 ^> M1 ^> M2 ^> M3
echo Config: configs/exp_haefnet.yml
echo ============================================================

REM Activate conda environment
call conda activate haefnet

REM CM1 — Simple Concat U-Net
echo.
echo [1/5] Training CM1 — Simple Concat U-Net...
python train.py --config configs/exp_haefnet.yml --model cm1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CM1 training failed. Stopping.
    pause
    exit /b 1
)
echo CM1 complete.

REM CM2 — Attention U-Net Late Fusion
echo.
echo [2/5] Training CM2 — Attention U-Net Late Fusion...
python train.py --config configs/exp_haefnet.yml --model cm2
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CM2 training failed. Stopping.
    pause
    exit /b 1
)
echo CM2 complete.

REM M1 — HAEFNet baseline
echo.
echo [3/5] Training M1 — HAEFNet baseline...
python train.py --config configs/exp_haefnet.yml --model m1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: M1 training failed. Stopping.
    pause
    exit /b 1
)
echo M1 complete.

REM M2 — HAEFNet + RO1
echo.
echo [4/5] Training M2 — HAEFNet + RO1...
python train.py --config configs/exp_haefnet.yml --model m2
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: M2 training failed. Stopping.
    pause
    exit /b 1
)
echo M2 complete.

REM M3 — Full proposed model
echo.
echo [5/5] Training M3 — Full Proposed Model (RO1 + RO2)...
python train.py --config configs/exp_haefnet.yml --model m3
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: M3 training failed. Stopping.
    pause
    exit /b 1
)
echo M3 complete.

REM Final evaluation and figure generation
echo.
echo [6/6] Running final evaluation and generating figures...
python evaluate_all.py --config configs/exp_haefnet.yml
echo Evaluation complete.

echo.
echo ============================================================
echo ALL DONE. Check checkpoints/ for results.
echo Results saved in:
echo   checkpoints/cm1_concat_unet/test_results.txt
echo   checkpoints/cm2_attention_unet/test_results.txt
echo   checkpoints/m1_haefnet_baseline/test_results.txt
echo   checkpoints/m2_haefnet_ro1/test_results.txt
echo   checkpoints/m3_proposed_ro1_ro2/test_results.txt
echo   figures/ for all plots
echo ============================================================
pause
