@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM run_l4s_indices.bat
REM Runs all 5 models on Landslide4Sense with NDVI/NDWI/NBR spectral indices
REM as the SAR modality slot (proposed fix for evidential fusion collapse)
REM
REM Preprocessing already done. Just run this file.
REM ═══════════════════════════════════════════════════════════════════════════

call conda activate haefnet

echo ============================================================
echo L4S Paper 2 — Spectral Indices Experiment
echo SAR slot: NDVI / NDWI / NBR
echo Config: configs/l4s_config_indices.yml
echo ============================================================

echo.
echo [1/5] CM1 — Concat U-Net...
python train.py --config configs/l4s_config_indices.yml --model cm1
if %ERRORLEVEL% NEQ 0 ( echo ERROR: CM1 failed & pause & exit /b 1 )

echo.
echo [2/5] CM2 — Attention U-Net...
python train.py --config configs/l4s_config_indices.yml --model cm2
if %ERRORLEVEL% NEQ 0 ( echo ERROR: CM2 failed & pause & exit /b 1 )

echo.
echo [3/5] M1 — HAEFNet (KEY TEST: will evidential fusion converge?)
python train.py --config configs/l4s_config_indices.yml --model m1
if %ERRORLEVEL% NEQ 0 ( echo ERROR: M1 failed & pause & exit /b 1 )

echo.
echo [4/5] M2 — HAEFNet + RO1 ablation...
python train.py --config configs/l4s_config_indices.yml --model m2
if %ERRORLEVEL% NEQ 0 ( echo ERROR: M2 failed & pause & exit /b 1 )

echo.
echo [5/5] M3 — Proposed model (RO1 + RO2)...
python train.py --config configs/l4s_config_indices.yml --model m3
if %ERRORLEVEL% NEQ 0 ( echo ERROR: M3 failed & pause & exit /b 1 )

echo.
echo [6/6] Evaluating all models...
python evaluate_all.py --config configs/l4s_config_indices.yml
echo Evaluation complete.

echo.
echo ============================================================
echo ALL RUNS COMPLETE
echo Results: checkpoints/l4s_indices/
echo ============================================================
pause
