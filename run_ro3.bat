@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM run_ro3.bat — RO3 Cross-Regional Evaluation
REM Trains M1 and M3 on each held-out region (6 runs total)
REM Run after run_all.bat has completed
REM
REM Estimated time: 6 runs x 8-10 hrs = 48-60 hrs
REM Run across 2-3 nights
REM ═══════════════════════════════════════════════════════════════════════════

echo ============================================================
echo RO3 — Cross-Regional Evaluation
echo Models: M1 and M3
echo Held-out regions: chimanimani, hiroshima, indonesia
echo ============================================================

call conda activate haefnet

REM Step 1: Generate region splits (run once)
echo.
echo [0/6] Generating RO3 splits...
python ro3_splits.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Split generation failed.
    pause
    exit /b 1
)

REM ── Chimanimani ──────────────────────────────────────────────────────────────
echo.
echo [1/6] M1 on held-out chimanimani...
python train_ro3.py --config configs/exp_haefnet.yml --model m1 --held_out chimanimani
if %ERRORLEVEL% NEQ 0 ( echo ERROR in run 1. & pause & exit /b 1 )

echo.
echo [2/6] M3 on held-out chimanimani...
python train_ro3.py --config configs/exp_haefnet.yml --model m3 --held_out chimanimani
if %ERRORLEVEL% NEQ 0 ( echo ERROR in run 2. & pause & exit /b 1 )

REM ── Hiroshima ─────────────────────────────────────────────────────────────────
echo.
echo [3/6] M1 on held-out hiroshima...
python train_ro3.py --config configs/exp_haefnet.yml --model m1 --held_out hiroshima
if %ERRORLEVEL% NEQ 0 ( echo ERROR in run 3. & pause & exit /b 1 )

echo.
echo [4/6] M3 on held-out hiroshima...
python train_ro3.py --config configs/exp_haefnet.yml --model m3 --held_out hiroshima
if %ERRORLEVEL% NEQ 0 ( echo ERROR in run 4. & pause & exit /b 1 )

REM ── Indonesia ─────────────────────────────────────────────────────────────────
echo.
echo [5/6] M1 on held-out indonesia...
python train_ro3.py --config configs/exp_haefnet.yml --model m1 --held_out indonesia
if %ERRORLEVEL% NEQ 0 ( echo ERROR in run 5. & pause & exit /b 1 )

echo.
echo [6/6] M3 on held-out indonesia...
python train_ro3.py --config configs/exp_haefnet.yml --model m3 --held_out indonesia
if %ERRORLEVEL% NEQ 0 ( echo ERROR in run 6. & pause & exit /b 1 )

REM ── Final evaluation ──────────────────────────────────────────────────────────
echo.
echo [7/7] Generating RO3 results table and figures...
python evaluate_ro3.py
echo RO3 evaluation complete.

echo.
echo ============================================================
echo ALL RO3 RUNS COMPLETE
echo Results saved in:
echo   checkpoints/ro3/m1_holdout_chimanimani/test_results.txt
echo   checkpoints/ro3/m3_holdout_chimanimani/test_results.txt
echo   checkpoints/ro3/m1_holdout_hiroshima/test_results.txt
echo   checkpoints/ro3/m3_holdout_hiroshima/test_results.txt
echo   checkpoints/ro3/m1_holdout_indonesia/test_results.txt
echo   checkpoints/ro3/m3_holdout_indonesia/test_results.txt
echo   figures/ro3_table.txt
echo   figures/ro3_comparison.png
echo ============================================================
pause
