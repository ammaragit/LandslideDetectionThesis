@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM run_l4s_paper2.bat — Paper 2 Experiments: L4S with modality substitution
REM
REM Priority order:
REM   Night 1: Preprocess all 3 modes + run INDICES group (most likely to fix M1/M3)
REM   Night 2: Run DEM FEATURES group
REM   Night 3: Run RAW group (re-confirm baseline non-convergence with cleaner logs)
REM
REM Run:  run_l4s_paper2.bat indices     ← Night 1 (START HERE)
REM       run_l4s_paper2.bat dem         ← Night 2
REM       run_l4s_paper2.bat raw         ← Night 3
REM ═══════════════════════════════════════════════════════════════════════════

call conda activate haefnet

set MODE=%1
if "%MODE%"=="" set MODE=indices

echo ============================================================
echo Paper 2 Experiments: mode = %MODE%
echo ============================================================

REM ── Step 0: Preprocess ───────────────────────────────────────────────────────
echo.
echo [0] Preprocessing with mode=%MODE%...
python preprocess_l4s_v2.py --mode %MODE%
if %ERRORLEVEL% NEQ 0 ( echo ERROR in preprocessing & pause & exit /b 1 )

REM ── Step 0b: Generate splits (uses same split file for all modes) ─────────────
echo.
echo [0b] Generating splits...
python l4s_splits.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in splits & pause & exit /b 1 )

if "%MODE%"=="indices" goto RUN_INDICES
if "%MODE%"=="dem" goto RUN_DEM
if "%MODE%"=="raw" goto RUN_RAW
echo Unknown mode: %MODE%
pause & exit /b 1

REM ── Indices group ─────────────────────────────────────────────────────────────
:RUN_INDICES
echo.
echo [1/4] CM1 — Spectral indices...
python train.py --config configs/l4s_config_v2.yml --model cm1_idx
if %ERRORLEVEL% NEQ 0 ( echo ERROR: cm1_idx & pause & exit /b 1 )

echo.
echo [2/4] CM2 — Spectral indices...
python train.py --config configs/l4s_config_v2.yml --model cm2_idx
if %ERRORLEVEL% NEQ 0 ( echo ERROR: cm2_idx & pause & exit /b 1 )

echo.
echo [3/4] M1 — Spectral indices (KEY TEST: will evidential fusion converge?)
python train.py --config configs/l4s_config_v2.yml --model m1_idx
if %ERRORLEVEL% NEQ 0 ( echo ERROR: m1_idx & pause & exit /b 1 )

echo.
echo [4/4] M3 — Spectral indices...
python train.py --config configs/l4s_config_v2.yml --model m3_idx
if %ERRORLEVEL% NEQ 0 ( echo ERROR: m3_idx & pause & exit /b 1 )

goto DONE

REM ── DEM features group ────────────────────────────────────────────────────────
:RUN_DEM
echo.
echo [1/3] CM1 — DEM features...
python train.py --config configs/l4s_config_v2.yml --model cm1_dem
if %ERRORLEVEL% NEQ 0 ( echo ERROR: cm1_dem & pause & exit /b 1 )

echo.
echo [2/3] M1 — DEM features...
python train.py --config configs/l4s_config_v2.yml --model m1_dem
if %ERRORLEVEL% NEQ 0 ( echo ERROR: m1_dem & pause & exit /b 1 )

echo.
echo [3/3] M3 — DEM features...
python train.py --config configs/l4s_config_v2.yml --model m3_dem
if %ERRORLEVEL% NEQ 0 ( echo ERROR: m3_dem & pause & exit /b 1 )

goto DONE

REM ── Raw group (baseline re-run) ────────────────────────────────────────────────
:RUN_RAW
echo.
echo [1/4] CM1 — Raw bands (baseline)...
python train.py --config configs/l4s_config_v2.yml --model cm1_raw
if %ERRORLEVEL% NEQ 0 ( echo ERROR: cm1_raw & pause & exit /b 1 )

echo.
echo [2/4] CM2 — Raw bands...
python train.py --config configs/l4s_config_v2.yml --model cm2_raw
if %ERRORLEVEL% NEQ 0 ( echo ERROR: cm2_raw & pause & exit /b 1 )

echo.
echo [3/4] M1 — Raw bands (expected to collapse)...
python train.py --config configs/l4s_config_v2.yml --model m1_raw
if %ERRORLEVEL% NEQ 0 ( echo ERROR: m1_raw & pause & exit /b 1 )

echo.
echo [4/4] M3 — Raw bands (expected to collapse)...
python train.py --config configs/l4s_config_v2.yml --model m3_raw
if %ERRORLEVEL% NEQ 0 ( echo ERROR: m3_raw & pause & exit /b 1 )

goto DONE

:DONE
echo.
echo ============================================================
echo MODE [%MODE%] COMPLETE
echo Results in: checkpoints/l4s_%MODE%/
echo ============================================================
pause
