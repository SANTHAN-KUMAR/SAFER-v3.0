#!/usr/bin/env python3
"""
SAFER v3.0 Complete Architecture Pipeline.

This master script runs the COMPLETE SAFER v3.0 architecture as proposed:

┌─────────────────────────────────────────────────────────────────────────┐
│                           SAFER v3.0 Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │    Mamba     │    │   LPV-SINDy  │    │   LSTM       │              │
│  │  Predictor   │    │   Physics    │    │  Baseline    │              │
│  │   (DAL E)    │    │   Monitor    │    │   (DAL C)    │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                            │
│                   ┌─────────────────┐                                   │
│                   │ Simplex Decision │                                   │
│                   │     Module       │                                   │
│                   └────────┬────────┘                                   │
│                            ▼                                             │
│                   ┌─────────────────┐                                   │
│                   │  Conformal UQ   │                                   │
│                   │  Alert Manager  │                                   │
│                   └─────────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Steps:
1. Train LSTM Baseline (DAL C) - if not already trained
2. Train LPV-SINDy Physics Monitor (DAL C) - if not already trained
3. Run Conformal Calibration
4. Run Full SAFER Pipeline with all components
5. Export ONNX model for deployment
6. Generate Comprehensive Report
7. Create Production Deployment Package

Usage:
    python scripts/complete_safer_pipeline.py --mamba_dir outputs/mamba_FD001_XXXXXX

"""

import sys
import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_complete_pipeline(
    mamba_dir: str,
    checkpoint_path: str = None,
    data_dir: str = "CMAPSSData",
    dataset: str = "FD001",
    train_baseline: bool = True,
    train_physics: bool = True,
    baseline_epochs: int = 30,
    skip_existing: bool = True,
):
    """Run the complete SAFER v3.0 architecture pipeline.
    
    Args:
        mamba_dir: Directory for outputs (will store baseline, physics, etc.)
        checkpoint_path: Explicit path to Mamba checkpoint (optional)
        data_dir: Path to C-MAPSS data
        dataset: Dataset name (FD001, FD002, etc.)
        train_baseline: Whether to train LSTM baseline if not found
        train_physics: Whether to train LPV-SINDy if not found
        baseline_epochs: Epochs for baseline training
        skip_existing: Skip steps if outputs already exist
    """
    start_time = time.time()
    
    mamba_dir = Path(mamba_dir)
    
    # If explicit checkpoint path provided, use it
    if checkpoint_path:
        mamba_checkpoint = Path(checkpoint_path)
        if not mamba_checkpoint.exists():
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
        print(f"Using specified checkpoint: {mamba_checkpoint}")
    else:
        # Find checkpoint - check multiple possible locations
        checkpoint_patterns = [
            mamba_dir / "best_model.pt",
            mamba_dir / "mamba_best.pt",
            mamba_dir / "checkpoint_best.pt",
            mamba_dir / "deployment_calibrated" / "model_checkpoint.pt",
            mamba_dir / "deployment" / "model_weights.pt",
            Path("/workspace/checkpoints/best_model.pt"),  # Docker container location
            Path("checkpoints/best_model.pt"),  # Relative path
            project_root / "checkpoints" / "best_model.pt",  # Project root
        ]
        
        mamba_checkpoint = None
        for candidate in checkpoint_patterns:
            if candidate.exists():
                mamba_checkpoint = candidate
                print(f"Found checkpoint at: {candidate}")
                break
        
        if mamba_checkpoint is None:
            # Look for any .pt file in mamba_dir
            pt_files = list(mamba_dir.glob("**/*.pt"))
            if pt_files:
                mamba_checkpoint = pt_files[0]
                print(f"Found checkpoint at: {mamba_checkpoint}")
            else:
                # Last resort: search in checkpoints directory
                checkpoints_dir = project_root / "checkpoints"
                if checkpoints_dir.exists():
                    pt_files = list(checkpoints_dir.glob("*.pt"))
                    if pt_files:
                        mamba_checkpoint = pt_files[0]
                        print(f"Found checkpoint at: {mamba_checkpoint}")
                
                if mamba_checkpoint is None:
                    raise FileNotFoundError(
                        f"No checkpoint found. Searched in:\n"
                        f"  - {mamba_dir}\n"
                        f"  - {project_root / 'checkpoints'}\n"
                        f"  - /workspace/checkpoints/\n"
                        f"Please specify --checkpoint_path explicitly."
                    )
    
    print(f"\n{'='*70}")
    print("SAFER v3.0 COMPLETE ARCHITECTURE PIPELINE")
    print(f"{'='*70}")
    print(f"Mamba checkpoint: {mamba_checkpoint}")
    print(f"Dataset: {dataset}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*70}\n")
    
    step_times = {}
    
    # ============================================================
    # Step 1: Train LSTM Baseline (DAL C)
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 1/7: LSTM Baseline Training (DAL C)")
    print(f"{'='*70}")
    
    baseline_dir = mamba_dir / "baseline"
    baseline_checkpoint = baseline_dir / "lstm_best.pt"
    
    step_start = time.time()
    
    if baseline_checkpoint.exists() and skip_existing:
        print(f"✓ Baseline already exists at {baseline_checkpoint}")
    elif train_baseline:
        print("Training LSTM baseline model...")
        from scripts.train_baseline_fd001 import train_baseline as train_baseline_fn
        
        train_baseline_fn(
            data_dir=data_dir,
            dataset=dataset,
            output_dir=str(baseline_dir),
            epochs=baseline_epochs,
            batch_size=64,
            d_model=64,
            n_layers=2,
        )
        print(f"✓ Baseline trained and saved to {baseline_dir}")
    else:
        baseline_checkpoint = None
        print("⚠ Skipping baseline training (will use mean fallback)")
    
    step_times['baseline_training'] = time.time() - step_start
    
    # ============================================================
    # Step 2: Train LPV-SINDy Physics Monitor (DAL C)
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 2/7: LPV-SINDy Physics Monitor Training (DAL C)")
    print(f"{'='*70}")
    
    physics_dir = mamba_dir / "physics"
    physics_model = physics_dir / "lpv_sindy_model"
    
    step_start = time.time()
    
    if Path(str(physics_model) + ".npz").exists() and skip_existing:
        print(f"✓ Physics model already exists at {physics_model}")
    elif train_physics:
        print("Training LPV-SINDy physics monitor...")
        from scripts.train_physics_fd001 import train_physics_monitor, LPVSINDyTrainConfig
        
        physics_config = LPVSINDyTrainConfig(
            window_size=5,
            polynomial_degree=2,
            threshold=0.1,
        )
        
        train_physics_monitor(
            data_dir=data_dir,
            dataset=dataset,
            output_dir=str(physics_dir),
            config=physics_config,
        )
        print(f"✓ Physics model trained and saved to {physics_dir}")
    else:
        physics_model = None
        print("⚠ Skipping physics training (will use 0.0 residuals)")
    
    step_times['physics_training'] = time.time() - step_start
    
    # ============================================================
    # Step 3: Conformal Calibration
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 3/7: Conformal Calibration")
    print(f"{'='*70}")
    
    calibration_dir = mamba_dir / "calibration"
    conformal_params = calibration_dir / "conformal_params.json"
    
    step_start = time.time()
    
    if conformal_params.exists() and skip_existing:
        print(f"✓ Conformal params already exist at {conformal_params}")
    else:
        print("Running conformal calibration...")
        from scripts.calibrate_fd001 import run_calibration
        
        run_calibration(
            checkpoint_path=str(mamba_checkpoint),
            output_dir=str(calibration_dir),
            data_dir=data_dir,
            dataset=dataset,
        )
        print(f"✓ Calibration complete: {calibration_dir}")
    
    step_times['conformal_calibration'] = time.time() - step_start
    
    # ============================================================
    # Step 4: Full SAFER Pipeline
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 4/7: Full SAFER Pipeline Integration")
    print(f"{'='*70}")
    
    safer_dir = mamba_dir / "full_safer_evaluation"
    
    step_start = time.time()
    
    print("Running complete SAFER architecture...")
    from scripts.run_full_safer_fd001 import run_full_safer_pipeline
    
    # Prepare paths
    baseline_path = str(baseline_checkpoint) if baseline_checkpoint and baseline_checkpoint.exists() else None
    physics_path = str(physics_model) if physics_model and Path(str(physics_model) + ".npz").exists() else None
    
    run_full_safer_pipeline(
        mamba_checkpoint_path=str(mamba_checkpoint),
        baseline_checkpoint_path=baseline_path,
        physics_model_path=physics_path,
        conformal_params_path=str(conformal_params),
        output_dir=str(safer_dir),
        data_dir=data_dir,
        dataset=dataset,
    )
    print(f"✓ Full SAFER pipeline complete: {safer_dir}")
    
    step_times['safer_pipeline'] = time.time() - step_start
    
    # ============================================================
    # Step 5: ONNX Export
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 5/7: ONNX Export")
    print(f"{'='*70}")
    
    onnx_path = mamba_dir / "mamba_model.onnx"
    
    step_start = time.time()
    
    if onnx_path.exists() and skip_existing:
        print(f"✓ ONNX model already exists at {onnx_path}")
    else:
        print("Exporting model to ONNX...")
        from scripts.export_onnx import export_to_onnx
        
        export_to_onnx(
            checkpoint_path=str(mamba_checkpoint),
            output_path=str(onnx_path),
            validate=True,
        )
        print(f"✓ ONNX export complete: {onnx_path}")
    
    step_times['onnx_export'] = time.time() - step_start
    
    # ============================================================
    # Step 6: Generate Report
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 6/7: Generate Comprehensive Report")
    print(f"{'='*70}")
    
    report_dir = mamba_dir / "report"
    
    step_start = time.time()
    
    print("Generating evaluation report...")
    from scripts.generate_report_fd001 import generate_report
    
    generate_report(
        checkpoint_path=str(mamba_checkpoint),
        output_dir=str(report_dir),
        data_dir=data_dir,
        dataset=dataset,
    )
    print(f"✓ Report generated: {report_dir}")
    
    step_times['report_generation'] = time.time() - step_start
    
    # ============================================================
    # Step 7: Create Deployment Package
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 7/7: Create Production Deployment Package")
    print(f"{'='*70}")
    
    deployment_dir = mamba_dir / "deployment_complete"
    
    step_start = time.time()
    
    print("Creating deployment package...")
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all necessary files
    files_to_copy = [
        (mamba_checkpoint, deployment_dir / "mamba_checkpoint.pt"),
        (onnx_path, deployment_dir / "mamba_model.onnx"),
        (conformal_params, deployment_dir / "conformal_params.json"),
    ]
    
    if baseline_checkpoint and baseline_checkpoint.exists():
        files_to_copy.append((baseline_checkpoint, deployment_dir / "lstm_baseline.pt"))
    
    if physics_model and Path(str(physics_model) + ".npz").exists():
        files_to_copy.append((Path(str(physics_model) + ".npz"), deployment_dir / "lpv_sindy_model.npz"))
        files_to_copy.append((Path(str(physics_model) + ".json"), deployment_dir / "lpv_sindy_model.json"))
    
    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied: {src.name}")
    
    # Create deployment manifest
    manifest = {
        'package': 'SAFER v3.0 Complete Deployment',
        'created': datetime.now().isoformat(),
        'dataset': dataset,
        'components': {
            'mamba_predictor': 'mamba_checkpoint.pt',
            'onnx_model': 'mamba_model.onnx',
            'conformal_params': 'conformal_params.json',
            'lstm_baseline': 'lstm_baseline.pt' if (deployment_dir / 'lstm_baseline.pt').exists() else None,
            'physics_model': 'lpv_sindy_model.npz' if (deployment_dir / 'lpv_sindy_model.npz').exists() else None,
        },
        'architecture': {
            'mamba_dal': 'E',
            'baseline_dal': 'C',
            'physics_dal': 'C',
            'simplex_dal': 'C',
        },
    }
    
    with open(deployment_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Deployment package created: {deployment_dir}")
    
    step_times['deployment_package'] = time.time() - step_start
    
    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE - SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n📁 Output Directories:")
    print(f"  Main:       {mamba_dir}")
    print(f"  Baseline:   {baseline_dir}")
    print(f"  Physics:    {physics_dir}")
    print(f"  Calibration: {calibration_dir}")
    print(f"  SAFER:      {safer_dir}")
    print(f"  Report:     {report_dir}")
    print(f"  Deployment: {deployment_dir}")
    
    print(f"\n⏱️ Step Timings:")
    for step, duration in step_times.items():
        print(f"  {step}: {duration:.1f}s")
    print(f"  TOTAL: {total_time:.1f}s")
    
    print(f"\n✅ Architecture Components Implemented:")
    print(f"  [✓] Mamba RUL Predictor (DAL E)")
    print(f"  [{'✓' if baseline_checkpoint and baseline_checkpoint.exists() else '○'}] LSTM Baseline (DAL C)")
    print(f"  [{'✓' if physics_model and Path(str(physics_model) + '.npz').exists() else '○'}] LPV-SINDy Physics Monitor (DAL C)")
    print(f"  [✓] Conformal Prediction (90% coverage)")
    print(f"  [✓] Simplex Decision Module")
    print(f"  [✓] Alert Manager")
    print(f"  [✓] ONNX Export")
    
    # Load and show final metrics
    safer_results = safer_dir / "full_safer_results.json"
    if safer_results.exists():
        with open(safer_results) as f:
            results = json.load(f)
        
        print(f"\n📊 Final Metrics:")
        metrics = results.get('metrics', {})
        print(f"  Mamba RMSE: {metrics.get('mamba_rmse', 'N/A'):.2f} cycles")
        print(f"  Final RMSE: {metrics.get('final_rmse', 'N/A'):.2f} cycles")
        print(f"  Coverage: {metrics.get('coverage', 'N/A'):.1%}")
    
    print(f"\n{'='*70}")
    print("🎉 SAFER v3.0 Architecture Fully Implemented!")
    print(f"{'='*70}\n")
    
    return deployment_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Complete SAFER v3.0 Architecture Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/complete_safer_pipeline.py \\
        --mamba_dir outputs/mamba_FD001_20251203_174328 \\
        --checkpoint_path checkpoints/best_model.pt \\
        --dataset FD001

This will:
1. Train LSTM baseline (DAL C) if not present
2. Train LPV-SINDy physics monitor (DAL C) if not present
3. Run conformal calibration
4. Execute full SAFER pipeline with all components
5. Export ONNX model
6. Generate comprehensive report
7. Create production deployment package
        """
    )
    parser.add_argument("--mamba_dir", type=str, required=True,
                        help="Directory for outputs (baseline, physics, reports, etc.)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Explicit path to Mamba checkpoint (e.g., checkpoints/best_model.pt)")
    parser.add_argument("--data_dir", type=str, default="CMAPSSData")
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline training")
    parser.add_argument("--no-physics", action="store_true",
                        help="Skip physics monitor training")
    parser.add_argument("--baseline_epochs", type=int, default=30)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps even if outputs exist")
    
    args = parser.parse_args()
    
    run_complete_pipeline(
        mamba_dir=args.mamba_dir,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        dataset=args.dataset,
        train_baseline=not args.no_baseline,
        train_physics=not args.no_physics,
        baseline_epochs=args.baseline_epochs,
        skip_existing=not args.force,
    )
