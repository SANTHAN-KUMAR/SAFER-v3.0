"""
End-to-End FD001 Pipeline Automation.

Complete workflow from model to deployment in one script:
1. Conformal calibration (validation set)
2. Simplex + alert integration (test set)
3. Evaluation report generation
4. Deployment package creation

This is the primary entry point for the SAFER v3.0 FD001 pipeline.

Usage:
    python scripts/end_to_end_fd001.py [--skip-calibration] [--skip-alerts] [--skip-report]

Options:
    --skip-calibration    Skip calibration step (use existing params)
    --skip-alerts         Skip alert/simplex integration
    --skip-report         Skip report generation
    --output-dir          Custom output directory
"""

import sys
import argparse
from pathlib import Path
import subprocess
from loguru import logger
from datetime import datetime


def run_script(script_name: str, description: str) -> bool:
    """Run a pipeline script.
    
    Args:
        script_name: Name of script to run
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info(f"STEP: {description}")
    logger.info("=" * 70)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.success(f"✓ {description} completed successfully")
            return True
        else:
            logger.error(f"✗ {description} failed with return code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with error: {e}")
        return False


def main():
    """Main end-to-end pipeline."""
    parser = argparse.ArgumentParser(
        description='End-to-end FD001 SAFER v3.0 pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip conformal calibration (use existing parameters)',
    )
    parser.add_argument(
        '--skip-alerts',
        action='store_true',
        help='Skip Simplex and alert integration',
    )
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip evaluation report generation',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory',
    )
    
    args = parser.parse_args()
    
    # Print banner
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "  SAFER v3.0 - FD001 RUL Prediction Pipeline".center(68) + "║")
    logger.info("║" + "  End-to-End Automation".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")
    
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("")
    
    # Track progress
    steps = []
    
    # Step 1: Conformal Calibration
    if not args.skip_calibration:
        steps.append(("calibrate_fd001.py", "1. Conformal Calibration (Validation Set)"))
    else:
        logger.info("⊘ Skipping conformal calibration (using existing parameters)")
    
    # Step 2: Simplex & Alerts
    if not args.skip_alerts:
        steps.append(("alert_and_simplex_fd001.py", "2. Simplex Decision & Alert Integration (Test Set)"))
    else:
        logger.info("⊘ Skipping Simplex and alert integration")
    
    # Step 3: Evaluation Report
    if not args.skip_report:
        steps.append(("generate_report_fd001.py", "3. Evaluation Report Generation"))
    else:
        logger.info("⊘ Skipping evaluation report generation")
    
    # Step 4: Deployment Package (always run)
    steps.append(("create_deployment_package.py", "4. Create Deployment Package"))
    
    # Execute pipeline
    logger.info(f"Running {len(steps)} steps...\n")
    
    results = {}
    for script_name, description in steps:
        success = run_script(script_name, description)
        results[description] = success
        
        if not success:
            logger.error("")
            logger.error("Pipeline halted due to failure")
            logger.error(f"Failed at: {description}")
            logger.error("")
            return 1
        
        logger.info("")
    
    # Print summary
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    
    for step, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {status:10s} {step}")
    
    logger.info("=" * 70)
    logger.info("")
    
    # Print next steps
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + "  NEXT STEPS".center(68) + "║")
    logger.info("╠" + "═" * 68 + "╣")
    logger.info("║" + " " * 68 + "║")
    logger.info("║  1. Review outputs in:                                           " + " " * 2 + "║")
    logger.info("║     outputs/mamba_FD001_20251203_174328/                          " + " " * 2 + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("║  2. Deployment package ready at:                                 " + " " * 2 + "║")
    logger.info("║     outputs/mamba_FD001_20251203_174328/deployment_calibrated/    " + " " * 2 + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("║  3. Key artifacts:                                               " + " " * 2 + "║")
    logger.info("║     - calibration/: Conformal prediction parameters              " + " " * 2 + "║")
    logger.info("║     - alerts/: Simplex decisions and alert history               " + " " * 2 + "║")
    logger.info("║     - report/: Evaluation report and dashboard                   " + " " * 2 + "║")
    logger.info("║     - deployment_calibrated/: Production-ready package           " + " " * 2 + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("║  4. For deployment:                                              " + " " * 2 + "║")
    logger.info("║     - Copy deployment_calibrated/ to production                  " + " " * 2 + "║")
    logger.info("║     - Review DEPLOYMENT_README.md                                " + " " * 2 + "║")
    logger.info("║     - Use inference.py module for predictions                    " + " " * 2 + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("║  5. For monitoring:                                              " + " " * 2 + "║")
    logger.info("║     - Track interval coverage on new data                        " + " " * 2 + "║")
    logger.info("║     - Monitor alert patterns for domain drift                    " + " " * 2 + "║")
    logger.info("║     - Schedule quarterly recalibration                           " + " " * 2 + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")
    
    logger.info(f"End time: {datetime.now().isoformat()}")
    logger.success("✓ Pipeline completed successfully!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
