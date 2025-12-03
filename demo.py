#!/usr/bin/env python3
"""
SAFER v3.0 Installation Test and Demo.

This script verifies that SAFER v3.0 is properly installed and
demonstrates basic functionality.

Usage:
    python demo.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all SAFER modules can be imported."""
    print("Testing imports...")
    
    try:
        from safer_v3.utils.config import SAFERConfig
        from safer_v3.utils.metrics import RULMetrics
        from safer_v3.core.mamba import MambaRULPredictor
        from safer_v3.core.baselines import LSTMPredictor, TransformerPredictor
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.decision.simplex import SimplexDecisionModule
        from safer_v3.decision.conformal import SplitConformalPredictor
        from safer_v3.decision.alerts import AlertManager
        from safer_v3.simulation.engine_sim import EngineSimulator
        from safer_v3.simulation.data_generator import generate_trajectory
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nTry installing the package:")
        print("  pip install -e .")
        return False


def test_data():
    """Test that C-MAPSS data is available."""
    print("\nTesting data availability...")
    
    data_dir = Path("CMAPSSData")
    required_files = [
        "train_FD001.txt",
        "test_FD001.txt",
        "RUL_FD001.txt",
    ]
    
    all_present = True
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
            all_present = False
    
    if all_present:
        print("✅ All data files present!")
    else:
        print("\n❌ Some data files missing.")
        print("Download C-MAPSS data from:")
        print("  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    
    return all_present


def demo_model_creation():
    """Demonstrate model creation."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from safer_v3.core.mamba import MambaRULPredictor
        from safer_v3.core.baselines import LSTMPredictor
        
        # Create Mamba model
        mamba = MambaRULPredictor(
            d_input=14, 
            d_model=64, 
            d_state=16, 
            n_layers=4,
            use_jit=False  # Disable JIT for demo
        )
        n_params = sum(p.numel() for p in mamba.parameters())
        print(f"  ✅ Mamba model created: {n_params:,} parameters")
        
        # Create LSTM baseline
        lstm = LSTMPredictor(d_input=14, d_model=64, n_layers=2)
        n_params_lstm = sum(p.numel() for p in lstm.parameters())
        print(f"  ✅ LSTM model created: {n_params_lstm:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(1, 50, 14)
        with torch.no_grad():
            mamba_out = mamba(dummy_input)
            lstm_out = lstm(dummy_input)
        
        print(f"  ✅ Forward pass successful")
        print(f"     Mamba output: {mamba_out.item():.2f}")
        print(f"     LSTM output: {lstm_out.item():.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def demo_simulation():
    """Demonstrate synthetic data generation."""
    print("\nTesting simulation...")
    
    try:
        from safer_v3.simulation.engine_sim import EngineSimulator
        from safer_v3.simulation.data_generator import generate_trajectory
        
        # Create simulator
        simulator = EngineSimulator(total_cycles=100, seed=42)
        trajectory = simulator.generate_trajectory()
        
        print(f"  ✅ Generated trajectory: {len(trajectory['cycle'])} cycles")
        print(f"     Sensors shape: {trajectory['sensors'].shape}")
        print(f"     Initial RUL: {trajectory['rul'][0]:.0f}")
        print(f"     Final RUL: {trajectory['rul'][-1]:.0f}")
        
        return True
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False


def demo_decision_modules():
    """Demonstrate decision modules."""
    print("\nTesting decision modules...")
    
    try:
        from safer_v3.decision.simplex import SimplexDecisionModule, SimplexConfig
        from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules
        
        # Create Simplex
        config = SimplexConfig(
            physics_threshold=0.1,
            divergence_threshold=30.0,
        )
        simplex = SimplexDecisionModule(config)
        
        # Make a decision
        result = simplex.decide(
            complex_rul=45.0,
            baseline_rul=48.0,
            rul_lower=40.0,
            rul_upper=50.0,
            physics_residual=0.05,
        )
        
        print(f"  ✅ Simplex decision: RUL={result.rul:.1f}, State={result.state.name}")
        
        # Create alert manager
        alert_manager = AlertManager()
        alert_manager.add_rules(create_rul_alert_rules())
        
        # Process RUL
        alerts = alert_manager.process(rul_value=result.rul)
        print(f"  ✅ Alert manager: {len(alerts)} alerts generated")
        
        return True
    except Exception as e:
        print(f"❌ Decision modules failed: {e}")
        return False


def show_next_steps():
    """Show next steps."""
    print("\n" + "="*60)
    print("SAFER v3.0 is ready! 🚀")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Train Mamba model:")
    print("   python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 50")
    print("\n2. Train baseline models:")
    print("   python -m safer_v3.scripts.train_baselines --data_dir CMAPSSData --dataset FD001 --model all")
    print("\n3. See QUICKSTART.md for detailed usage examples")
    print("\n4. Check outputs/ directory for trained models and metrics")


def main():
    """Run all tests."""
    print("="*60)
    print("SAFER v3.0 Installation Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data),
        ("Model Creation", demo_model_creation),
        ("Simulation", demo_simulation),
        ("Decision Modules", demo_decision_modules),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        show_next_steps()
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
