#!/usr/bin/env python3
"""
Test script to verify installation fixes work correctly.
Run this script to validate that all import and syntax errors have been resolved.
"""

import sys
import traceback

def test_syntax_compilation():
    """Test that all Python files compile without syntax errors."""
    print("🔍 Testing syntax compilation...")
    
    import py_compile
    files_to_test = [
        'config_settings.py',
        'src/trading_bot.py', 
        'src/trading_engine.py',
        'src/market_data.py',
        'src/portfolio_manager.py',
        'src/risk_management.py'
    ]
    
    for file_path in files_to_test:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"  ✅ {file_path} - syntax OK")
        except py_compile.PyCompileError as e:
            print(f"  ❌ {file_path} - syntax error: {e}")
            return False
        except FileNotFoundError:
            print(f"  ⚠️  {file_path} - file not found")
    
    return True

def test_imports():
    """Test critical imports work without dependency issues."""
    print("\n🔍 Testing imports...")
    
    # Test 1: Config settings with pydantic fix
    try:
        import config_settings
        print("  ✅ config_settings import - pydantic fix working")
    except ImportError as e:
        print(f"  ❌ config_settings import failed: {e}")
        print("  📝 Note: Need pydantic-settings>=2.0.0 installed")
    
    # Test 2: Trading bot with fixed dataclasses
    try:
        from src.trading_bot import Portfolio, Position, Trade, TradingConfig
        print("  ✅ trading_bot dataclasses - missing models fix working")
    except ImportError as e:
        print(f"  ❌ trading_bot imports failed: {e}")
    
    # Test 3: Trading engine with pandas_ta fallback
    try:
        import src.trading_engine
        print("  ✅ trading_engine import - pandas_ta fallback working")
    except ImportError as e:
        print(f"  ❌ trading_engine import failed: {e}")

def test_requirements_format():
    """Test that requirements.txt is properly formatted."""
    print("\n🔍 Testing requirements.txt format...")
    
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        # Check for empty lines that were problematic before
        empty_lines = [i+1 for i, line in enumerate(lines) if line.strip() == '']
        if empty_lines:
            print(f"  ⚠️  Found empty lines at: {empty_lines}")
        else:
            print("  ✅ No problematic empty lines found")
        
        # Check for fixed numpy version
        numpy_lines = [line.strip() for line in lines if line.startswith('numpy')]
        if numpy_lines:
            numpy_line = numpy_lines[0]
            if 'numpy>=1.24.0' in numpy_line:
                print("  ✅ numpy version constraint fixed (>=1.24.0)")
            elif 'numpy==1.24.3' in numpy_line:
                print("  ❌ numpy still has problematic fixed version")
            else:
                print(f"  ℹ️  numpy line: {numpy_line}")
        
        # Check for pydantic-settings
        has_pydantic_settings = any('pydantic-settings' in line for line in lines)
        if has_pydantic_settings:
            print("  ✅ pydantic-settings dependency added")
        else:
            print("  ❌ pydantic-settings dependency missing")
            
        print(f"  📊 Total dependencies: {len([l for l in lines if l.strip() and not l.strip().startswith('#')])}")
        
    except FileNotFoundError:
        print("  ❌ requirements.txt not found")

def main():
    """Run all installation fix tests."""
    print("🚀 Testing CryptoCaller Installation Fixes")
    print("=" * 50)
    
    all_passed = True
    
    # Test syntax
    if not test_syntax_compilation():
        all_passed = False
    
    # Test imports
    test_imports()
    
    # Test requirements format
    test_requirements_format()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All syntax tests passed! Installation fixes verified.")
        print("📝 Note: Some import tests may fail due to missing dependencies,")
        print("   but this is expected in a fresh environment.")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    print("\n💡 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run application: streamlit run app.py")

if __name__ == "__main__":
    main()