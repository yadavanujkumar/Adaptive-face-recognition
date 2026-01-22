"""
Test script to verify all components are properly structured
and can be imported (requires dependencies installed)
"""

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test model imports
        from src.models import AdaFace, AdaFaceLoss
        print("✓ AdaFace loss imported successfully")
        
        from src.models import LivenessDetectionNet, LivenessLoss
        print("✓ Liveness detection network imported successfully")
        
        from src.models import get_mobilefacenet, get_resnet50
        print("✓ Face recognition backbones imported successfully")
        
        # Test pipeline imports
        from src.pipeline import FaceDetector, FaceAligner, FaceRecognitionPipeline
        print("✓ Pipeline components imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False


def test_documentation():
    """Test that documentation files exist"""
    import os
    
    print("\nChecking documentation...")
    
    files_to_check = [
        'README.md',
        'TRAINING.md',
        'requirements.txt',
        '.gitignore',
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
    
    print("\n✅ Documentation check complete!")


def test_structure():
    """Test that project structure is correct"""
    import os
    
    print("\nChecking project structure...")
    
    dirs_to_check = [
        'src/models',
        'src/pipeline',
        'src/api',
        'src/frontend',
        'src/utils',
        'weights',
        'data',
        'configs',
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/ exists")
        else:
            print(f"✗ {dir_path}/ missing")
    
    print("\n✅ Project structure check complete!")


if __name__ == "__main__":
    print("=" * 80)
    print("Adaptive Face Recognition System - Test Suite")
    print("=" * 80)
    
    # Test structure (doesn't require dependencies)
    test_structure()
    
    # Test documentation
    test_documentation()
    
    # Test imports (requires dependencies)
    print("\n" + "=" * 80)
    print("Note: The following test requires dependencies to be installed")
    print("=" * 80)
    test_imports()
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
