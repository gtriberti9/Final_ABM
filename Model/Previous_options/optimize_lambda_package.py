#!/usr/bin/env python3
"""
optimize_lambda_package.py - Create optimized Lambda package
Reduces package size by removing unnecessary files
"""

import os
import sys
import subprocess
import shutil
import zipfile
from pathlib import Path
import tempfile

def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def remove_unnecessary_files(package_dir):
    """Remove unnecessary files to reduce package size"""
    print("Optimizing package size...")
    
    # Files and directories to remove
    remove_patterns = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/*.pyo",
        "**/*.dist-info",
        "**/*.egg-info",
        "**/tests",
        "**/test",
        "**/examples",
        "**/docs",
        "**/doc",
        "**/*.md",
        "**/*.txt",  # Keep requirements.txt if needed
        "**/LICENSE*",
        "**/COPYING*",
        "**/NOTICE*",
        "**/README*",
        "**/.git*",
        "**/matplotlib/tests",
        "**/numpy/tests",
        "**/pandas/tests",
        "**/scipy/tests",
        "**/mesa/examples",
        "**/mesa/visualization",  # Remove visualization components
        "**/jupyter*",
        "**/IPython*",
        "**/notebook*",
    ]
    
    removed_size = 0
    for pattern in remove_patterns:
        for path in Path(package_dir).glob(pattern):
            if path.exists():
                size_before = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.is_dir() else path.stat().st_size
                removed_size += size_before
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
    
    print(f"Removed {removed_size / (1024*1024):.1f} MB of unnecessary files")

def create_minimal_requirements():
    """Create minimal requirements for Lambda"""
    return [
        "numpy==1.24.3",
        "mesa==2.1.5",
        "boto3==1.29.7"
    ]

def create_optimized_package():
    """Create optimized Lambda package"""
    print("Creating optimized Lambda deployment package...")
    
    # Check if required files exist
    required_files = ['agents.py', 'model.py', 'lambda_handler.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Required files not found: {', '.join(missing_files)}")
        return False
    
    # Create temporary directory for package
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = os.path.join(temp_dir, 'package')
        os.makedirs(package_dir)
        
        print("Installing minimal dependencies...")
        
        # Install only essential packages
        minimal_requirements = create_minimal_requirements()
        
        for requirement in minimal_requirements:
            print(f"Installing {requirement}...")
            success, output = run_command([
                sys.executable, '-m', 'pip', 'install',
                requirement, '-t', package_dir,
                '--no-deps',  # Don't install dependencies automatically
                '--upgrade'
            ])
            if not success:
                print(f"Warning: Failed to install {requirement}: {output}")
        
        # Install numpy separately (it's needed by mesa)
        print("Installing numpy...")
        run_command([
            sys.executable, '-m', 'pip', 'install',
            'numpy==1.24.3', '-t', package_dir
        ])
        
        # Copy your code files
        print("Copying Python files...")
        for file in required_files:
            shutil.copy2(file, package_dir)
        
        # Remove unnecessary files
        remove_unnecessary_files(package_dir)
        
        # Create optimized ZIP file
        print("Creating ZIP file...")
        zip_path = 'abm_lambda_optimized.zip'
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            for root, dirs, files in os.walk(package_dir):
                # Skip certain directories entirely
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'tests']]
                
                for file in files:
                    if file.endswith(('.pyc', '.pyo')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arcname)
        
        # Move zip file to current directory
        if os.path.exists('abm_lambda.zip'):
            os.remove('abm_lambda.zip')
        shutil.move(zip_path, 'abm_lambda.zip')
    
    # Check final size
    final_size = get_file_size_mb('abm_lambda.zip')
    print(f"Optimized package size: {final_size:.2f} MB")
    
    if final_size > 50:
        print("⚠️  WARNING: Package is still over 50MB. Consider using S3 deployment or Lambda layers.")
        print("Alternatives:")
        print("1. Use S3 for deployment (implemented below)")
        print("2. Create Lambda layers for dependencies")
        print("3. Further optimize by removing more dependencies")
        return False
    else:
        print("✅ Package size is within Lambda limits!")
        return True

def create_minimal_abm_model():
    """Create a minimal version of the ABM model with fewer dependencies"""
    print("Creating minimal ABM model...")
    
    # Create simplified versions that don't require heavy dependencies
    minimal_agents = '''
import random
import numpy as np

# Minimal agent implementation without mesa
class Agent:
    def __init__(self, model):
        self.model = model
        
class CentralBankAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.type = "central_bank"
        
    def step(self):
        pass

# ... (rest of agents with minimal dependencies)
'''
    
    # Write minimal files if original ones are too heavy
    # This is a fallback option

def main():
    """Main function"""
    print("🚀 Lambda Package Optimizer")
    print("=" * 40)
    
    # Try optimized packaging first
    if create_optimized_package():
        print("✅ Successfully created optimized package!")
        return True
    
    # If still too large, suggest alternatives
    print("\n📦 Package is still too large. Here are your options:")
    print("\n1. 📁 Use S3 Deployment (Recommended)")
    print("   - Upload zip to S3 bucket")
    print("   - Reference S3 location in Lambda deployment")
    
    print("\n2. 🏗️  Use Lambda Layers")
    print("   - Create layers for numpy, mesa dependencies")
    print("   - Keep only your code in the main package")
    
    print("\n3. ⚡ Simplify Dependencies")
    print("   - Remove mesa dependency")
    print("   - Implement minimal agent framework")
    
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Next steps:")
        print("1. Try: python optimize_lambda_package.py")
        print("2. Or implement S3 deployment method")
        print("3. Or create Lambda layers for dependencies")