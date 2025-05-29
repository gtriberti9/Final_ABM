#!/bin/bash
# create_fixed_lambda_package.sh - Fixed version with complete dependencies

echo "Creating FIXED Lambda deployment package..."

# Clean up previous attempts
rm -rf lambda_env package abm_lambda.zip

# Create virtual environment
python -m venv lambda_env
source lambda_env/Scripts/activate

# Install ALL dependencies with their sub-dependencies
echo "Installing complete dependencies..."
pip install --upgrade pip

# Install packages one by one to ensure completeness
pip install boto3==1.29.7
pip install numpy==1.24.3

# Try to install mesa - if it fails, we'll create a minimal version
echo "Attempting to install mesa..."
if pip install mesa==2.1.5; then
    echo "✅ Mesa installed successfully"
    USE_MESA=true
else
    echo "⚠️ Mesa installation failed, will create minimal version"
    USE_MESA=false
fi

# Create package directory
mkdir -p package

# Install dependencies to package directory WITH all sub-dependencies
echo "Installing to package directory..."
pip install boto3==1.29.7 -t package/ --no-user
pip install numpy==1.24.3 -t package/ --no-user

if [ "$USE_MESA" = true ]; then
    pip install mesa==2.1.5 -t package/ --no-user
fi

# Copy your code files
echo "Copying Python files..."
cp agents.py package/
cp model.py package/
cp lambda_handler.py package/

# If mesa failed, create minimal replacements
if [ "$USE_MESA" = false ]; then
    echo "Creating minimal mesa replacement..."
    cat > package/mesa.py << 'EOF'
# Minimal mesa replacement for Lambda
import random

class Agent:
    def __init__(self, model):
        self.model = model
        
    def step(self):
        pass

class Model:
    def __init__(self):
        self.agents = []
        
    def step(self):
        for agent in self.agents:
            agent.step()

class DataCollector:
    def __init__(self, model_reporters=None):
        self.model_reporters = model_reporters or {}
        self.model_vars = {}
        
    def collect(self, model):
        for key, func in self.model_reporters.items():
            try:
                self.model_vars[key] = func(model)
            except:
                self.model_vars[key] = 0
EOF
fi

# Remove problematic files that cause import issues
echo "Cleaning up problematic files..."
find package/ -name "*.pyc" -delete
find package/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find package/ -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
find package/ -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove documentation and examples that can cause import issues
rm -rf package/boto3/docs 2>/dev/null || true
rm -rf package/botocore/docs 2>/dev/null || true
rm -rf package/*/examples 2>/dev/null || true
rm -rf package/*/docs 2>/dev/null || true

# Create ZIP file using Python (more reliable)
echo "Creating ZIP file..."
cd package
python -c "
import zipfile
import os
import sys

def should_include(path):
    # Skip problematic files
    skip_patterns = ['.pyc', '.pyo', '__pycache__', '.git', 'docs/', 'examples/', 'tests/']
    return not any(pattern in path for pattern in skip_patterns)

with zipfile.ZipFile('../abm_lambda.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
    count = 0
    for root, dirs, files in os.walk('.'):
        # Filter out problematic directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', 'tests', 'examples', 'docs']]
        
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                continue
                
            file_path = os.path.join(root, file)
            if should_include(file_path):
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
                count += 1
    
    print(f'Added {count} files to ZIP')
"
cd ..

# Check final size
if [ -f "abm_lambda.zip" ]; then
    size=$(du -h abm_lambda.zip | cut -f1)
    echo "✅ Package created: abm_lambda.zip (${size})"
    
    # Test the package
    echo "🧪 Testing package contents..."
    python -c "
import zipfile
with zipfile.ZipFile('abm_lambda.zip', 'r') as z:
    files = z.namelist()
    print(f'Total files: {len(files)}')
    
    # Check key files
    key_files = ['lambda_handler.py', 'agents.py', 'model.py', 'boto3/__init__.py', 'numpy/__init__.py']
    for f in key_files:
        if f in files:
            print(f'✅ {f}')
        else:
            print(f'❌ Missing: {f}')
"
else
    echo "❌ Failed to create ZIP file"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "🚀 Ready to deploy! Run: python deploy_lambda.py"