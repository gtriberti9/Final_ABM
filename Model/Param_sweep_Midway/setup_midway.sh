#!/bin/bash
# Setup script for Midway ABM simulation environment

echo "Setting up ABM simulation environment on Midway..."

# Create project directory structure
mkdir -p ~/abm_simulation/{results,analysis_results,logs}
cd ~/abm_simulation

# Create conda environment with required packages
echo "Creating conda environment..."
module load python/anaconda-2022.05

# Create environment file
cat > environment.yml << 'EOF'
name: abm_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy
  - jupyter
  - pyspark
  - pip
  - pip:
    - pickle5
EOF

# Create the environment
conda env create -f environment.yml

echo "Environment created. To activate: conda activate abm_env"

# Create module loading script
cat > load_modules.sh << 'EOF'
#!/bin/bash
# Load required modules for ABM simulation
module load python/anaconda-2022.05
conda activate abm_env
export PYTHONPATH="${PYTHONPATH}:${PWD}"
EOF

chmod +x load_modules.sh

echo "Setup complete!"
echo "1. Copy your Python files to ~/abm_simulation/"
echo "2. Run: source load_modules.sh"
echo "3. Submit jobs using the provided sbatch scripts"
