import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure"""
    # Define directories to create
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'models/logs',
        'src',
        'notebooks',
        'tests'
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py in Python package directories
        if directory in ['src', 'tests']:
            Path(directory) / '__init__.py'.touch()
    
    # Create placeholder files
    placeholder_files = {
        'notebooks/exploration.ipynb': '# Data Exploration Notebook\n',
        'notebooks/visualization.ipynb': '# Results Visualization Notebook\n',
        'tests/test_model.py': '# Model tests\n',
        'tests/test_utils.py': '# Utility function tests\n',
        'src/utils.py': '# Utility functions\n',
        'src/data_processing.py': '# Data processing utilities\n'
    }
    
    for file_path, content in placeholder_files.items():
        with open(file_path, 'w') as f:
            f.write(content)
    
    # Move existing files to their new locations
    file_moves = {
        'model.py': 'src/model.py',
        'train.py': 'src/train.py',
        'main_script.py': 'src/main_script.py'
    }
    
    for src, dst in file_moves.items():
        if os.path.exists(src):
            shutil.move(src, dst)

if __name__ == '__main__':
    create_directory_structure()
    print("Project directory structure created successfully!")
    print("\nNext steps:")
    print("1. Create a virtual environment: python -m venv venv")
    print("2. Activate the virtual environment:")
    print("   - On Unix/macOS: source venv/bin/activate")
    print("   - On Windows: venv\\Scripts\\activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Start developing!") 