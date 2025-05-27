import json
import os
import sys

def py_to_ipynb(py_file, ipynb_file):
    """Convert a Python file to a Jupyter notebook."""
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into cells
    cells = []
    current_cell = []
    current_type = None
    
    for line in content.split('\n'):
        if line.startswith('# '):
            # If we have content in the current cell, add it to cells
            if current_cell:
                cells.append({
                    'cell_type': current_type,
                    'metadata': {},
                    'source': '\n'.join(current_cell) + '\n',
                })
                current_cell = []
            
            # Start a new markdown cell
            current_type = 'markdown'
            current_cell.append(line[2:])  # Remove the '# ' prefix
        elif line.startswith('#'):
            # Continue the current markdown cell
            if current_type != 'markdown':
                # If we have content in the current cell, add it to cells
                if current_cell:
                    cells.append({
                        'cell_type': current_type,
                        'metadata': {},
                        'source': '\n'.join(current_cell) + '\n',
                    })
                    current_cell = []
                current_type = 'markdown'
            current_cell.append(line[1:])  # Remove the '#' prefix
        else:
            # Code cell
            if current_type != 'code' and line.strip():
                # If we have content in the current cell, add it to cells
                if current_cell:
                    cells.append({
                        'cell_type': current_type,
                        'metadata': {},
                        'source': '\n'.join(current_cell) + '\n',
                    })
                    current_cell = []
                current_type = 'code'
            
            if current_type == 'code' or line.strip():
                current_cell.append(line)
    
    # Add the last cell
    if current_cell:
        cells.append({
            'cell_type': current_type,
            'metadata': {},
            'source': '\n'.join(current_cell) + '\n',
        })
    
    # Format code cells
    for cell in cells:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    
    # Create the notebook
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8.10'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    # Write the notebook
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Converted {py_file} to {ipynb_file}")

if __name__ == '__main__':
    # Convert the data exploration notebook
    py_to_ipynb(
        '/Users/onurakyuz/Desktop/Brain Tumor MRI/notebooks/01_data_exploration.py',
        '/Users/onurakyuz/Desktop/Brain Tumor MRI/notebooks/jupyter/01_data_exploration.ipynb'
    )
    
    # Convert the model development notebook
    py_to_ipynb(
        '/Users/onurakyuz/Desktop/Brain Tumor MRI/notebooks/02_model_development.py',
        '/Users/onurakyuz/Desktop/Brain Tumor MRI/notebooks/jupyter/02_model_development.ipynb'
    )
    
    print("Conversion complete!")
