"# RAPTOR---Advanced-Retrieval"

```bash
# Create environment
conda create --name ragraptor python=3.9 -y

# Activate
conda activate ragraptor

# Install
conda install -c anaconda ipykernel -y

# Create environment
python -m ipykernel install --user --name=ragraptor -y

# Install
pip install --no-input -r requirements.txt
```

Save environment if needed

```bash
# Save environment
pip list --format=freeze > requirements.txt
```

Remove environment

```bash
conda deactivate
conda remove --name ragraptor --all -y
```
