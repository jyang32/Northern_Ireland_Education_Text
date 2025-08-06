# BERTopic Environment Setup

## Create Environment from YAML

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate bertopic_env

# Verify installation
python -c "import bertopic, sentence_transformers, datamapplot; print('✅ All packages imported successfully!')"
```

## Alternative: Update Existing Environment

```bash
# Update existing environment
conda env update -f environment.yml --prune

# Or create a new environment with a different name
conda env create -f environment.yml -n bertopic_env_new
```

## Key Package Versions

- **Python**: 3.9.23
- **NumPy**: 1.26.4 (pinned to avoid 2.x conflicts)
- **BERTopic**: 0.17.3
- **Sentence-Transformers**: 5.0.0
- **PyTorch**: 2.2.2 (CPU)
- **Transformers**: 4.54.1
- **UMAP**: 0.5.9.post2
- **HDBSCAN**: 0.8.40
- **DataMapPlot**: 0.6.3

## Troubleshooting

If you encounter dependency conflicts:

1. **Clear conda cache**: `conda clean --all`
2. **Use mamba**: `mamba env create -f environment.yml` (faster solver)
3. **Create fresh environment**: Delete old env and recreate