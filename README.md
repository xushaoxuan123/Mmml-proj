### Run Demo Code

**Environment**

```bash
    conda create --name multimodal5 python=3.10
    conda activate multimodal5

    pip install torch torchvision
    pip install spacy
    pip install pandas
    pip install matplotlib

    python -m spacy download en_core_web_sm

    python main.py
```

**Dataset**

```bash
    pip install kagglehub
```

```python
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("adityajn105/flickr8k")

    print("Path to dataset files:", path)
```