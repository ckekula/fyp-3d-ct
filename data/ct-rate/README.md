# Download script for CT-RATE dataset
This is an example download script to download CT-RATE dataset

## Steps

### Go to the Dataset Page
1. Open the CT-RATE dataset page on Hugging Face:\
https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

2. Click “Access repository” and fill out the request form with:
    * Your name
    * Institution/organization
    * Email
    * Agree to terms

3. Create an access token in your Hugging Face settings → Access Tokens

### Setup development environment
1. Install Required Tools

```
# create virtual environment
uv venv --python 3.12.12

# Install required packages
uv pip install huggingface_hub datasets
```

2. Login to Huggingface
```
hf auth login
```

3. Paste the access token

### Download the dataset

Use the `download_only_train_data.py` script to download the training data