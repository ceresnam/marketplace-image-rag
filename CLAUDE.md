See README.md for details about project goals

## Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies using uv
uv sync

# Install and configure Databricks CLI
# https://docs.databricks.com/aws/en/dev-tools/cli/install
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks auth login --host https://[DBC_HOST].cloud.databricks.com/

# Run the application locally
python main.py

# Or deploy to Databricks cluster
databricks workspace import main.py /Workspace/marketplace-classification/main.py
```

## Project Structure
```
marketplace-image-classification/
├── main.py              # Entry point
├── CLAUDE.md           # This file
└── ...
```

## Usage
TBD - Implementation in progress
