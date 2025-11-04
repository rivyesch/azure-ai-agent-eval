# Azure AI Agent Evaluation Toolkit

This repository provides tools for evaluating Azure AI agents, with a focus on processing multi-turn conversations and optimizing evaluation datasets for computational efficiency.

## Key Components

### Dependency Management
This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management. The `uv.lock` file contains the exact dependency versions used during development.

### Main Notebooks
- `agent25_batch_eval.ipynb`: **Primary evaluation notebook** that handles:
  - Processing multi-turn conversation threads
  - Batch evaluation of agent responses
  - Integration with Azure AI evaluation services
  - Post-processing of evaluation results

### Supporting Files
- `postprocess_evaluation_jsonl.py`: Core processing module that:
  - Cleans and optimizes Azure conversation datasets
  - Extracts relevant evaluation data
  - Reduces computational overhead by removing redundant information
  
### Reference/Example
- `agent25_eval.ipynb`: Simple testing notebook for single question-answer evaluation
- `Evaluate_Azure_AI_Agent_Quality.ipynb`: Reference notebook from Azure samples (for comparison)

## Getting Started

### Prerequisites
- Python 3.8+
- Azure AI services access
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installation
1. Clone the repository
2. Install dependencies using `uv`:
   ```bash
   # Create and activate a virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies from uv.lock
   uv pip sync
   ```
3. Copy `.env.example` to `.env` and configure your Azure credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## Usage

The main workflow is handled through `agent25_batch_eval.ipynb`:

1. **Prepare your conversation data** in the expected format
2. Run the notebook cells to:
   - Load and preprocess conversation threads
   - Execute batch evaluations
   - Process and analyze results
   - Generate evaluation metrics

For single question-answer evaluation, use `agent25_eval.ipynb` as a testing ground.

## Data Processing Pipeline

The evaluation pipeline follows these steps:
1. Load raw conversation threads
2. Process through `postprocess_evaluation_jsonl.py` to:
   - Extract relevant conversation turns
   - Remove redundant metadata
   - Format for efficient evaluation
3. Submit to Azure AI evaluation services
4. Process and visualize results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
