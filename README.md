# xai-mini-research

A holiday project created for the purpose of testing learned XAI methods and evaluating them against time series regression.

# Evaluation of XAI Methods in Time Series Regression Models

This project implements a simple regression model trained on synthetic data including explainable AI methods for the purpose of evaluating the effectiveness of current XAI methods. Optionally, clustering analysis inspired by SpRAy will also be evaluated to see if spurious correlations or Clever Hans behavior can be correctly recognized [Lapuschkin et al., 2019].

## Project Structure

```text
xai-mini-research/
|-- configs/
|   `-- default.yaml      # Default experiment configuration (data, model, explainability)
|-- reports/              # Experiment logs / write-ups
|-- src/
|   `-- xai_mini_research/
|       |-- __init__.py
|       |-- config.py     # Config loading helpers
|       `-- data.py       # Synthetic time-series data generation and visualization helpers
|-- tests/
|   |-- test_initial.py   # Basic config path tests
|   `-- test_data.py      # Test data loader
|-- pytest.ini
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Mellosteen/xai-mini-research.git
cd xai-mini-research
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

## Author
 - Austin Samuel Qiu

## References

- Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W.,
  and Müller, K.-R. (2019). *Unmasking Clever Hans predictors and assessing
  what machines really learn*. Nature Communications, 10, 1096.
  https://doi.org/10.1038/s41467-019-08987-4

- Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R.,
  and Samek, W. (2015). *On pixel-wise explanations for non-linear
  classifier decisions by layer-wise relevance propagation*.
  PLOS ONE, 10(7), e0130140.
  https://doi.org/10.1371/journal.pone.0130140

- Yassen, M.A., El-Kenawy, ES.M., Abdel-Fattah, M.G. et al.
  Explainable artificial intelligence for wind power forecasting
  model based on long short-term memory. Neural Comput & Applic 37,
  14589–14611 (2025).
  https://doi.org/10.1007/s00521-025-11230-5