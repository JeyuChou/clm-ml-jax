# CLM-ML-JAX: AI-Powered Fortran to JAX Translation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE)

> An ambitious project to translate the Community Land Model (CLM) from Fortran to Python/JAX using a multi-agent AI system powered by Claude, maintaining scientific accuracy while enabling modern optimization and enhanced testing.


### Key Goals

- ✅ Convert Fortran CLM source to JAX Python equivalents
- ✅ Maintain exact scientific accuracy and physics formulations
- ✅ Generate comprehensive test suites automatically
- ✅ Create detailed documentation and translation notes
- ✅ Enable performance optimization through JAX's JIT compilation


## Features

### 🤖 Multi-Agent System

- **TranslatorAgent**: Converts Fortran modules to JAX Python with type hints, docstrings, and functional patterns
- **TestAgent**: Analyzes Python signatures and generates comprehensive pytest files
- **RepairAgent**: Debugs failed translations, identifies root causes, and iteratively fixes code
- **BaseAgent**: Provides Claude API integration, conversation management, and cost tracking

### 🔬 Translation Capabilities

- Fortran 90/95 to Python 3.9+ with JAX
- Module-level variables → NamedTuples/dataclasses
- Subroutines → Pure functions with type hints
- DO loops → JAX vmap or vectorized operations
- Fortran types → Python dataclasses
- Parameter modules → Immutable configuration classes

---
## Pipeline

```
1. Fortran Source → Fortran Analyzer → JSON Analysis
2. JSON Analysis → TranslatorAgent → JAX Python + Notes
3. JAX Python → TestAgent → Pytest Files + Test Data
4. Run Tests → [PASS ✓] Done
            └─ [FAIL ✗] → RepairAgent → Fixed Code → Repeat
```

---

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **JAX**: Latest version
- **Anthropic API Key**: Get from [console.anthropic.com](https://console.anthropic.com)

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk**: 2GB free space

### Step 1: Clone Repository

```bash
git clone https://github.com/AyaLahlou/clm-ml-jax.git
cd clm-ml-jax
```

### Step 2: Install JAX Agent System and Main Project Dependencies

```bash
cd jax-agents
pip install -e .
pip install jax jaxlib numpy pytest

```

### Step 3: Create `jax-agents/.env`

Configure API key 

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-xxx

# Optional
LOG_LEVEL=INFO
JAX_ENABLE_X64=True
```

---

## Quick Start

### 1. Translate a Fortran Module

```bash
cd jax-agents
./run_translation_workflow.sh --translate --module clm_varctl
```

This will:
- Load static analysis JSON for `clm_varctl`
- Translate each unit iteratively
- Assemble into complete module
- Save to `translated_modules/`

### 2. Generate Tests

```bash
./run_translation_workflow.sh --test --module clm_varctl
```

Creates:
- `test_clm_varctl.py` - Pytest file
- `test_data_clm_varctl.json` - Synthetic test data
- `test_documentation_clm_varctl.md` - Test docs

### 3. Auto-Repair Failed Tests

```bash
./run_translation_workflow.sh --repair --module clm_varctl --max-iterations 5
```

Automatically:
- Analyzes test failures
- Identifies root causes
- Generates fixes
- Runs tests again
- Iterates until passing (or max iterations)

### 4. Complete Workflow

```bash
./run_translation_workflow.sh --all --module clm_varctl
```
Runs: Translate → Test → Repair in sequence.

---

## Project Structure

```
clm-ml-jax/
├── src/                              # Translated JAX modules 
│   └── ...
├── jax-agents/                       # AI Translation System 
│   ├── src/jax_agents/
│   │   ├── base_agent.py             # Base agent with Claude API
│   │   ├── translator.py             # Fortran → JAX translator
│   │   ├── test_agent.py             # Test generator
│   │   ├── repair_agent.py           # Auto-repair agent
│   │   ├── prompts/                  # LLM prompts
│   │   │   ├── translation_prompts.py
│   │   │   ├── test_prompts.py
│   │   │   └── repair_prompts.py
│   │   └── ...
│   ├── run_translation_workflow.sh   # Main workflow script
│   └── ...
├── tests/                            # Comprehensive test suite (68 files)
│   ├── conftest.py                   # Pytest fixtures & config
│   ├── clm_src_main/
│   │   └── ...
│   └── ...
└── ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/clm_src_main/test_clm_varctl.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"

# Run with markers
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only
```

---

## Development

### Adding a New Agent

1. Create agent in `src/jax_agents/your_agent.py`:

```python
from jax_agents.base_agent import BaseAgent

class YourAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="YourAgent",
            role="Your agent's role",
            **kwargs
        )

    def your_method(self, input_data):
        prompt = f"Process this: {input_data}"
        response = self.query_claude(prompt)
        return response
```

2. Add prompts in `src/jax_agents/prompts/your_prompts.py`

3. Add tests in `tests/test_your_agent.py`

4. Update configuration in `config.yaml`

---

## How to Contribute

We welcome contributions! 

1. **Report Bugs**: Open an issue with reproduction steps
2. **Suggest Features**: Open an issue with use case description
3. **Submit PRs**: Fork, create feature branch, submit PR
4. **Improve Docs**: Documentation improvements are always welcome

### Development Guidelines

- Follow existing code style (Black, Ruff)
- Add tests for new features
- Update documentation
- Ensure all tests pass
- Add type hints to all functions

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on improving the project

---

## License

This project is licensed under the **BSD-3-Clause License**. See [LICENSE](LICENSE) for details.

---


## Citation

If you use this project in your research, please cite:

```bibtex
@software{clm_ml_jax,
  title={CLM-ML-JAX: AI-Powered Fortran to JAX Translation System},
  author={Aya Lahlou},
  year={2024},
  url={https://github.com/AyaLahlou/clm-ml-jax}
}
```


⭐ **Star this repo** if you find it useful!

🐛 **Report issues** to help us improve!

🤝 **Contribute** to advance scientific computing!
