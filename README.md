# CLM-ML-JAX: AI-Powered Fortran to JAX Translation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE)

> An ambitious project to translate the Community Land Model (CLM) from Fortran to Python/JAX using a multi-agent AI system powered by Claude, maintaining scientific accuracy while enabling modern optimization and enhanced testing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Translation Workflow](#translation-workflow)
  - [Test Generation](#test-generation)
  - [Automated Repair](#automated-repair)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project builds an LLM translation system designed to convert complex Fortran Earth system modeling code from the Community Land Model (CLM) to modern Python/JAX. The project uses a multi-agent architecture powered by Anthropic's Claude to automate the translation process while maintaining exact scientific accuracy.

### Key Goals

- ✅ Convert Fortran CLM source to JAX Python equivalents
- ✅ Maintain exact scientific accuracy and physics formulations
- ✅ Generate comprehensive test suites automatically
- ✅ Create detailed documentation and translation notes
- ✅ Enable performance optimization through JAX's JIT compilation

### What Makes This Special

1. **AI-Powered Translation**: Uses Claude Sonnet 4.5 with specialized prompts for Fortran → JAX conversion
2. **Unit-by-Unit Approach**: Breaks complex modules into manageable translation units
3. **Automated Testing**: Generates pytest files with synthetic test data
4. **Self-Healing**: RepairAgent automatically debugs and fixes failed translations
5. **Scientific Fidelity**: Preserves exact physics while modernizing code structure

---

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
## Architecture

```
1. Fortran Source → Fortran Analyzer → JSON Analysis
2. JSON Analysis → TranslatorAgent → JAX Python + Notes
3. JAX Python → TestAgent → Pytest Files + Test Data
4. Run Tests → [PASS ✓] Done
            └─ [FAIL ✗] → RepairAgent → Fixed Code → Repeat
```

### Agent System

```python
BaseAgent (base_agent.py)
├── Claude API integration with streaming support
├── Conversation history management
├── Retry logic with exponential backoff
├── Token usage tracking & cost estimation
└── Comprehensive logging

TranslatorAgent (translator.py)
├── Unit-by-unit translation approach
├── Static analysis JSON integration
├── Reference pattern matching
├── Module assembly & code generation
└── Structured output (src/, tests/, docs/)

TestAgent (test_agent.py)
├── Function signature analysis
├── Synthetic test data generation
├── Edge case identification
├── Pytest file creation
└── Test documentation generation

RepairAgent (repair_agent.py)
├── Test failure analysis
├── Root cause identification
├── Iterative code fixing (max 5 iterations)
├── Test execution & validation
└── Comprehensive RCA reports
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

### Step 2: Install JAX Agent System

```bash
cd jax-agents
pip install -e .
```

This installs:
- `anthropic>=0.40.0` - Claude API client
- `python-dotenv>=1.0.0` - Environment variable management
- `pyyaml>=6.0` - Configuration parsing
- `pydantic>=2.0.0` - Data validation
- `rich>=13.0.0` - Beautiful console output
- `tenacity>=8.0.0` - Retry logic

### Step 3: Install Development Dependencies (Optional)

```bash
pip install -e ".[dev]"
```

Adds:
- `pytest>=7.0.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Linting
- `mypy>=1.0.0` - Type checking

### Step 4: Install Main Project Dependencies

```bash
cd ..
pip install jax jaxlib numpy pytest
```

### Step 5: Configure API Key

Create a `.env` file in the `jax-agents/` directory:

```bash
cd jax-agents
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

**Security Note**: Never commit `.env` files to version control. The `.gitignore` should already exclude them.

### Step 6: Verify Installation

```bash
# Test imports
python -c "import jax; import anthropic; print('✓ Installation successful')"

# Run a simple test
cd ..
pytest tests/ -v -k "test_clm_varpar" --maxfail=1
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

## Usage

### Translation Workflow

#### Using the Shell Script (Recommended)

```bash
cd jax-agents

# Interactive mode
./run_translation_workflow.sh --interactive

# Translate specific module
./run_translation_workflow.sh --translate --module SoilTemperatureMod

# Translate multiple modules
./run_translation_workflow.sh --translate --module "clm_varctl,clm_varpar,clm_varcon"

# Complete workflow for multilayer canopy
./run_translation_workflow.sh --all --module MLCanopyFluxesMod
```

#### Using Python API

```python
from jax_agents import TranslatorAgent
from pathlib import Path

# Initialize agent
translator = TranslatorAgent(
    analysis_results_path=Path("path/to/analysis_results.json"),
    translation_units_path=Path("path/to/translation_units.json"),
    fortran_root=Path("../CLM-ml_v1"),
    model="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=48000,
)

# Translate module
result = translator.translate_module(
    module_name="clm_varctl",
    output_dir=Path("translated_modules/clm_varctl"),
)

# Or use structured output
result.save_structured(project_root=Path(".."))

# Check cost
cost = translator.get_cost_estimate()
print(f"Translation cost: ${cost['total_cost_usd']:.2f}")
```

### Test Generation

#### Using Shell Script

```bash
./run_translation_workflow.sh --test --module clm_varctl
```

#### Using Python API

```python
from jax_agents import TestAgent
from pathlib import Path

# Initialize test agent
test_agent = TestAgent()

# Generate tests
test_result = test_agent.generate_tests(
    module_name="clm_varctl",
    python_code=open("translated_modules/clm_varctl/clm_varctl.py").read(),
    output_dir=Path("translated_modules/clm_varctl/tests"),
)

print(f"Generated {test_result.num_tests} tests")
```

### Automated Repair

#### Using Shell Script

```bash
# Basic repair
./run_translation_workflow.sh --repair --module clm_varctl

# With custom max iterations
./run_translation_workflow.sh --repair --module clm_varctl --max-iterations 10
```

#### Using Python API

```python
from jax_agents import RepairAgent
from pathlib import Path

# Initialize repair agent
repair_agent = RepairAgent(max_repair_iterations=5)

# Read files
fortran_code = open("CLM-ml_v1/src/clm_varctl.F90").read()
failed_python = open("src/clm_src_main/clm_varctl.py").read()
test_report = open("pytest_output.txt").read()

# Repair translation
repair_result = repair_agent.repair_translation(
    module_name="clm_varctl",
    fortran_code=fortran_code,
    failed_python_code=failed_python,
    test_report=test_report,
    test_file_path=Path("tests/clm_src_main/test_clm_varctl.py"),
    output_dir=Path("repair_output"),
)

print(f"Repaired in {repair_result.iterations} iterations")
print(f"All tests passed: {repair_result.all_tests_passed}")
```

### Advanced Usage

#### Custom Prompts

```python
# Create custom translator with modified prompts
from jax_agents.prompts import translation_prompts_v2

# Modify prompts
translation_prompts_v2.TRANSLATION_PROMPTS["system"] = """
Your custom system prompt here...
"""

translator = TranslatorAgent()
```

#### Multi-Turn Conversations

```python
# Start conversation
response = translator.multi_turn_conversation(
    initial_prompt="Analyze the complexity of clm_varctl module",
    system_prompt="You are a Fortran code analyzer",
)

# Continue conversation
follow_up = translator.continue_conversation(
    "What are the main dependencies?"
)
```

---

## Configuration

### Config File (`jax-agents/config.yaml`)

```yaml
# LLM Configuration
llm:
  model: "claude-sonnet-4-5"
  temperature: 0.0  # Deterministic
  max_tokens: 48000
  timeout: 600  # seconds

# Paths
paths:
  ctsm_root: "../CTSM"
  jax_ctsm_root: "../jax-ctsm"
  output_dir: "../jax-ctsm/src/jax_ctsm"

# JAX Conversion Patterns
jax_patterns:
  use_immutable_state: true
  use_pure_functions: true
  use_jit: true
  use_vmap: true
  add_type_hints: true
  docstring_style: "google"

# Cost Management
cost_management:
  max_cost_per_module: 10.0  # USD
  warn_on_large_context: true
  context_size_threshold: 50000  # tokens
```

### Environment Variables

Create `jax-agents/.env`:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-xxx

# Optional
LOG_LEVEL=INFO
JAX_ENABLE_X64=True
```

---

## Project Structure

```
clm-ml-jax/
├── src/                              # Translated JAX modules (32,689 LOC)
│   ├── clm_src_main/                 # Core CLM modules (16 files)
│   │   ├── clm_driver.py             # Main driver
│   │   ├── clm_varctl.py             # Control variables
│   │   ├── clm_varpar.py             # Parameters
│   │   ├── PatchType.py              # Patch hierarchy
│   │   └── ...
│   ├── clm_src_biogeophys/           # Biogeophysical processes
│   │   ├── SoilTemperatureMod.py
│   │   ├── SoilStateType.py
│   │   └── ...
│   ├── multilayer_canopy/            # Advanced canopy model (20 modules)
│   │   ├── MLCanopyFluxesMod.py
│   │   ├── MLLeafPhotosynthesisMod.py
│   │   └── ...
│   └── ...
│
├── jax-agents/                       # AI Translation System (2,773 LOC)
│   ├── src/jax_agents/
│   │   ├── base_agent.py             # Base agent with Claude API
│   │   ├── translator.py             # Fortran → JAX translator
│   │   ├── test_agent.py             # Test generator
│   │   ├── repair_agent.py           # Auto-repair agent
│   │   ├── prompts/                  # LLM prompts
│   │   │   ├── translation_prompts_v2.py
│   │   │   ├── test_prompts.py
│   │   │   └── repair_prompts.py
│   │   └── utils/
│   │       └── config_loader.py
│   ├── examples/                     # Example scripts
│   │   ├── translate_with_json.py
│   │   ├── generate_tests.py
│   │   └── repair_agent_example.py
│   ├── run_translation_workflow.sh   # Main workflow script
│   ├── config.yaml                   # Configuration
│   ├── pyproject.toml                # Package config
│   └── .env                          # API keys (gitignored)
│
├── tests/                            # Comprehensive test suite (68 files)
│   ├── conftest.py                   # Pytest fixtures & config
│   ├── clm_src_main/
│   │   ├── test_clm_driver.py
│   │   └── ...
│   ├── clm_src_biogeophys/
│   └── multilayer_canopy/
│
├── docs/                             # Documentation
│   ├── translation_notes/            # Module-by-module notes
│   ├── test_documentation/           # Test coverage docs
│   ├── code_review_report_2024-12.md
│   └── src_main_translation_report.md
│
├── pytest.ini                        # Pytest configuration
├── README.md                         # This file
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

### Test Configuration

The `pytest.ini` configures:
- Test paths: `tests/`
- Strict markers and config
- Verbose output with failure reports
- Custom markers: `slow`, `unit`, `integration`

### Test Fixtures

Common fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def sample_grid():
    """Provides standard grid dimensions."""
    return {'begp': 1, 'endp': 10, 'begc': 1, 'endc': 5, ...}

@pytest.fixture
def sample_arrays():
    """Provides sample JAX arrays."""
    return {'temperature': ..., 'moisture': ..., 'pressure': ...}

@pytest.fixture(autouse=True)
def jax_config():
    """Configures JAX for testing (CPU, float64)."""
    ...
```

### Writing Tests

Example test structure:

```python
import pytest
import jax.numpy as jnp
from clm_src_main.clm_varctl import YourFunction

def test_your_function_basic():
    """Test basic functionality."""
    result = YourFunction(param1=1.0, param2=2.0)
    assert jnp.allclose(result, expected_value)

def test_your_function_edge_cases():
    """Test edge cases."""
    # Test with zeros
    result = YourFunction(param1=0.0, param2=0.0)
    assert jnp.isfinite(result).all()

    # Test with negative values
    result = YourFunction(param1=-1.0, param2=-2.0)
    assert result.shape == expected_shape

@pytest.mark.slow
def test_your_function_performance():
    """Test performance characteristics."""
    import time
    start = time.time()
    YourFunction(large_array)
    duration = time.time() - start
    assert duration < 1.0  # Should complete in < 1 second
```

---

## Development

### Development Setup

```bash
# Install in development mode
cd jax-agents
pip install -e ".[dev]"

# Install pre-commit hooks (if using)
pre-commit install
```

### Code Style

- **Formatting**: Black with 100-character line length
- **Linting**: Ruff
- **Type Checking**: Mypy
- **Docstrings**: Google style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/jax_agents/
```

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

### Contributing Workflow

1. **Fork & Clone**
```bash
git clone https://github.com/your-username/clm-ml-jax.git
cd clm-ml-jax
git checkout -b feature/your-feature
```

2. **Make Changes**
```bash
# Edit files
# Add tests
# Update docs
```

3. **Test**
```bash
pytest tests/ -v
black src/ jax-agents/src/
ruff check src/ jax-agents/src/
```

4. **Commit**
```bash
git add .
git commit -m "Add: Your feature description"
```

5. **Push & PR**
```bash
git push origin feature/your-feature
# Create pull request on GitHub
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Error**: `ANTHROPIC_API_KEY not found in environment`

**Solution**:
```bash
cd jax-agents
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'jax_agents'`

**Solution**:
```bash
cd jax-agents
pip install -e .
```

#### 3. JAX Float64 Issues

**Error**: Numerical precision errors

**Solution**:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

Or set environment variable:
```bash
export JAX_ENABLE_X64=True
```

#### 4. Rate Limiting

**Error**: `anthropic.RateLimitError`

**Solution**: The system has built-in retry logic. If persistent:
- Check your API tier at console.anthropic.com
- Reduce `max_tokens` in config.yaml
- Add delays between requests

#### 5. Test Failures After Translation

**Solution**: Use the RepairAgent:
```bash
./run_translation_workflow.sh --repair --module YourModule
```

#### 6. Out of Memory

**Error**: OOM when running large models

**Solution**:
```python
# In your code
jax.config.update("jax_platform_name", "cpu")  # Use CPU instead of GPU
# Or reduce batch sizes in tests
```

### Debugging Tips

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in config.yaml:
```yaml
logging:
  level: "DEBUG"
```

#### Check Token Usage

```python
from jax_agents import TranslatorAgent

translator = TranslatorAgent()
# ... do work ...
cost = translator.get_cost_estimate()
print(f"Input: {cost['input_tokens']:,} tokens (${cost['input_cost_usd']:.2f})")
print(f"Output: {cost['output_tokens']:,} tokens (${cost['output_cost_usd']:.2f})")
print(f"Total: ${cost['total_cost_usd']:.2f}")
```

#### View Logs

```bash
# Agent logs
ls -ltr jax-agents/logs/

# Latest translator log
tail -f jax-agents/logs/translator_*.log
```

---

## Contributing

We welcome contributions! Please see our contributing guidelines:

### How to Contribute

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

## Acknowledgments

### Project Team

- **CTSM-JAX Team**: Core developers and maintainers

### Technologies

- **[JAX](https://github.com/google/jax)**: Google's numerical computing library
- **[Anthropic Claude](https://www.anthropic.com)**: LLM for code translation
- **[Community Land Model](https://www.cesm.ucar.edu/models/cesm2/land/)**: Original Fortran source
- **[pytest](https://pytest.org)**: Testing framework

### Inspiration

This project demonstrates the power of LLM-assisted code translation for scientific computing, maintaining the rigorous standards required for climate modeling while modernizing the codebase.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{clm_ml_jax,
  title={CLM-ML-JAX: AI-Powered Fortran to JAX Translation System},
  author={CTSM-JAX Team},
  year={2024},
  url={https://github.com/AyaLahlou/clm-ml-jax}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/AyaLahlou/clm-ml-jax/issues)
- **Documentation**: See `docs/` directory
- **Examples**: See `jax-agents/examples/`

---

## Project Status

**Status**: Active Development 🚀

- ✅ Core translation system complete
- ✅ Test generation functional
- ✅ Auto-repair system operational
- ✅ 76 modules translated (32,689 LOC)
- ✅ 68 test files generated
- 🔄 Ongoing: Multilayer canopy system refinement
- 🔄 Ongoing: Performance optimization
- 📅 Planned: Parallel translation support
- 📅 Planned: Web interface for translation monitoring

---

**Last Updated**: 2026-01-20

**Version**: 0.1.0 (Alpha)

---

⭐ **Star this repo** if you find it useful!

🐛 **Report issues** to help us improve!

🤝 **Contribute** to advance scientific computing!
