# Quantule Mapper Wiki

Welcome to the **Quantule Mapper** project wiki! This is your guide to understanding and using the GPU/JAX stack for discovering and validating stable localized field structures in nonlinear complex-field PDEs.

## What is Quantule Mapper?

Quantule Mapper is an advanced computational framework designed to:
- **Discover** stable localized field structures (dissipative solitons) in nonlinear complex-field partial differential equations
- **Validate** these structures through long-time integration
- **Hunt parameters** using evolutionary algorithms with dual solver support
- **Analyze results** with spectral, topological, and conformal-tensor validation pipelines

## Key Features

- 🚀 **GPU-Accelerated**: Leverages JAX and CuPy for high-performance computing
- 🔬 **Dual Solvers**: Flexible implementation with both CuPy and JAX backends
- 🎯 **Evolutionary Parameter Search**: Intelligent parameter hunting and optimization
- 📊 **Multi-faceted Validation**: Spectral, topological, and conformal-tensor analysis
- 🔄 **Long-Time Validation**: Robust integration schemes for stability assessment

## Quick Navigation

- [[Getting Started]] - Installation and setup guide
- [[Core Concepts]] - Understanding solitons and the mathematical framework
- [[API Reference]] - Detailed documentation of key modules
- [[Tutorials]] - Step-by-step examples and workflows
- [[Configuration]] - Parameter settings and solver options
- [[Troubleshooting]] - Common issues and solutions
- [[Contributing]] - Guidelines for contributing to the project

## Project Structure

```
quantule_mapper/
├── solvers/           # JAX and CuPy solver implementations
├── parameter_hunter/  # Evolutionary algorithm components
├── validation/        # Spectral and topological validation
├── utils/             # Utility functions
└── examples/          # Example notebooks and scripts
```

## Getting Help

- Check the [[Troubleshooting]] page for common issues
- Review [[Tutorials]] for usage examples
- Consult the [[API Reference]] for function documentation
- Open an issue on the main repository

---

Last updated: 2026-10-05
