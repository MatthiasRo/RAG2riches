# Credits and Acknowledgments

RAG2riches is built on the excellent work of the open-source community. We gratefully acknowledge the following projects and their contributors.

## Core Dependencies

### Data and Configuration

- **[Pydantic](https://docs.pydantic.dev/)** — Data validation and settings management using Python type annotations. Used for all configuration schemas and type-safe data structures throughout RAG2riches.
  - License: MIT
  - Authors: Samuel Colvin, Eric Jolly, and contributors

- **[Pandas](https://pandas.pydata.org/)** — Flexible and powerful data analysis library. Used for DataFrame-based chunk management and export.
  - License: BSD 3-Clause
  - Authors: Wes McKinney, The Pandas Development Team

- **[PyArrow](https://arrow.apache.org/docs/python/)** — Efficient columnar memory format. Underlying data format for efficient storage and processing.
  - License: Apache License 2.0
  - Authors: The Apache Arrow Project

### LLM and Embedding Interfaces (v0.2+)

- **[LiteLLM](https://docs.litellm.ai/)** — Unified interface for LLM and embedding APIs across providers (OpenAI, Gemini, Claude, Ollama, etc.).
  - License: MIT
  - Authors: Ishaan Jaffer, LiteLLM Contributors
  - This abstraction layer is critical to RAG2riches's multi-provider support

### Vector Storage (v0.2+)

- **[LanceDB](https://lancedb.com/)** — Vector database with native metadata filtering. Essential for metadata-pre-filtered retrieval, the core distinctive feature of RAG2riches.
  - License: Apache License 2.0 and Elastic License 2.0
  - Authors: The LanceDB Project
  - Alternative backends (Chroma, Qdrant) will be supported via abstract interface

### Progress and Logging

- **[tqdm](https://tqdm.github.io/)** — Progress bars for long-running operations.
  - License: MIT
  - Authors: Noam Raphael, tqdm contributors

- **[loguru](https://loguru.readthedocs.io/)** — Modern logging library with intuitive API.
  - License: MIT
  - Authors: Delgan, loguru contributors

### PDF Processing

- **[PyPDF](https://github.com/py-pdf/pypdf)** — PDF text extraction.
  - License: BSD 3-Clause
  - Authors: PyPDF Contributors

### Development and Testing

- **[pytest](https://pytest.org/)** — Testing framework.
  - License: MIT
  - Authors: Holger Krekel, pytest contributors

- **[ruff](https://github.com/astral-sh/ruff)** — Fast Python linter and formatter.
  - License: MIT
  - Authors: Charlie Marsh, Astral contributors

### UI (v0.4+)

- **[Streamlit](https://streamlit.io/)** — Build interactive data apps.
  - License: Elastic License 2.0 + Server Side Public License
  - Authors: Adrien Treuille, Amanda Kelly, Thorsten Ruotolo, and the Streamlit team

## Inspiration and Related Work

The design of RAG2riches is informed by:

- The **RAG (Retrieval-Augmented Generation)** literature, particularly Karpukhin et al. (2020) and Lewis et al. (2020)
- **Comparative methodology** in social science research, especially methods for cross-national, cross-temporal, and cross-organizational analysis
- Existing open-source RAG libraries (LangChain, LlamaIndex) that demonstrate modular architecture principles
- The **LanceDB** team's pioneering work on metadata-native vector storage

## Community and Contributors

RAG2riches benefits from:

- Early testers and feedback providers from the social science research community
- The broader Python data science ecosystem (NumPy, Scipy, scikit-learn communities)
- Open-source maintainers who answer questions and provide support

## Citation

If you use RAG2riches in research, please cite:

Roesti, Matthias, From RAGs to (feature) Riches - An Efficient Pipeline for Exploratory Text Analysis (June 29, 2025). Available at SSRN: https://ssrn.com/abstract=5331899 or http://dx.doi.org/10.2139/ssrn.5331899

```bibtex
@software{roesti2024rag2riches,
  title = {RAG2riches: Comparative Retrieval-Augmented Generation for Social Science},
  author = {Roesti, Matthias},
  year = {2024},
  url = {https://github.com/MatthiasRo/RAG2riches}
}

@article{roesti2025rags,
  title = {From RAGs to (feature) Riches - An Efficient Pipeline for Exploratory Text Analysis},
  author = {Roesti, Matthias},
  year = {2025},
  month = jun,
  url = {https://ssrn.com/abstract=5331899},
  doi = {10.2139/ssrn.5331899},
  note = {Available at SSRN}
}
```

## How to Contribute

Contributions are welcome! Please consider:

1. **Using** RAG2riches and providing feedback
2. **Testing** with your own data and workflows
3. **Reporting** bugs and suggesting features via GitHub Issues
4. **Contributing** code via pull requests
5. **Sharing** examples and use cases
6. **Citing** RAG2riches when you use it in research
7. **Improving** documentation and examples

See the main [README](../README.md) for contribution guidelines.

## License

RAG2riches is licensed under the **Apache License 2.0**, which permits free use, modification, and distribution in both open-source and commercial projects. See [LICENSE](../LICENSE) for details.

## Disclaimer

RAG2riches uses third-party services for LLM and embedding generation (optional). These services have their own terms of service and privacy policies. Users are responsible for complying with the terms of the services they use. See documentation for each provider's policies.

