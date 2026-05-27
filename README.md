## Overview

The repository is organized around hands-on labs. The main focus is to show how graph-structured data can be represented, embedded, trained, evaluated, and visualized using both classical knowledge graph embedding models and graph neural network approaches.

Topics covered include:

- Knowledge graphs and triples
- Knowledge graph embedding models
- Translational models such as TransE, TransH, TransR, and RotatE
- Factorization models such as DistMult, ComplEx, and RESCAL
- Link prediction and record linkage evaluation
- Random walks and Node2Vec intuition
- Inductive representation learning with GraphSAGE
- Interactive geometric visualizations of embedding models

## Repository Structure

```text
.
├── solutions_lab_gnn.ipynb              # Graph neural network workshop notebook
├── lab.ipynb                       # Main lab notebook
├── graph.py                        # Graph utility class for triples and structural features
├── utils.py                        # Utility functions for structural features and model saving/loading
├── kg_translation_geometry.py      # Plotly visualizations for translational KGE models
├── assets/                         # Images and supporting assets
├── checkpoints/                    # Saved model checkpoints
├── LICENSE                         # MIT license
└── README.md
```

## Main Concepts

### Knowledge Graphs

A knowledge graph represents information as triples:

```text
(head, relation, tail)
```

For example:

```text
(John Smith, appears_in, Census 1880)
(John Smith, lives_in, Eivissa)
(Mary Smith, parent_of, John Smith)
```
These triples can be used to build a graph where entities are nodes and relations are typed edges.

### Knowledge Graph Embeddings

Knowledge graph embedding models learn vector representations for entities and relations. These embeddings are trained so that true triples receive higher scores than corrupted or false triples.

A general scoring function is:

```text
f(h, r, t)
```

where:

- `h` is the head entity embedding,
- `r` is the relation embedding,
- `t` is the tail entity embedding.

The score measures how plausible the triple is.

### Translational Models

Translational models interpret relations as geometric transformations.

Examples:

- **TransE**: models relations as translations, `h + r ≈ t`.
- **TransH**: projects entities onto relation-specific hyperplanes.
- **TransR**: projects entities into relation-specific spaces.
- **RotatE**: represents relations as rotations in the complex plane.

The repository includes interactive Plotly visualizations to make these geometric intuitions easier to understand.

### Factorization Models

Factorization models score triples using multiplicative interactions.

Examples:

- **RESCAL**: uses a full relation-specific matrix.
- **DistMult**: simplifies RESCAL using diagonal relation matrices.
- **ComplEx**: extends DistMult with complex-valued embeddings to model asymmetric relations.

### Random Walks and Node2Vec

Node2Vec learns node embeddings by generating random walks over the graph. These walks are treated like sentences, and nodes are treated like words. The method then applies a Word2Vec-style objective to learn embeddings for nodes that appear in similar graph contexts.

### GraphSAGE and Inductive Learning

GraphSAGE replaces static embedding lookup tables with a neural encoder that computes node representations from node features and neighborhoods:

```text
z_v = GraphSAGE(x_v, N(v))
```

This makes the method inductive: it can compute representations for new nodes if their features and graph connections are available.

## Installation

Create a Python environment and install the main dependencies:

    torch_cluster.__version__
    '1.6.3+pt25cu121'

    torch_geometric.__version__
    '2.7.0'

    torch.__version__
    '2.5.1+cu121'

    torch_scatter
    '2.1.2+pt25cu121'

The code has been tested with this versions based on the CUDA version. Depending on your CUDA and PyTorch version, PyTorch Geometric may require a version-specific installation. Check the official PyTorch Geometric installation instructions if you encounter issues with packages such as `torch-scatter`, `torch-sparse`, or `torch-cluster`.

## Usage

Clone the repository:

```bash
git clone https://github.com/Bycarkos/Labs.git
cd Labs
```

## Visualization Modules

The repository includes reusable Plotly visualization modules.

### Translational models

```python
from kg_translation_geometry import plot_translation_models_clean

fig = plot_translation_models_clean()
fig.show()
```

## Utility Functions

The `utils.py` module includes helper functions for:

- saving PyTorch or PyTorch Geometric models,
- loading model checkpoints into existing model instances.

Example:

```python
from utils import save_model, load_model

save_model(model, "checkpoints/model.pt", optimizer=optimizer, epoch=10)

model, optimizer, checkpoint = load_model(
    model=model,
    path="checkpoints/model.pt",
    optimizer=optimizer,
    map_location="cpu",
)
```

## Graph Utilities

The `graph.py` module provides a `Graph` class for working with triples. It supports:

- tokenizing raw triples into integer IDs,
- detokenizing IDs back to raw triples,
- updating the graph with new triples,
- computing structural node embeddings,
- converting triples into heterogeneous edge dictionaries.

## Evaluation Metrics

The notebooks use common link prediction metrics:

- **Mean Rank (MR)**: average rank of the correct entity.
- **Mean Reciprocal Rank (MRR)**: average inverse rank of the correct entity.
- **Hits@K**: proportion of queries where the correct entity appears in the top K predictions.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Notes

This repository is designed as an educational resource. The code favors clarity and interpretability over maximum performance. It is suitable for workshops, tutorials, and exploratory experiments on graph learning and knowledge graph embeddings.
