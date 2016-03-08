# Incremental Sparse Linear Algebra

Incremental linear algebra. Many problems in the area of Social Network Analysis and “Big Data” can be expressed in the language of linear algebra. A library for incremental sparse linear algebra would allow for faster computations on changing real world graphs. I would like to create a proof of concept by starting with sparse matrix - dense vector multiplication. My first goal is to characterize the typical changes a real world graph would experience over time. This will guide my implementation.

## Critical Path

1. Find real-world dynamic graph
  - Appears that real-world dynamic graphs are hard to get
  - Perhaps generate my own dynamic graph
2. Determine which correct structure for sparse matrix
  - Structure choice depends on type of updates to matrix
  - Block matrix layout is probably easiest and most flexible
3. Implement sparse matrix - dense vector multiply
  - Most likely in Julia
4. Profile performance

## If there is extra time

5. Implement sparse matrix - sparse matrix multiply
  - Might actually be simpler than sparse matrix - dense vector
6. Create macro to write complicated linear algebraic expressions
