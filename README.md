# Incremental Sparse Linear Algebra

Incremental linear algebra. Many problems in the area of Social Network Analysis and “Big Data” can be expressed in the language of linear algebra. A library for incremental sparse linear algebra would allow for faster computations on changing real world graphs. I would like to create a proof of concept by starting with sparse matrix - dense vector multiplication. My first goal is to characterize the typical changes a real world graph would experience over time. This will guide my implementation.

## Critical Path

1. Find real-world dynamic graph
  - [ ] Appears that real-world dynamic graphs are hard to get
  - [ ] Perhaps generate my own dynamic graph
2. Determine which correct structure for sparse matrix
  - [X] Structure choice depends on type of updates to matrix
  - [X] Block matrix layout is probably easiest and most flexible.  Block
    structure suffers if there are few off diagnol entries. A sparse matrix of
    sparse matrices would probably be better. I'm not sure how to get that
    working in Julia though.
3. Implement sparse matrix - dense vector multiply
  - Most likely in Julia
  1. [X] Start with just changing of nonzero values
  2. [X] Add changing of sparsity pattern
  3. [ ] Add increasing matrix size
  4. [ ] Handle decreasing matrix size
  5. [X] Support mat-mat mult
  6. [X] Support mat-vec mult
  7. [X] Support incremental binary elementwise operations
4. [ ] Profile performance

## If there is extra time

1. [ ] Create macro to write complicated linear algebraic expressions
2. [ ] Support complex expressions that use the same variable multiple times
