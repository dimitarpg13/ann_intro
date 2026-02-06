# Inverted File Index with Flat Quantizer (IVF-Flat) in FAISS

## 1. Introduction

The Inverted File Index with Flat Quantizer (IVF-Flat) is one of the most fundamental approximate nearest neighbor (ANN) search algorithms implemented in Facebook AI Similarity Search (FAISS). It combines **Voronoi partitioning** of the vector space with **exact distance computation** within selected partitions, achieving a practical balance between search speed and recall accuracy.

Unlike pure brute-force search, IVF-Flat narrows the search scope through spatial partitioning, and unlike heavily quantized methods (e.g., IVF-PQ), it preserves full vector fidelity within each partition — hence the name "Flat" quantizer.

---

## 2. High-Level Architecture

```mermaid
graph TD
    A["Vector Database<br/>(N vectors, d dimensions)"] --> B["K-Means Training<br/>(nlist centroids)"]
    B --> C["Voronoi Partitioning"]
    C --> D["Inverted Lists"]
    
    D --> D1["List 0: vectors assigned<br/>to centroid 0"]
    D --> D2["List 1: vectors assigned<br/>to centroid 1"]
    D --> D3["List 2: vectors assigned<br/>to centroid 2"]
    D --> Dk["List k: vectors assigned<br/>to centroid k"]
    
    Q["Query Vector q"] --> E["Coarse Quantizer"]
    E --> F["Select nprobe<br/>nearest centroids"]
    F --> G["Exhaustive Search<br/>within selected lists"]
    G --> H["Top-K Results"]

    style A fill:#2d3748,stroke:#4a90d9,color:#fff
    style B fill:#1a365d,stroke:#63b3ed,color:#fff
    style C fill:#1a365d,stroke:#63b3ed,color:#fff
    style Q fill:#744210,stroke:#ecc94b,color:#fff
    style E fill:#744210,stroke:#ecc94b,color:#fff
    style H fill:#22543d,stroke:#68d391,color:#fff
```

The architecture consists of two distinct phases: **index construction** (offline) and **query processing** (online).

---

## 3. Algorithmic Foundations

### 3.1 Coarse Quantizer: K-Means Clustering

The first stage of IVF-Flat employs Lloyd's algorithm (K-Means) to partition the vector space into `nlist` Voronoi cells.

**Objective Function (K-Means):**

$$\underset{C}{\text{minimize}} \sum_{i=1}^{N} \min_{j \in \{1, \ldots, \text{nlist}\}} \| x_i - c_j \|^2$$

where $C = \{c_1, c_2, \ldots, c_{\text{nlist}}\}$ are the centroids and $x_i$ are the database vectors.

```mermaid
flowchart LR
    subgraph Training["K-Means Training Loop"]
        direction TB
        INIT["Initialize nlist centroids<br/>(random or K-Means++)"] --> ASSIGN["Assignment Step:<br/>Assign each vector to<br/>nearest centroid"]
        ASSIGN --> UPDATE["Update Step:<br/>Recompute centroids as<br/>mean of assigned vectors"]
        UPDATE --> CONV{"Converged?"}
        CONV -- No --> ASSIGN
        CONV -- Yes --> DONE["Final Centroids"]
    end

    style INIT fill:#2d3748,stroke:#a0aec0,color:#fff
    style ASSIGN fill:#2c5282,stroke:#63b3ed,color:#fff
    style UPDATE fill:#2c5282,stroke:#63b3ed,color:#fff
    style CONV fill:#744210,stroke:#ecc94b,color:#fff
    style DONE fill:#22543d,stroke:#68d391,color:#fff
```

**Key properties of the coarse quantizer:**

- It defines a **quantization function** $q(x) = \arg\min_{j} \|x - c_j\|^2$ that maps each vector to its nearest centroid.
- The resulting Voronoi cells are **convex polytopes** — every point within a cell is closer to that cell's centroid than to any other.
- The quantizer is "flat" because the original vectors are stored **without any compression** inside each inverted list.

### 3.2 Inverted File Structure

After training, each database vector $x_i$ is assigned to the inverted list corresponding to its nearest centroid:

$$\text{list}_j = \{ (id_i, x_i) \mid q(x_i) = j \}$$

```mermaid
graph LR
    subgraph InvertedIndex["Inverted File Index"]
        direction TB
        C0["Centroid 0"] --> L0["ID₁, Vec₁<br/>ID₅, Vec₅<br/>ID₁₂, Vec₁₂"]
        C1["Centroid 1"] --> L1["ID₂, Vec₂<br/>ID₇, Vec₇"]
        C2["Centroid 2"] --> L2["ID₃, Vec₃<br/>ID₈, Vec₈<br/>ID₉, Vec₉<br/>ID₁₁, Vec₁₁"]
        Ck["Centroid k"] --> Lk["ID₄, Vec₄<br/>ID₆, Vec₆<br/>ID₁₀, Vec₁₀"]
    end

    style C0 fill:#2c5282,stroke:#63b3ed,color:#fff
    style C1 fill:#2c5282,stroke:#63b3ed,color:#fff
    style C2 fill:#2c5282,stroke:#63b3ed,color:#fff
    style Ck fill:#2c5282,stroke:#63b3ed,color:#fff
    style L0 fill:#2d3748,stroke:#a0aec0,color:#fff
    style L1 fill:#2d3748,stroke:#a0aec0,color:#fff
    style L2 fill:#2d3748,stroke:#a0aec0,color:#fff
    style Lk fill:#2d3748,stroke:#a0aec0,color:#fff
```

Each inverted list stores:
- The **vector IDs** (for result mapping)
- The **full original vectors** (for exact distance computation — this is the "Flat" part)

### 3.3 Query Processing: Probe and Scan

```mermaid
sequenceDiagram
    participant Q as Query Vector
    participant CQ as Coarse Quantizer
    participant IL as Inverted Lists
    participant R as Result Heap

    Q->>CQ: Compute distances to all nlist centroids
    CQ->>CQ: Sort centroids by distance
    CQ->>IL: Select top nprobe closest centroids
    
    loop For each of nprobe lists
        IL->>IL: Exhaustive scan of all vectors in list
        IL->>R: Push (distance, id) pairs onto max-heap
    end
    
    R->>R: Extract top-K nearest neighbors
    R->>Q: Return K results
```

**Search procedure:**

1. **Coarse assignment**: Compute distances from query $q$ to all `nlist` centroids → $O(\text{nlist} \cdot d)$
2. **Probe selection**: Select the `nprobe` nearest centroids
3. **Fine search**: For each selected list, compute exact distances between $q$ and every vector in that list
4. **Result aggregation**: Maintain a max-heap of size $K$ across all probed lists

**Total search complexity:**

$$T_{\text{search}} = O(\text{nlist} \cdot d) + O\left(\text{nprobe} \cdot \frac{N}{\text{nlist}} \cdot d\right)$$

The first term is the coarse quantizer cost; the second is the fine scan cost (assuming uniform distribution of vectors across lists).

---

## 4. Theoretical Analysis

### 4.1 The Speed-Recall Tradeoff

The fundamental tradeoff in IVF-Flat is controlled by two parameters:

| Parameter | Effect on Speed | Effect on Recall |
|-----------|----------------|-----------------|
| `nlist` ↑ | Faster (smaller lists to scan) | Lower (more boundary effects) |
| `nlist` ↓ | Slower (larger lists to scan) | Higher (fewer missed neighbors) |
| `nprobe` ↑ | Slower (more lists scanned) | Higher (more coverage) |
| `nprobe` ↓ | Faster (fewer lists scanned) | Lower (more missed neighbors) |

**Boundary condition**: When `nprobe = nlist`, IVF-Flat degenerates to exact brute-force search with 100% recall (plus the overhead of the coarse quantizer pass).

### 4.2 Voronoi Cell Geometry and Recall Loss

Recall loss in IVF-Flat arises from **cell boundary effects**: a query's true nearest neighbor may reside in a Voronoi cell whose centroid is not among the `nprobe` closest to the query.

```mermaid
graph TD
    subgraph VoronoiProblem["Boundary Effect Illustration"]
        direction LR
        A["Query q sits near the<br/>boundary of Cell A and Cell B"] --> B["True NN is in Cell B"]
        A --> C["But Cell B's centroid is<br/>the (nprobe+1)-th closest"]
        C --> D["Result: True NN is MISSED"]
    end

    style A fill:#744210,stroke:#ecc94b,color:#fff
    style B fill:#9b2c2c,stroke:#fc8181,color:#fff
    style D fill:#9b2c2c,stroke:#fc8181,color:#fff
```

The probability of missing a true nearest neighbor depends on:
- The **angular relationship** between the query-to-NN vector and the cell boundary normal
- The **density of centroids** relative to the data distribution
- The **dimensionality** $d$ — in high dimensions, vectors tend to be equidistant (concentration of measure), making centroid discrimination harder

### 4.3 Quantization Error Analysis

For IVF-Flat specifically, the quantization error is **zero at search time** because original vectors are stored intact. However, the coarse quantizer introduces an **assignment error**:

$$\epsilon_{\text{assign}} = P(\text{NN}(q) \notin \bigcup_{j \in S_{\text{nprobe}}} \text{list}_j)$$

This is the probability that the true nearest neighbor is not found within the probed lists. Increasing `nprobe` monotonically decreases $\epsilon_{\text{assign}}$.

### 4.4 Complexity Comparison

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Training (K-Means) | $O(I \cdot N \cdot \text{nlist} \cdot d)$ | $O(\text{nlist} \cdot d)$ |
| Index construction | $O(N \cdot \text{nlist} \cdot d)$ | $O(N \cdot d + \text{nlist} \cdot d)$ |
| Single query | $O(\text{nlist} \cdot d + \text{nprobe} \cdot \frac{N}{\text{nlist}} \cdot d)$ | $O(K)$ for heap |

where $I$ is the number of K-Means iterations.

---

## 5. FAISS Implementation Details

### 5.1 Core API

```python
import faiss
import numpy as np

d = 128          # vector dimension
nlist = 256      # number of Voronoi cells
nprobe = 16      # number of cells to probe at search time

# Build the index
quantizer = faiss.IndexFlatL2(d)           # Coarse quantizer (flat = exact)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Train on representative data
training_data = np.random.random((100000, d)).astype('float32')
index.train(training_data)

# Add vectors
database = np.random.random((1000000, d)).astype('float32')
index.add(database)

# Search
index.nprobe = nprobe
queries = np.random.random((100, d)).astype('float32')
distances, indices = index.search(queries, k=10)
```

### 5.2 Internal Architecture in FAISS

```mermaid
classDiagram
    class IndexIVF {
        <<abstract>>
        +quantizer: Index
        +nlist: int
        +nprobe: int
        +invlists: InvertedLists
        +train(x)
        +add(x)
        +search(x, k)
        +train_residual(x)*
        +encode_vectors(x)*
    }

    class IndexIVFFlat {
        +code_size: int
        +encode_vectors(x)
        +sa_decode(codes)
        +search_preassigned(x, k, keys)
    }

    class IndexFlatL2 {
        +xb: float array
        +search(x, k)
        +add(x)
    }

    class InvertedLists {
        <<abstract>>
        +list_size(list_no)
        +get_codes(list_no)
        +get_ids(list_no)
        +add_entries(list_no, codes, ids)
    }

    class ArrayInvertedLists {
        +codes: vector of vectors
        +ids: vector of vectors
    }

    IndexIVF <|-- IndexIVFFlat
    IndexIVF o-- IndexFlatL2 : quantizer
    IndexIVF o-- InvertedLists
    InvertedLists <|-- ArrayInvertedLists

    note for IndexIVFFlat "Stores full vectors\n(no compression)\ncode_size = d * sizeof(float)"
```

### 5.3 Key Implementation Optimizations in FAISS

FAISS implements several critical optimizations:

**BLAS-accelerated distance computation**: The fine search within inverted lists uses optimized matrix-matrix multiplication (via BLAS libraries like MKL/OpenBLAS) to compute distances in batch, exploiting the identity:

$$\|q - x_i\|^2 = \|q\|^2 - 2 \langle q, x_i \rangle + \|x_i\|^2$$

where the inner product $\langle q, x_i \rangle$ is computed via `sgemm` for all vectors in a list simultaneously.

**Multi-threaded search**: Queries are parallelized across OpenMP threads, with each thread processing a subset of query vectors independently.

**Precomputed norms**: Vector norms $\|x_i\|^2$ can be precomputed and stored alongside the inverted lists, reducing per-query computation.

**Memory layout**: Vectors within each inverted list are stored contiguously in memory, maximizing cache locality during sequential scans.

---

## 6. Relationship to Other ANN Algorithms

### 6.1 Taxonomy of ANN Methods

```mermaid
graph TD
    ANN["Approximate Nearest<br/>Neighbor Methods"] --> HASH["Hashing-Based"]
    ANN --> TREE["Tree-Based"]
    ANN --> GRAPH["Graph-Based"]
    ANN --> QUANT["Quantization-Based"]
    
    HASH --> LSH["Locality-Sensitive<br/>Hashing (LSH)"]
    
    TREE --> KDT["KD-Trees"]
    TREE --> BT["Ball Trees"]
    TREE --> ANNOY["Annoy<br/>(Random Projection Trees)"]
    
    GRAPH --> HNSW["HNSW"]
    GRAPH --> NSG["NSG / DiskANN"]
    GRAPH --> VAMANA["Vamana"]
    
    QUANT --> FLAT["Flat (Exact)"]
    QUANT --> IVF["IVF Family"]
    QUANT --> PQ["Product Quantization"]
    
    IVF --> IVFFLAT["IVF-Flat ⭐"]
    IVF --> IVFPQ["IVF-PQ"]
    IVF --> IVFSQ["IVF-SQ<br/>(Scalar Quantization)"]
    IVF --> IVFPQR["IVF-PQR<br/>(PQ + Refine)"]
    
    style IVFFLAT fill:#22543d,stroke:#68d391,color:#fff,stroke-width:3px
    style ANN fill:#2d3748,stroke:#a0aec0,color:#fff
    style QUANT fill:#1a365d,stroke:#63b3ed,color:#fff
    style IVF fill:#2c5282,stroke:#63b3ed,color:#fff
    style GRAPH fill:#553c9a,stroke:#b794f4,color:#fff
    style HNSW fill:#553c9a,stroke:#b794f4,color:#fff
```

### 6.2 IVF-Flat vs. Other IVF Variants

```mermaid
graph LR
    subgraph IVFFamily["IVF Family Spectrum"]
        direction TB
        IVFFLAT["IVF-Flat<br/>───────────<br/>✅ Exact vectors stored<br/>✅ No quantization error<br/>❌ High memory usage<br/>⏱ Moderate speed"]
        
        IVFSQ["IVF-SQ8<br/>───────────<br/>⚠️ Scalar quantized (8-bit)<br/>⚠️ Small quantization error<br/>✅ 4× memory reduction<br/>⏱ Faster scans"]
        
        IVFPQ["IVF-PQ<br/>───────────<br/>⚠️ Product quantized<br/>⚠️ Larger quantization error<br/>✅ 16-64× memory reduction<br/>⏱ Fastest scans"]
    end
    
    IVFFLAT -.->|"Add scalar<br/>quantization"| IVFSQ
    IVFSQ -.->|"Add product<br/>quantization"| IVFPQ

    style IVFFLAT fill:#22543d,stroke:#68d391,color:#fff
    style IVFSQ fill:#744210,stroke:#ecc94b,color:#fff
    style IVFPQ fill:#742a2a,stroke:#fc8181,color:#fff
```

| Variant | Memory per Vector | Quantization Error | Best For |
|---------|------------------|--------------------|----------|
| **IVF-Flat** | $4d$ bytes (float32) | None | High-recall requirements, datasets that fit in RAM |
| **IVF-SQ8** | $d$ bytes (uint8) | Small | Moderate memory savings with minimal recall loss |
| **IVF-PQ** | $M$ bytes ($M$ subquantizers) | Moderate | Large-scale datasets exceeding RAM |
| **IVF-PQR** | $M + M'$ bytes | Small (refined) | High recall with PQ-level memory |

### 6.3 IVF-Flat vs. HNSW

```mermaid
graph TD
    subgraph Comparison["IVF-Flat vs HNSW"]
        direction LR
        subgraph IVF_Approach["IVF-Flat"]
            direction TB
            I1["Partition-based"]
            I2["Tunable via nlist, nprobe"]
            I3["Supports easy add/remove"]
            I4["Lower memory overhead"]
            I5["Batch-friendly (BLAS)"]
            I6["Moderate recall at speed"]
        end
        
        subgraph HNSW_Approach["HNSW"]
            direction TB
            H1["Graph-based"]
            H2["Tunable via M, efSearch"]
            H3["Difficult to delete vectors"]
            H4["Higher memory overhead"]
            H5["Sequential memory access"]
            H6["Higher recall at speed"]
        end
    end

    style I1 fill:#22543d,stroke:#68d391,color:#fff
    style I2 fill:#22543d,stroke:#68d391,color:#fff
    style I3 fill:#22543d,stroke:#68d391,color:#fff
    style I4 fill:#22543d,stroke:#68d391,color:#fff
    style I5 fill:#22543d,stroke:#68d391,color:#fff
    style I6 fill:#22543d,stroke:#68d391,color:#fff
    style H1 fill:#553c9a,stroke:#b794f4,color:#fff
    style H2 fill:#553c9a,stroke:#b794f4,color:#fff
    style H3 fill:#553c9a,stroke:#b794f4,color:#fff
    style H4 fill:#553c9a,stroke:#b794f4,color:#fff
    style H5 fill:#553c9a,stroke:#b794f4,color:#fff
    style H6 fill:#553c9a,stroke:#b794f4,color:#fff
```

**When to prefer IVF-Flat over HNSW:**
- The dataset is updated frequently (IVF supports efficient add/remove)
- Batch query throughput matters more than single-query latency (BLAS acceleration)
- Memory is constrained (HNSW requires additional graph storage: ~$4 \cdot M \cdot N$ bytes for edge lists)
- GPU acceleration is needed (FAISS IVF-Flat has native GPU support)

**When to prefer HNSW over IVF-Flat:**
- Single-query latency is critical
- The dataset is relatively static
- Very high recall (>99%) is needed at moderate speed
- The dataset has complex, non-uniform clustering structure

### 6.4 IVF-Flat vs. Annoy

Annoy (Approximate Nearest Neighbors Oh Yeah) uses **random projection trees** — an ensemble of binary space partition trees built from random hyperplanes. Compared to IVF-Flat:

- Annoy is **memory-mapped** by design, making it ideal for read-only, disk-based workloads
- IVF-Flat provides **better throughput** for batch queries via BLAS
- Annoy's index is **immutable** after construction; IVF-Flat supports dynamic insertion
- IVF-Flat generally achieves **higher recall** at equivalent query times for high-dimensional data

### 6.5 IVF-Flat vs. LSH

Locality-Sensitive Hashing provides **theoretical guarantees** (sub-linear query time with bounded approximation ratio) that IVF-Flat lacks. However:

- LSH requires many hash tables for high recall, leading to **significant memory overhead**
- IVF-Flat is **far more practical** — empirically achieving better recall-speed tradeoffs
- LSH guarantees hold for worst-case data; IVF-Flat exploits **data structure** (clustering) for efficiency

---

## 7. The Composite Index Pipeline

FAISS supports combining IVF-Flat with preprocessing and post-processing steps:

```mermaid
flowchart LR
    RAW["Raw Vectors<br/>(d dimensions)"] --> PCA["PCA / OPQ<br/>Dimensionality<br/>Reduction"]
    PCA --> NORM["L2 Normalization<br/>(for cosine similarity)"]
    NORM --> IVF["IVF-Flat<br/>Index"]
    IVF --> REFINE["Optional Refine<br/>(re-rank with<br/>exact distances)"]
    REFINE --> RESULTS["Final Top-K"]

    style RAW fill:#2d3748,stroke:#a0aec0,color:#fff
    style PCA fill:#553c9a,stroke:#b794f4,color:#fff
    style NORM fill:#553c9a,stroke:#b794f4,color:#fff
    style IVF fill:#22543d,stroke:#68d391,color:#fff
    style REFINE fill:#744210,stroke:#ecc94b,color:#fff
    style RESULTS fill:#1a365d,stroke:#63b3ed,color:#fff
```

**FAISS index factory string**: `"PCA64,IVF256,Flat"` — reduces to 64 dimensions via PCA, then builds IVF-Flat with 256 cells.

---

## 8. Parameter Tuning Guidelines

### 8.1 Choosing `nlist`

The number of Voronoi cells should scale with the dataset size:

$$\text{nlist} \approx \sqrt{N} \quad \text{(rule of thumb)}$$

For very large datasets ($N > 10^7$), values in the range $4\sqrt{N}$ to $16\sqrt{N}$ are common.

### 8.2 Choosing `nprobe`

```mermaid
graph LR
    subgraph NprobeEffect["nprobe Effect on Recall"]
        direction TB
        NP1["nprobe = 1<br/>~40-60% recall<br/>Fastest"] --> NP8["nprobe = 8<br/>~80-90% recall<br/>Fast"]
        NP8 --> NP32["nprobe = 32<br/>~95-98% recall<br/>Moderate"]
        NP32 --> NP128["nprobe = 128<br/>~99%+ recall<br/>Slower"]
        NP128 --> NPALL["nprobe = nlist<br/>100% recall<br/>Brute-force"]
    end

    style NP1 fill:#9b2c2c,stroke:#fc8181,color:#fff
    style NP8 fill:#744210,stroke:#ecc94b,color:#fff
    style NP32 fill:#22543d,stroke:#68d391,color:#fff
    style NP128 fill:#2c5282,stroke:#63b3ed,color:#fff
    style NPALL fill:#553c9a,stroke:#b794f4,color:#fff
```

A typical starting point is `nprobe = nlist / 16` or `nprobe = nlist / 8`, then tuning based on empirical recall measurements.

### 8.3 Training Data Requirements

K-Means training requires a representative sample of the data distribution. FAISS recommends:

$$N_{\text{train}} \geq 30 \cdot \text{nlist}$$

Using too few training vectors leads to poor centroid placement and degraded recall.

---

## 9. GPU Acceleration

FAISS provides native GPU support for IVF-Flat via `GpuIndexIVFFlat`:

```python
import faiss

res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
gpu_index.train(training_data)
gpu_index.add(database)

gpu_index.nprobe = nprobe
distances, indices = gpu_index.search(queries, k=10)
```

GPU acceleration provides **10-100× speedup** over CPU for large batch queries by parallelizing both the coarse quantizer and the fine scan stages across thousands of GPU threads.

---

## 10. Summary Decision Flowchart

```mermaid
flowchart TD
    START["Need ANN Search?"] --> SIZE{"Dataset size?"}
    
    SIZE -->|"< 50K vectors"| FLAT["Use IndexFlatL2<br/>(Exact brute-force)"]
    SIZE -->|"50K - 10M"| MEM{"Fits in RAM<br/>with full vectors?"}
    SIZE -->|"> 10M"| LARGE["Consider IVF-PQ<br/>or HNSW"]
    
    MEM -->|Yes| RECALL{"Recall requirement?"}
    MEM -->|No| COMPRESS["Use IVF-PQ or IVF-SQ"]
    
    RECALL -->|"> 99%"| HNSW_USE["Consider HNSW<br/>or IVF-Flat<br/>(high nprobe)"]
    RECALL -->|"90-99%"| IVFFLAT_USE["IVF-Flat ⭐<br/>Best balance of<br/>speed and accuracy"]
    RECALL -->|"< 90%"| IVFFLAT_FAST["IVF-Flat<br/>(low nprobe)<br/>or IVF-PQ"]
    
    style START fill:#2d3748,stroke:#a0aec0,color:#fff
    style IVFFLAT_USE fill:#22543d,stroke:#68d391,color:#fff,stroke-width:3px
    style FLAT fill:#1a365d,stroke:#63b3ed,color:#fff
    style HNSW_USE fill:#553c9a,stroke:#b794f4,color:#fff
```

---

## 11. Key Takeaways

IVF-Flat occupies a **sweet spot** in the ANN algorithm landscape. It introduces no quantization error (preserving exact distances), provides tunable speed-recall tradeoffs via `nlist` and `nprobe`, supports dynamic updates and GPU acceleration, and benefits from BLAS-optimized batch processing. Its primary limitation is memory — it stores full vectors, making it impractical for billion-scale datasets without combining it with quantization (IVF-PQ) or dimensionality reduction (PCA). For datasets in the millions where RAM is available and high recall matters, IVF-Flat remains one of the most reliable and well-understood choices in the FAISS toolkit.

---

*Generated for reference on ANN algorithms in FAISS — February 2026*
