# Flat Index Implementation in FAISS

## Table of Contents
1. [Introduction](#introduction)
2. [Flat Index Fundamentals](#flat-index-fundamentals)
3. [Class Hierarchy](#class-hierarchy)
4. [Core Data Structures](#core-data-structures)
5. [Add Operation](#add-operation)
6. [Search Operation](#search-operation)
7. [Distance Computation](#distance-computation)
8. [Index Variants](#index-variants)
9. [Performance Considerations](#performance-considerations)
10. [Usage Examples](#usage-examples)

---

## Introduction

The Flat Index (`IndexFlat`) is the simplest and most fundamental index type in FAISS. It performs **brute-force exact nearest neighbor search** by storing vectors in their original form and computing distances to all database vectors during search.

### Key Characteristics
- **Exact search**: Returns the true nearest neighbors (100% recall)
- **No training required**: Ready to use immediately after creation
- **Full precision storage**: Vectors stored as float32 without compression
- **Simple but expensive**: O(n × d) per query, where n = database size, d = dimension
- **Foundation for other indexes**: Used as storage backend for many advanced index types

### When to Use Flat Index
- Small to medium datasets (up to ~1 million vectors)
- When exact results are required
- As a baseline for evaluating approximate methods
- As storage backend for composite indexes (HNSW, IVF, etc.)

---

## Flat Index Fundamentals

### Brute-Force Search Concept

The Flat Index implements the most straightforward approach to nearest neighbor search: compare the query vector against every vector in the database.

```mermaid
graph LR
    subgraph "Query"
        Q((q))
    end
    
    subgraph "Database Vectors"
        V1((v₁))
        V2((v₂))
        V3((v₃))
        VN((vₙ))
    end
    
    subgraph "Distance Computation"
        D1["d(q,v₁)"]
        D2["d(q,v₂)"]
        D3["d(q,v₃)"]
        DN["d(q,vₙ)"]
    end
    
    subgraph "Result"
        R["Top-k smallest<br/>distances"]
    end
    
    Q --> D1
    Q --> D2
    Q --> D3
    Q --> DN
    
    V1 --> D1
    V2 --> D2
    V3 --> D3
    VN --> DN
    
    D1 --> R
    D2 --> R
    D3 --> R
    DN --> R
```

### Memory Layout

Vectors are stored contiguously in memory as a simple array:

```mermaid
graph TB
    subgraph "Memory Layout"
        direction LR
        subgraph "Vector 0"
            V0D0["v₀[0]"]
            V0D1["v₀[1]"]
            V0DD["..."]
            V0Dd["v₀[d-1]"]
        end
        
        subgraph "Vector 1"
            V1D0["v₁[0]"]
            V1D1["v₁[1]"]
            V1DD["..."]
            V1Dd["v₁[d-1]"]
        end
        
        subgraph "Vector n-1"
            VnD0["vₙ₋₁[0]"]
            VnD1["vₙ₋₁[1]"]
            VnDD["..."]
            VnDd["vₙ₋₁[d-1]"]
        end
    end
    
    subgraph "codes array (uint8_t*)"
        Addr["codes.data()"]
        Off0["offset 0"]
        Off1["offset d×sizeof(float)"]
        Offn["offset (n-1)×d×sizeof(float)"]
    end
    
    V0D0 -.->|"byte 0"| Addr
    V1D0 -.->|"byte d×4"| Off1
    VnD0 -.->|"byte (n-1)×d×4"| Offn
```

### Supported Distance Metrics

```mermaid
graph TB
    subgraph "Distance Metrics"
        L2["METRIC_L2<br/>Squared Euclidean<br/>d(x,y) = Σᵢ(xᵢ - yᵢ)²"]
        IP["METRIC_INNER_PRODUCT<br/>Inner Product (similarity)<br/>s(x,y) = Σᵢ(xᵢ × yᵢ)"]
        
        subgraph "Extra Metrics"
            L1["METRIC_L1<br/>Manhattan Distance"]
            Linf["METRIC_Linf<br/>Chebyshev Distance"]
            Canberra["METRIC_Canberra"]
            BrayCurtis["METRIC_BrayCurtis"]
            JensenShannon["METRIC_JensenShannon"]
        end
    end
    
    L2 -->|"Default"| FlatL2
    IP -->|"Cosine similarity"| FlatIP
```

---

## Class Hierarchy

### UML Class Diagram

```mermaid
classDiagram
    class Index {
        <<abstract>>
        +int d
        +idx_t ntotal
        +MetricType metric_type
        +float metric_arg
        +bool is_trained
        +bool verbose
        +add(n, x)*
        +search(n, x, k, distances, labels)*
        +train(n, x)
        +reset()*
        +reconstruct(key, recons)
        +remove_ids(sel)
        +get_distance_computer() DistanceComputer*
    }
    
    class IndexFlatCodes {
        +size_t code_size
        +MaybeOwnedVector~uint8_t~ codes
        +add(n, x)
        +reset()
        +reconstruct_n(i0, ni, recons)
        +reconstruct(key, recons)
        +sa_code_size() size_t
        +remove_ids(sel) size_t
        +get_FlatCodesDistanceComputer() FlatCodesDistanceComputer*
        +search(n, x, k, distances, labels, params)
        +range_search(n, x, radius, result, params)
        +merge_from(otherIndex, add_id)
        +permute_entries(perm)
    }
    
    class IndexFlat {
        +search(n, x, k, distances, labels, params)
        +range_search(n, x, radius, result, params)
        +reconstruct(key, recons)
        +compute_distance_subset(n, x, k, distances, labels)
        +get_xb() float*
        +get_FlatCodesDistanceComputer() FlatCodesDistanceComputer*
        +sa_encode(n, x, bytes)
        +sa_decode(n, bytes, x)
    }
    
    class IndexFlatL2 {
        +vector~float~ cached_l2norms
        +get_FlatCodesDistanceComputer() FlatCodesDistanceComputer*
        +sync_l2norms()
        +clear_l2norms()
    }
    
    class IndexFlatIP {
        <<Inner Product metric>>
    }
    
    class IndexFlat1D {
        +bool continuous_update
        +vector~idx_t~ perm
        +update_permutation()
        +add(n, x)
        +reset()
        +search(n, x, k, distances, labels, params)
    }
    
    class IndexFlatPanorama {
        +size_t batch_size
        +size_t n_levels
        +vector~float~ cum_sums
        +Panorama pano
        +add(n, x)
        +search(n, x, k, distances, labels, params)
        +range_search(n, x, radius, result, params)
        +reset()
        +reconstruct(key, recons)
        +remove_ids(sel)
    }
    
    class IndexFlatL2Panorama {
        <<L2 with Panorama>>
    }
    
    class IndexFlatIPPanorama {
        <<IP with Panorama>>
    }
    
    class IndexBinaryFlat {
        +MaybeOwnedVector~uint8_t~ xb
        +bool use_heap
        +size_t query_batch_size
        +add(n, x)
        +search(n, x, k, distances, labels, params)
        +range_search(n, x, radius, result, params)
        +reset()
        +remove_ids(sel)
    }
    
    Index <|-- IndexFlatCodes
    IndexFlatCodes <|-- IndexFlat
    IndexFlat <|-- IndexFlatL2
    IndexFlat <|-- IndexFlatIP
    IndexFlatL2 <|-- IndexFlat1D
    IndexFlat <|-- IndexFlatPanorama
    IndexFlatPanorama <|-- IndexFlatL2Panorama
    IndexFlatPanorama <|-- IndexFlatIPPanorama
    
    Index <|-- IndexBinary
    IndexBinary <|-- IndexBinaryFlat
```

### Distance Computer Hierarchy

```mermaid
classDiagram
    class DistanceComputer {
        <<abstract>>
        +set_query(x)*
        +operator()(i) float*
        +distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3)
        +symmetric_dis(i, j) float*
    }
    
    class FlatCodesDistanceComputer {
        +const uint8_t* codes
        +size_t code_size
        +const float* q
        +operator()(i) float
        +distance_to_code(code) float*
        +partial_dot_product(i, offset, num_components) float
        +partial_dot_product_batch_4(...)
    }
    
    class FlatL2Dis {
        +size_t d
        +idx_t nb
        +const float* b
        +size_t ndis
        +distance_to_code(code) float
        +symmetric_dis(i, j) float
        +distances_batch_4(...)
        +partial_dot_product(...)
    }
    
    class FlatIPDis {
        +size_t d
        +idx_t nb
        +const float* q
        +const float* b
        +size_t ndis
        +distance_to_code(code) float
        +symmetric_dis(i, j) float
        +distances_batch_4(...)
    }
    
    class FlatL2WithNormsDis {
        +const float* l2norms
        +float query_l2norm
        +operator()(i) float
        +symmetric_dis(i, j) float
        +distances_batch_4(...)
    }
    
    class NegativeDistanceComputer {
        +DistanceComputer* basedis
        +operator()(i) float
        +symmetric_dis(i, j) float
    }
    
    DistanceComputer <|-- FlatCodesDistanceComputer
    DistanceComputer <|-- NegativeDistanceComputer
    FlatCodesDistanceComputer <|-- FlatL2Dis
    FlatCodesDistanceComputer <|-- FlatIPDis
    FlatCodesDistanceComputer <|-- FlatL2WithNormsDis
```

### Component Relationships

```mermaid
graph TB
    subgraph "Index Layer"
        IF[IndexFlat]
        IFL2[IndexFlatL2]
        IFLIP[IndexFlatIP]
        IFLP[IndexFlatPanorama]
    end
    
    subgraph "Base Storage"
        IFC[IndexFlatCodes]
        CODES["codes: MaybeOwnedVector<uint8_t>"]
    end
    
    subgraph "Distance Computation"
        DC[DistanceComputer]
        FDC[FlatCodesDistanceComputer]
        FL2D[FlatL2Dis]
        FIPD[FlatIPDis]
    end
    
    subgraph "BLAS Operations"
        KNN_L2["knn_L2sqr()"]
        KNN_IP["knn_inner_product()"]
        FVEC["fvec_L2sqr()<br/>fvec_inner_product()"]
    end
    
    subgraph "Heap Management"
        HEAP["float_maxheap_array_t<br/>float_minheap_array_t"]
    end
    
    IF --> IFC
    IFL2 --> IF
    IFLIP --> IF
    IFLP --> IF
    
    IFC --> CODES
    
    IF --> DC
    DC --> FDC
    FDC --> FL2D
    FDC --> FIPD
    
    IF --> KNN_L2
    IF --> KNN_IP
    KNN_L2 --> FVEC
    KNN_IP --> FVEC
    
    KNN_L2 --> HEAP
    KNN_IP --> HEAP
```

---

## Core Data Structures

### IndexFlatCodes Storage

```mermaid
graph TB
    subgraph "IndexFlatCodes Members"
        CS["code_size: size_t<br/>= sizeof(float) × d for IndexFlat"]
        CODES["codes: MaybeOwnedVector<uint8_t><br/>Raw byte storage for all vectors"]
        
        subgraph "Inherited from Index"
            D["d: int<br/>Vector dimension"]
            NT["ntotal: idx_t<br/>Number of vectors"]
            MT["metric_type: MetricType"]
            IT["is_trained: bool<br/>Always true for Flat"]
        end
    end
    
    subgraph "Memory Calculation"
        MEM["Total Memory = ntotal × code_size bytes<br/>For IndexFlat: ntotal × d × 4 bytes"]
    end
    
    CODES --> MEM
    CS --> MEM
    NT --> MEM
```

### Heap Structures for k-NN

```mermaid
graph TB
    subgraph "float_maxheap_array_t (for L2)"
        direction TB
        NH1["nh: size_t<br/>Number of queries"]
        K1["k: size_t<br/>Neighbors per query"]
        IDS1["ids: idx_t*<br/>Labels array"]
        VAL1["val: float*<br/>Distances array"]
        
        Note1["Max-heap: largest distance at top<br/>Replace top when finding smaller distance"]
    end
    
    subgraph "float_minheap_array_t (for IP)"
        direction TB
        NH2["nh: size_t<br/>Number of queries"]
        K2["k: size_t<br/>Neighbors per query"]
        IDS2["ids: idx_t*<br/>Labels array"]
        VAL2["val: float*<br/>Distances array"]
        
        Note2["Min-heap: smallest IP at top<br/>Replace top when finding larger IP"]
    end
```

---

## Add Operation

### Add Operation Flowchart

```mermaid
flowchart TD
    START([Start: add n vectors]) --> CHECK[Check is_trained == true]
    CHECK --> ZERO{n == 0?}
    ZERO -->|Yes| END([Return])
    ZERO -->|No| RESIZE[Resize codes array:<br/>codes.resize((ntotal + n) × code_size)]
    RESIZE --> ENCODE[Encode vectors:<br/>sa_encode(n, x, codes.data() + ntotal × code_size)]
    ENCODE --> UPDATE[Update count:<br/>ntotal += n]
    UPDATE --> END
```

### Add Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant IndexFlat
    participant IndexFlatCodes
    participant MaybeOwnedVector as codes
    
    User->>IndexFlat: add(n, x)
    IndexFlat->>IndexFlatCodes: add(n, x)
    
    IndexFlatCodes->>IndexFlatCodes: Check is_trained
    
    alt n == 0
        IndexFlatCodes-->>User: Return immediately
    end
    
    IndexFlatCodes->>codes: resize((ntotal + n) × code_size)
    Note over codes: Allocate space for new vectors
    
    IndexFlatCodes->>IndexFlat: sa_encode(n, x, codes + offset)
    Note over IndexFlat: For IndexFlat: memcpy(bytes, x, sizeof(float) × d × n)
    
    IndexFlatCodes->>IndexFlatCodes: ntotal += n
    
    IndexFlatCodes-->>User: Done
```

### Encoding for IndexFlat

For `IndexFlat`, encoding is simply a memory copy:

```mermaid
flowchart LR
    subgraph "Input (float*)"
        X["x: float[n × d]"]
    end
    
    subgraph "sa_encode()"
        MEMCPY["memcpy(bytes, x, sizeof(float) × d × n)"]
    end
    
    subgraph "codes array"
        BYTES["bytes: uint8_t[n × d × 4]"]
    end
    
    X --> MEMCPY
    MEMCPY --> BYTES
```

---

## Search Operation

### Search Flowchart

```mermaid
flowchart TD
    START([Start: search n queries for k neighbors]) --> CHECK[Check k > 0]
    CHECK --> GET_SEL[Get IDSelector from params]
    
    GET_SEL --> METRIC{metric_type?}
    
    METRIC -->|METRIC_L2| L2_SEARCH[Create max-heap result structure<br/>knn_L2sqr(x, xb, d, n, ntotal, res)]
    METRIC -->|METRIC_INNER_PRODUCT| IP_SEARCH[Create min-heap result structure<br/>knn_inner_product(x, xb, d, n, ntotal, res)]
    METRIC -->|Other| EXTRA_SEARCH[knn_extra_metrics(...)]
    
    L2_SEARCH --> END([Return sorted results])
    IP_SEARCH --> END
    EXTRA_SEARCH --> END
```

### Search Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant IndexFlat
    participant knn_func as knn_L2sqr/knn_inner_product
    participant BLAS as BLAS Operations
    participant Heap
    
    User->>IndexFlat: search(n, x, k, distances, labels)
    
    IndexFlat->>IndexFlat: Get IDSelector from params
    IndexFlat->>IndexFlat: Validate k > 0
    
    alt metric_type == METRIC_L2
        IndexFlat->>IndexFlat: Create float_maxheap_array_t
        IndexFlat->>knn_func: knn_L2sqr(x, xb, d, n, ntotal, res, nullptr, sel)
    else metric_type == METRIC_INNER_PRODUCT
        IndexFlat->>IndexFlat: Create float_minheap_array_t
        IndexFlat->>knn_func: knn_inner_product(x, xb, d, n, ntotal, res, sel)
    end
    
    loop For each query batch
        knn_func->>BLAS: Compute distance matrix<br/>(uses GEMM for large batches)
        BLAS-->>knn_func: Distance values
        
        loop For each query in batch
            loop For each database vector
                knn_func->>Heap: heap_addn() or heap_replace_top()
            end
        end
    end
    
    knn_func->>Heap: heap_reorder() for each query
    knn_func-->>IndexFlat: Results in distances/labels
    
    IndexFlat-->>User: Sorted (distances, labels)
```

### Distance Computation Strategies

```mermaid
flowchart TD
    subgraph "Strategy Selection"
        SIZE{Database size?}
        
        SIZE -->|"< threshold<br/>(~20 vectors)"| DIRECT[Direct SIMD computation<br/>fvec_L2sqr / fvec_inner_product]
        
        SIZE -->|">= threshold"| BLAS[BLAS-based computation<br/>Matrix multiplication + heap]
    end
    
    subgraph "BLAS Strategy"
        direction TB
        BATCH_Q[Batch queries<br/>query_bs = 4096]
        BATCH_DB[Batch database<br/>database_bs = 1024]
        GEMM["GEMM: compute<br/>distance matrix block"]
        HEAP_ADD[Update heaps with<br/>distance values]
        
        BATCH_Q --> BATCH_DB
        BATCH_DB --> GEMM
        GEMM --> HEAP_ADD
    end
    
    subgraph "Direct Strategy"
        direction TB
        LOOP[Loop over all vectors]
        SIMD[SIMD distance computation<br/>4 vectors at a time]
        HEAP_UP[Heap update]
        
        LOOP --> SIMD
        SIMD --> HEAP_UP
    end
    
    DIRECT --> Direct Strategy
    BLAS --> BLAS Strategy
```

### L2 Distance Computation via BLAS

For L2 distance, the squared distance can be decomposed:

$$\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 \langle x, y \rangle$$

```mermaid
flowchart LR
    subgraph "Precompute"
        XN["||x||² for queries"]
        YN["||y||² for database"]
    end
    
    subgraph "GEMM"
        IP["X × Y^T<br/>Inner products matrix"]
    end
    
    subgraph "Combine"
        DIST["D[i,j] = ||x_i||² + ||y_j||² - 2×IP[i,j]"]
    end
    
    XN --> DIST
    YN --> DIST
    IP --> DIST
```

---

## Distance Computation

### FlatL2Dis Implementation

```mermaid
classDiagram
    class FlatL2Dis {
        -size_t d
        -idx_t nb
        -const float* b
        -size_t ndis
        -size_t npartial_dot_products
        
        +distance_to_code(code) float
        +partial_dot_product(i, offset, num_components) float
        +symmetric_dis(i, j) float
        +distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3)
        +partial_dot_product_batch_4(...)
    }
    
    note for FlatL2Dis "distance_to_code:\n  return fvec_L2sqr(q, code, d)\n\nsymmetric_dis:\n  return fvec_L2sqr(b+j*d, b+i*d, d)"
```

### Batched Distance Computation

```mermaid
flowchart TD
    subgraph "distances_batch_4"
        IN[Input: idx0, idx1, idx2, idx3]
        GET["Get 4 vector pointers:<br/>y0 = codes + idx0 × code_size<br/>y1 = codes + idx1 × code_size<br/>y2 = codes + idx2 × code_size<br/>y3 = codes + idx3 × code_size"]
        BATCH["fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3)"]
        OUT[Output: dis0, dis1, dis2, dis3]
    end
    
    IN --> GET
    GET --> BATCH
    BATCH --> OUT
    
    subgraph "SIMD Benefits"
        NOTE["Processing 4 vectors together allows:<br/>- Better cache utilization<br/>- More efficient SIMD operations<br/>- Reduced loop overhead"]
    end
```

### L2 With Cached Norms

`IndexFlatL2` can cache L2 norms for faster distance computation:

```mermaid
flowchart TD
    subgraph "Without Cached Norms"
        FULL["L2sqr(q, y) = Σᵢ(qᵢ - yᵢ)²<br/>Requires full vector traversal"]
    end
    
    subgraph "With Cached Norms"
        CACHED["L2sqr(q, y) = ||q||² + ||y||² - 2⟨q,y⟩<br/>||y||² is precomputed and cached"]
        
        STEPS["1. Precompute ||q||² once per query<br/>2. Look up ||y||² from cache<br/>3. Compute ⟨q,y⟩ via inner product"]
        
        BENEFIT["Inner product is faster than<br/>full L2sqr due to BLAS optimization"]
    end
    
    FULL --> CACHED
    CACHED --> STEPS
    STEPS --> BENEFIT
```

---

## Index Variants

### IndexFlatL2 and IndexFlatIP

```mermaid
classDiagram
    class IndexFlatL2 {
        <<L2 Squared Distance>>
        +vector~float~ cached_l2norms
        +sync_l2norms()
        +clear_l2norms()
        Note: Use max-heap during search
        Note: Smaller distance = more similar
    }
    
    class IndexFlatIP {
        <<Inner Product>>
        Note: Use min-heap during search
        Note: Larger IP = more similar
        Note: For cosine similarity
        Note: normalize vectors first
    }
    
    IndexFlat <|-- IndexFlatL2
    IndexFlat <|-- IndexFlatIP
```

### IndexFlat1D - Optimized for 1D Vectors

```mermaid
flowchart TD
    subgraph "IndexFlat1D Structure"
        PERM["perm: sorted indices<br/>perm[i] = index of i-th smallest value"]
        CU["continuous_update: bool<br/>Auto-update permutation on add"]
    end
    
    subgraph "Binary Search"
        QUERY["Query value q"]
        FIND["Binary search to find<br/>position where q would be inserted"]
        EXPAND["Expand left and right<br/>to find k nearest"]
    end
    
    subgraph "Complexity"
        ADD["Add: O(n log n) if continuous_update"]
        SEARCH["Search: O(log n + k)<br/>vs O(n) for regular flat"]
    end
    
    PERM --> FIND
    QUERY --> FIND
    FIND --> EXPAND
```

### IndexFlatPanorama - Progressive Refinement

```mermaid
flowchart TD
    subgraph "Panorama Concept"
        LEVELS["Split dimensions into levels<br/>Level 0: dims 0 to w-1<br/>Level 1: dims w to 2w-1<br/>..."]
        CUMSUMS["Precompute cumulative sums<br/>for Cauchy-Schwarz bounds"]
    end
    
    subgraph "Progressive Search"
        INIT["Initialize with full norm bounds"]
        LEVEL_LOOP{More levels?}
        COMPUTE["Compute partial distance<br/>for current level"]
        BOUND["Compute lower bound using<br/>Cauchy-Schwarz inequality"]
        PRUNE{bound > threshold?}
        SKIP["Skip this vector"]
        CONTINUE["Continue to next level"]
        RESULT["Add to results if survived"]
    end
    
    LEVELS --> INIT
    CUMSUMS --> INIT
    INIT --> LEVEL_LOOP
    LEVEL_LOOP -->|Yes| COMPUTE
    COMPUTE --> BOUND
    BOUND --> PRUNE
    PRUNE -->|Yes| SKIP
    SKIP --> LEVEL_LOOP
    PRUNE -->|No| CONTINUE
    CONTINUE --> LEVEL_LOOP
    LEVEL_LOOP -->|No| RESULT
```

### IndexBinaryFlat - Binary Vector Search

```mermaid
classDiagram
    class IndexBinaryFlat {
        +MaybeOwnedVector~uint8_t~ xb
        +bool use_heap
        +size_t query_batch_size
        +ApproxTopK_mode_t approx_topk_mode
        
        +add(n, x)
        +search(n, x, k, distances, labels)
        +range_search(n, x, radius, result)
    }
    
    note for IndexBinaryFlat "Stores binary vectors (bits packed into bytes)\nUses Hamming distance\nVector size: d/8 bytes per vector"
```

---

## Performance Considerations

### Memory Requirements

| Index Type | Memory per Vector | Formula |
|------------|------------------|---------|
| IndexFlat | 4d bytes | `sizeof(float) × d` |
| IndexFlatL2 (with cache) | 4d + 4 bytes | `sizeof(float) × (d + 1)` |
| IndexBinaryFlat | d/8 bytes | `d / 8` |

### Complexity Analysis

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Add n vectors | O(n × d) | Memory copy |
| Search (single query) | O(ntotal × d) | Brute force |
| Search (batch of nq queries) | O(nq × ntotal × d) | Can use BLAS |
| Reconstruct | O(d) | Memory copy |
| Remove IDs | O(ntotal × code_size) | Compaction |

### BLAS Optimization

```mermaid
graph TB
    subgraph "BLAS Configuration"
        THR["distance_compute_blas_threshold = 20<br/>Use BLAS if ntotal >= 20"]
        QBS["distance_compute_blas_query_bs = 4096<br/>Query batch size"]
        DBS["distance_compute_blas_database_bs = 1024<br/>Database batch size"]
    end
    
    subgraph "Benefits"
        B1["Leverages optimized GEMM routines"]
        B2["Better cache utilization"]
        B3["Automatic multi-threading in MKL/OpenBLAS"]
    end
    
    THR --> B1
    QBS --> B2
    DBS --> B3
```

### Parallelization

```mermaid
flowchart LR
    subgraph "Query-Level Parallelism"
        Q1["Query 1"]
        Q2["Query 2"]
        Q3["Query 3"]
        QN["Query n"]
    end
    
    subgraph "OpenMP"
        OMP["#pragma omp parallel for"]
    end
    
    subgraph "Per-Query Processing"
        P1["Thread 1"]
        P2["Thread 2"]
        P3["Thread 3"]
        PN["Thread n"]
    end
    
    Q1 --> OMP
    Q2 --> OMP
    Q3 --> OMP
    QN --> OMP
    
    OMP --> P1
    OMP --> P2
    OMP --> P3
    OMP --> PN
```

### When to Use Each Variant

```mermaid
graph TD
    START{What's your use case?}
    
    START -->|Euclidean distance| L2{Dataset size?}
    START -->|Cosine similarity| IP[IndexFlatIP<br/>Normalize vectors first]
    START -->|Binary vectors| BIN[IndexBinaryFlat]
    START -->|1D data| D1[IndexFlat1D<br/>O(log n) search]
    START -->|High-dim, want pruning| PANO[IndexFlatPanorama]
    
    L2 -->|Small, need caching| L2CACHE[IndexFlatL2<br/>with sync_l2norms()]
    L2 -->|General use| L2REG[IndexFlatL2]
```

---

## Usage Examples

### Basic Usage (C++)

```cpp
#include <faiss/IndexFlat.h>

int main() {
    int d = 64;       // Dimension
    int nb = 100000;  // Database size
    int nq = 10;      // Number of queries
    int k = 4;        // Number of results
    
    // Create index (L2 distance)
    faiss::IndexFlatL2 index(d);
    
    // Check that no training is needed
    assert(index.is_trained);
    
    // Create and add vectors
    std::vector<float> xb(d * nb);
    // ... fill xb with database vectors ...
    index.add(nb, xb.data());
    
    printf("ntotal = %zd\n", index.ntotal);
    
    // Search
    std::vector<float> xq(d * nq);
    // ... fill xq with query vectors ...
    
    std::vector<float> distances(k * nq);
    std::vector<faiss::idx_t> labels(k * nq);
    
    index.search(nq, xq.data(), k, distances.data(), labels.data());
    
    // Results are sorted by distance (ascending for L2)
    return 0;
}
```

### Using Inner Product (Cosine Similarity)

```cpp
#include <faiss/IndexFlat.h>
#include <cmath>

// Normalize vectors for cosine similarity
void normalize(float* x, int d) {
    float norm = 0;
    for (int i = 0; i < d; i++) {
        norm += x[i] * x[i];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < d; i++) {
        x[i] /= norm;
    }
}

int main() {
    int d = 64;
    
    // Use inner product index
    faiss::IndexFlatIP index(d);
    
    // Normalize vectors before adding
    std::vector<float> xb(d * nb);
    for (int i = 0; i < nb; i++) {
        normalize(xb.data() + i * d, d);
    }
    index.add(nb, xb.data());
    
    // Normalize queries before searching
    std::vector<float> xq(d);
    normalize(xq.data(), d);
    
    // Search - larger IP means more similar
    index.search(1, xq.data(), k, distances.data(), labels.data());
    
    return 0;
}
```

### Python Usage

```python
import faiss
import numpy as np

d = 64        # Dimension
nb = 100000   # Database size
nq = 1000     # Number of queries
k = 4         # Number of results

# Generate random vectors
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Create L2 index
index = faiss.IndexFlatL2(d)
print(f"is_trained: {index.is_trained}")

# Add vectors
index.add(xb)
print(f"ntotal: {index.ntotal}")

# Search
distances, labels = index.search(xq, k)

# Sanity check: search first 5 database vectors
D, I = index.search(xb[:5], k)
print("First result for each query should be itself:")
print(I)
print(D)  # First distance should be 0
```

### Range Search

```python
import faiss
import numpy as np

d = 64
nb = 10000
nq = 10

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

index = faiss.IndexFlatL2(d)
index.add(xb)

# Range search: find all vectors within radius
radius = 5.0
lims, distances, labels = index.range_search(xq, radius)

# Results are grouped by query
for i in range(nq):
    start, end = lims[i], lims[i + 1]
    print(f"Query {i}: {end - start} results within radius {radius}")
```

### Using with IDSelector

```python
import faiss
import numpy as np

d = 64
nb = 10000
nq = 10
k = 4

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

index = faiss.IndexFlatL2(d)
index.add(xb)

# Search only in a subset of IDs
subset_ids = np.array([100, 200, 300, 400, 500], dtype=np.int64)
selector = faiss.IDSelectorArray(subset_ids)

params = faiss.SearchParameters()
params.sel = selector

# Search only considers vectors in subset_ids
distances, labels = index.search(xq, k, params=params)
```

---

## Summary

The Flat Index in FAISS provides:

1. **Exact nearest neighbor search** - 100% recall guaranteed
2. **No training overhead** - Ready to use immediately
3. **Simple implementation** - Easy to understand and debug
4. **Foundation for other indexes** - Used as storage backend for HNSW, IVF, etc.
5. **Multiple variants** - L2, Inner Product, Binary, 1D, Panorama

### Trade-offs

| Advantage | Disadvantage |
|-----------|--------------|
| Exact results | O(n) search time |
| No training | High memory (full vectors) |
| Simple API | Not scalable to billions of vectors |
| Supports all metrics | CPU intensive for large datasets |

### Recommended Usage

- **Dataset < 100K vectors**: Use directly
- **Dataset 100K - 1M vectors**: Consider with BLAS optimization
- **Dataset > 1M vectors**: Use as storage for approximate indexes (HNSW, IVF)
