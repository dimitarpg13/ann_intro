# FAISS IVFPQ Implementation Details

## Table of Contents
1. [Introduction to Inverted File with Product Quantization](#1-introduction-to-inverted-file-with-product-quantization)
2. [Product Quantization Fundamentals](#2-product-quantization-fundamentals)
3. [IVFPQ Conceptual Overview](#3-ivfpq-conceptual-overview)
4. [Class Hierarchy and UML Diagrams](#4-class-hierarchy-and-uml-diagrams)
5. [Component Details](#5-component-details)
6. [Operation Flowcharts](#6-operation-flowcharts)
7. [Sequence Diagrams](#7-sequence-diagrams)
8. [Distance Computation](#8-distance-computation)
9. [Data Structures](#9-data-structures)
10. [Performance Considerations](#10-performance-considerations)

---

## 1. Introduction to Inverted File with Product Quantization

### What is IVFPQ?

**IVFPQ** (Inverted File with Product Quantization) is one of the most important and widely used approximate nearest neighbor (ANN) search algorithms in FAISS. It combines two powerful techniques:

1. **Inverted File (IVF)**: Partitions the vector space into clusters for non-exhaustive search
2. **Product Quantization (PQ)**: Compresses vectors into compact codes for memory efficiency

### Why Use IVFPQ?

IVFPQ is the workhorse of billion-scale similarity search because it provides:

- **Memory Efficiency**: Compresses d-dimensional vectors to just M bytes (typically 8-64 bytes)
- **Speed**: Combines IVF's cluster-based search with efficient table-based distance computation
- **Scalability**: Can handle billions of vectors on a single machine
- **Tunable Trade-offs**: Parameters allow fine control over memory/speed/accuracy

### Compression Example

```
Original vector:    128 floats × 4 bytes = 512 bytes
With PQ (M=8, 8bit): 8 bytes
Compression ratio:  64× reduction!
```

---

## 2. Product Quantization Fundamentals

### The Core Idea

Product Quantization divides a d-dimensional vector into M equal sub-vectors, then quantizes each sub-vector independently using its own codebook of k=2^nbits centroids.

```mermaid
flowchart LR
    subgraph OriginalVector["Original Vector (d dimensions)"]
        V["x = [x₁, x₂, ..., xd]"]
    end
    
    subgraph SubVectors["Split into M Sub-vectors"]
        S1["x¹ (d/M dims)"]
        S2["x² (d/M dims)"]
        S3["..."]
        SM["xᴹ (d/M dims)"]
    end
    
    subgraph Quantized["Quantize Each Sub-vector"]
        Q1["q₁ ∈ {0..255}"]
        Q2["q₂ ∈ {0..255}"]
        Q3["..."]
        QM["qₘ ∈ {0..255}"]
    end
    
    subgraph Code["PQ Code (M bytes)"]
        C["[q₁, q₂, ..., qₘ]"]
    end
    
    V --> S1 & S2 & S3 & SM
    S1 --> Q1
    S2 --> Q2
    S3 --> Q3
    SM --> QM
    Q1 & Q2 & Q3 & QM --> C
```

### PQ Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `d` | Vector dimension | 64, 128, 256, etc. |
| `M` | Number of sub-quantizers | 8, 16, 32, 64 |
| `nbits` | Bits per sub-quantizer index | 8 (256 centroids), rarely 4 or 16 |
| `dsub` | Sub-vector dimension (d/M) | 8, 16, etc. |
| `ksub` | Centroids per sub-quantizer (2^nbits) | 256 (for nbits=8) |

### Codebook Structure

```mermaid
graph TB
    subgraph Codebook["PQ Codebook: M × ksub × dsub floats"]
        subgraph CB1["Sub-quantizer 1"]
            C1_0["Centroid 0"]
            C1_1["Centroid 1"]
            C1_255["Centroid 255"]
        end
        
        subgraph CB2["Sub-quantizer 2"]
            C2_0["Centroid 0"]
            C2_1["Centroid 1"]
            C2_255["Centroid 255"]
        end
        
        subgraph CBM["Sub-quantizer M"]
            CM_0["Centroid 0"]
            CM_1["Centroid 1"]
            CM_255["Centroid 255"]
        end
    end
    
    Note1["Each centroid is dsub = d/M floats"]
```

---

## 3. IVFPQ Conceptual Overview

### How IVFPQ Works

IVFPQ uses a two-level quantization approach:

1. **Coarse Quantization (IVF)**: Assigns vectors to one of nlist clusters
2. **Fine Quantization (PQ)**: Encodes the residual (vector - centroid) using PQ

```mermaid
flowchart TB
    subgraph Training["Training Phase"]
        T1[Training Vectors] --> T2[K-Means: Learn nlist Centroids]
        T2 --> T3[Compute Residuals]
        T3 --> T4[Train PQ on Residuals]
        T4 --> T5[Learn M × ksub Sub-centroids]
    end
    
    subgraph Adding["Adding Vectors Phase"]
        A1[Input Vector x] --> A2[Find Nearest Centroid c]
        A2 --> A3[Compute Residual: r = x - c]
        A3 --> A4[Encode r with PQ → M-byte code]
        A4 --> A5[Store code in Inverted List]
    end
    
    subgraph Searching["Search Phase"]
        S1[Query Vector q] --> S2[Find nprobe Nearest Centroids]
        S2 --> S3[Compute Distance Tables]
        S3 --> S4[Scan Codes in Selected Lists]
        S4 --> S5[Accumulate Distances from Tables]
        S5 --> S6[Maintain Top-K Heap]
        S6 --> S7[Return K Nearest Neighbors]
    end
    
    Training --> Adding
    Training --> Searching
```

### The Residual Encoding (by_residual = true)

```mermaid
flowchart LR
    subgraph Original["Original Vector"]
        X["x"]
    end
    
    subgraph CoarseQuant["Coarse Quantization"]
        C["centroid c = quantizer(x)"]
    end
    
    subgraph Residual["Residual Computation"]
        R["r = x - c"]
    end
    
    subgraph PQEncode["PQ Encoding"]
        Code["PQ code of r"]
    end
    
    subgraph Storage["Stored Data"]
        Store["list_no + code"]
    end
    
    X --> C
    X --> R
    C --> R
    R --> Code
    Code --> Store
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `nlist` | Number of IVF clusters | 256 to 65536 |
| `nprobe` | Clusters to search | 1 to 256 |
| `M` | PQ sub-quantizers | 8, 16, 32 |
| `nbits` | Bits per code | 8 (standard) |
| `by_residual` | Encode residuals vs raw vectors | true (recommended) |
| `use_precomputed_table` | Precompute distance terms | 0, 1, or 2 |

---

## 4. Class Hierarchy and UML Diagrams

### Main Class Hierarchy

```mermaid
classDiagram
    class Index {
        <<abstract>>
        +int d
        +idx_t ntotal
        +bool is_trained
        +MetricType metric_type
        +train(n, x)*
        +add(n, x)*
        +search(n, x, k, distances, labels)*
    }
    
    class IndexIVF {
        +InvertedLists* invlists
        +size_t code_size
        +DirectMap direct_map
        +bool by_residual
        +train(n, x)
        +add_core(n, x, xids, precomputed_idx)
        +encode_vectors()*
        +search_preassigned()
    }
    
    class IndexIVFPQ {
        +ProductQuantizer pq
        +bool do_polysemous_training
        +PolysemousTraining* polysemous_training
        +size_t scan_table_threshold
        +int polysemous_ht
        +int use_precomputed_table
        +AlignedTable~float~ precomputed_table
        +train_encoder(n, x, assign)
        +encode(key, x, code)
        +encode_multiple(n, keys, x, codes)
        +decode_multiple(n, keys, codes, x)
        +precompute_table()
        +get_InvertedListScanner()
        +find_duplicates(ids, lims)
    }
    
    class IndexIVFPQR {
        +ProductQuantizer refine_pq
        +vector~uint8_t~ refine_codes
        +float k_factor
        +train_encoder(n, x, assign)
        +search_preassigned()
    }
    
    class IndexIVFPQFastScan {
        +ProductQuantizer pq
        +int use_precomputed_table
        +train_encoder(n, x, assign)
        +compute_LUT()
    }
    
    Index <|-- IndexIVF
    IndexIVF <|-- IndexIVFPQ
    IndexIVFPQ <|-- IndexIVFPQR
    IndexIVF <|-- IndexIVFFastScan
    IndexIVFFastScan <|-- IndexIVFPQFastScan
```

### ProductQuantizer Class

```mermaid
classDiagram
    class Quantizer {
        <<abstract>>
        +size_t d
        +size_t code_size
        +compute_codes(x, codes, n)*
        +decode(codes, x, n)*
    }
    
    class ProductQuantizer {
        +size_t M
        +size_t nbits
        +size_t dsub
        +size_t ksub
        +bool verbose
        +train_type_t train_type
        +ClusteringParameters cp
        +Index* assign_index
        +vector~float~ centroids
        +vector~float~ transposed_centroids
        +vector~float~ centroids_sq_lengths
        +vector~float~ sdc_table
        +get_centroids(m, i) float*
        +train(n, x)
        +set_derived_values()
        +compute_code(x, code)
        +compute_codes(x, codes, n)
        +decode(code, x)
        +compute_distance_table(x, dis_table)
        +compute_inner_prod_table(x, dis_table)
        +search(x, nx, codes, ncodes, res)
        +compute_sdc_table()
    }
    
    class PQEncoder8 {
        +uint8_t* code
        +encode(x)
    }
    
    class PQDecoder8 {
        +uint8_t* code
        +decode() uint64
    }
    
    class PQEncoderGeneric {
        +uint8_t* code
        +int nbits
        +encode(x)
    }
    
    class PQDecoderGeneric {
        +uint8_t* code
        +int nbits
        +decode() uint64
    }
    
    Quantizer <|-- ProductQuantizer
    ProductQuantizer ..> PQEncoder8 : uses
    ProductQuantizer ..> PQDecoder8 : uses
    ProductQuantizer ..> PQEncoderGeneric : uses
    ProductQuantizer ..> PQDecoderGeneric : uses
```

### Scanner and Query Tables

```mermaid
classDiagram
    class InvertedListScanner {
        <<abstract>>
        +idx_t list_no
        +bool keep_max
        +bool store_pairs
        +IDSelector* sel
        +size_t code_size
        +set_query(query_vector)*
        +set_list(list_no, coarse_dis)
        +distance_to_code(code)* float
        +scan_codes(n, codes, ids, handler)
    }
    
    class QueryTables {
        +IndexIVFPQ& ivfpq
        +int d
        +ProductQuantizer& pq
        +MetricType metric_type
        +bool by_residual
        +int use_precomputed_table
        +int polysemous_ht
        +float* sim_table
        +float* sim_table_2
        +float* residual_vec
        +float* decoded_vec
        +vector~float*~ sim_table_ptrs
        +init_query(qi)
        +precompute_list_tables() float
        +precompute_list_table_pointers() float
    }
    
    class IVFPQScannerT~IDType, MetricType, PQDecoder~ {
        +uint8_t* list_codes
        +IDType* list_ids
        +size_t list_size
        +float dis0
        +init_list(list_no, coarse_dis, mode)
        +scan_list_with_table(ncode, codes, res)
        +scan_list_with_pointer(ncode, codes, res)
        +scan_on_the_fly_dist(ncode, codes, res)
        +scan_list_polysemous(ncode, codes, res)
    }
    
    class IVFPQScanner~MetricType, C, PQDecoder, use_sel~ {
        +int precompute_mode
        +IDSelector* sel
        +set_query(query)
        +set_list(list_no, coarse_dis)
        +distance_to_code(code) float
        +scan_codes(ncode, codes, ids, handler)
    }
    
    QueryTables <|-- IVFPQScannerT
    IVFPQScannerT <|-- IVFPQScanner
    InvertedListScanner <|-- IVFPQScanner
```

### Component Relationships

```mermaid
classDiagram
    class IndexIVFPQ {
        Core IVFPQ Index
    }
    
    class Quantizer_Coarse {
        Coarse Quantizer
        Maps vectors to IVF cells
    }
    
    class ProductQuantizer {
        Fine Quantizer
        Compresses residuals
    }
    
    class InvertedLists {
        Stores PQ codes per cluster
    }
    
    class PrecomputedTable {
        Distance table cache
        nlist × M × ksub
    }
    
    class IVFPQScanner {
        Computes distances
        using lookup tables
    }
    
    IndexIVFPQ "1" --> "1" Quantizer_Coarse : quantizer
    IndexIVFPQ "1" --> "1" ProductQuantizer : pq
    IndexIVFPQ "1" --> "1" InvertedLists : invlists
    IndexIVFPQ "1" --> "0..1" PrecomputedTable : precomputed_table
    IndexIVFPQ "1" ..> "*" IVFPQScanner : creates
```

---

## 5. Component Details

### 5.1 ProductQuantizer

The `ProductQuantizer` is the heart of IVFPQ. It:
- Divides vectors into M sub-vectors
- Trains M independent codebooks using k-means
- Encodes/decodes vectors using the codebooks

**Training Modes:**
```cpp
enum train_type_t {
    Train_default,       // Standard k-means per subspace
    Train_hot_start,     // Use provided centroids as initialization
    Train_shared,        // Share one codebook across all subspaces
    Train_hypercube,     // Initialize with hypercube corners
    Train_hypercube_pca, // Initialize with PCA-rotated hypercube
};
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `train(n, x)` | Train M codebooks using k-means |
| `compute_code(x, code)` | Encode one vector |
| `decode(code, x)` | Decode one code to vector |
| `compute_distance_table(x, table)` | Precompute distances from x to all centroids |

### 5.2 IndexIVFPQ

The main index class that combines IVF and PQ:

**Key Characteristics:**
- `by_residual = true`: Encodes residuals (x - centroid) not raw vectors
- `code_size = M * nbits / 8`: Typically M bytes for nbits=8
- Supports polysemous filtering for faster search
- Can use precomputed tables for L2 search

**Precomputed Tables:**

For L2 search with residuals, the distance is:
```
d = ||x - y_C - y_R||²
  = ||x - y_C||² + ||y_R||² + 2*(y_C|y_R) - 2*(x|y_R)
    -----------   ---------------------------   -------
       term 1            term 2                 term 3
```

- Term 1: Coarse distance (from IVF search)
- Term 2: Can be precomputed (nlist × M × ksub table)
- Term 3: Computed per query (M × ksub table)

### 5.3 Distance Computation

```mermaid
flowchart TB
    subgraph ComputeDistance["Distance Computation for PQ Code"]
        Query["Query vector q"] --> DisTable["Compute distance table<br/>M × ksub floats"]
        DisTable --> Sum["For each code byte qᵢ:<br/>dis += table[m][qᵢ]"]
        Sum --> Total["Total distance"]
    end
    
    subgraph Table["Distance Table (M × ksub)"]
        T1["dis_table[0][0..255]<br/>distances to subquantizer 0 centroids"]
        T2["dis_table[1][0..255]<br/>distances to subquantizer 1 centroids"]
        TM["dis_table[M-1][0..255]<br/>distances to subquantizer M-1 centroids"]
    end
    
    DisTable --> T1 & T2 & TM
```

---

## 6. Operation Flowcharts

### 6.1 Training Flowchart

```mermaid
flowchart TD
    Start([Start Training]) --> TrainQ1[Train Level-1 Quantizer<br/>K-means with nlist clusters]
    TrainQ1 --> GetCentroids[Store nlist centroids in quantizer]
    
    GetCentroids --> CheckByRes{by_residual?}
    CheckByRes -->|Yes| ComputeAssign[Assign training vectors to clusters]
    ComputeAssign --> ComputeResiduals[Compute residuals:<br/>r = x - centroid]
    ComputeResiduals --> TrainPQ[Train ProductQuantizer on residuals]
    
    CheckByRes -->|No| TrainPQDirect[Train ProductQuantizer on raw vectors]
    
    TrainPQ --> PQKMeans
    TrainPQDirect --> PQKMeans
    
    subgraph PQKMeans["PQ Training (for each subspace m)"]
        PQ1[Extract sub-vectors of dimension dsub]
        PQ1 --> PQ2[Run k-means with ksub clusters]
        PQ2 --> PQ3[Store ksub centroids for subspace m]
    end
    
    PQKMeans --> CheckPoly{do_polysemous_training?}
    CheckPoly -->|Yes| PolyTrain[Reorder centroids for<br/>Hamming distance correlation]
    CheckPoly -->|No| PrecompTable
    PolyTrain --> PrecompTable
    
    PrecompTable{use_precomputed_table?}
    PrecompTable -->|Yes| BuildTable[Build precomputed distance table<br/>nlist × M × ksub]
    PrecompTable -->|No| Done
    BuildTable --> Done
    
    Done --> SetTrained[is_trained = true]
    SetTrained --> End([End Training])
```

### 6.2 Add Vectors Flowchart

```mermaid
flowchart TD
    Start([Start add_core_o]) --> CheckTrained{is_trained?}
    CheckTrained -->|No| Error([Throw Error])
    CheckTrained -->|Yes| NeedAssign{precomputed_idx<br/>provided?}
    
    NeedAssign -->|No| Quantize[quantizer.assign → get cluster IDs]
    NeedAssign -->|Yes| UseIdx[Use provided cluster IDs]
    
    Quantize --> PrepareResiduals
    UseIdx --> PrepareResiduals
    
    PrepareResiduals{by_residual?}
    PrepareResiduals -->|Yes| ComputeRes[Compute residuals:<br/>r[i] = x[i] - centroid[idx[i]]]
    PrepareResiduals -->|No| UseRaw[to_encode = raw vectors]
    ComputeRes --> EncodePQ
    UseRaw --> EncodePQ
    
    EncodePQ[pq.compute_codes → M-byte codes]
    
    EncodePQ --> LoopVectors[For each vector i]
    
    subgraph AddLoop["Add to Inverted Lists"]
        LoopVectors --> GetKey[key = cluster ID]
        GetKey --> CheckKey{key >= 0?}
        CheckKey -->|No| SkipVec[Skip vector, update direct_map]
        CheckKey -->|Yes| AddEntry[invlists.add_entry<br/>key, id, PQ_code]
        AddEntry --> UpdateDM[Update DirectMap]
        UpdateDM --> ComputeRes2{Need 2nd residual?}
        ComputeRes2 -->|Yes| Decode[Decode PQ code<br/>residual_2 = original - decoded]
        ComputeRes2 -->|No| NextVec
        Decode --> NextVec[Next vector]
        SkipVec --> NextVec
    end
    
    NextVec --> |More vectors| LoopVectors
    NextVec --> |Done| UpdateTotal[ntotal += n]
    UpdateTotal --> End([End add])
```

### 6.3 Search Flowchart

```mermaid
flowchart TD
    Start([Start Search]) --> CoarseSearch[Quantizer search:<br/>Find nprobe nearest centroids]
    CoarseSearch --> InitTables[Initialize QueryTables]
    
    InitTables --> InitQuery[init_query: Compute query-specific tables]
    
    subgraph QueryInit["Query Initialization"]
        InitQuery --> MetricCheck{metric_type?}
        MetricCheck -->|IP| ComputeIP[compute_inner_prod_table<br/>→ sim_table]
        MetricCheck -->|L2| L2Check{by_residual AND<br/>use_precomputed?}
        L2Check -->|Yes| ComputeIP2[compute_inner_prod_table<br/>→ sim_table_2]
        L2Check -->|No| ComputeL2[compute_distance_table<br/>→ sim_table]
    end
    
    ComputeIP --> ProbeLoop
    ComputeIP2 --> ProbeLoop
    ComputeL2 --> ProbeLoop
    
    ProbeLoop[For each probe in nprobe]
    
    subgraph ListScan["Scan Inverted List"]
        ProbeLoop --> GetList[Get inverted list for centroid]
        GetList --> PrecompList[precompute_list_tables:<br/>Combine query + list tables]
        PrecompList --> GetCodes[Get codes and IDs from list]
        GetCodes --> ScanMethod{Scan method?}
        
        ScanMethod -->|Table| ScanTable[scan_list_with_table]
        ScanMethod -->|Pointer| ScanPointer[scan_list_with_pointer]
        ScanMethod -->|OnTheFly| ScanOTF[scan_on_the_fly_dist]
        ScanMethod -->|Polysemous| ScanPoly[scan_list_polysemous]
        
        subgraph TableScan["Table-based Scan (optimized)"]
            ScanTable --> BatchCodes[Process 4 codes at a time]
            BatchCodes --> LookupSum[For each code byte:<br/>dis += sim_table[m][code[m]]]
            LookupSum --> AddResult[Add to result heap if good]
        end
    end
    
    ScanTable --> NextProbe
    ScanPointer --> NextProbe
    ScanOTF --> NextProbe
    ScanPoly --> NextProbe
    
    NextProbe[Next probe] --> |More probes| ProbeLoop
    NextProbe --> |Done| FinalizeHeap[Finalize heap → sorted results]
    FinalizeHeap --> End([Return distances, labels])
```

### 6.4 PQ Encoding Flowchart

```mermaid
flowchart LR
    subgraph Input["Input Vector (d dims)"]
        X["x = [x₀, x₁, ..., x_{d-1}]"]
    end
    
    subgraph Split["Split into M Sub-vectors"]
        X --> S0["x⁰ = x[0:dsub]"]
        X --> S1["x¹ = x[dsub:2*dsub]"]
        X --> SM["xᴹ⁻¹ = x[(M-1)*dsub:d]"]
    end
    
    subgraph Quantize["Quantize Each Sub-vector"]
        S0 --> Q0["Find nearest centroid in codebook 0<br/>q₀ = argmin ||x⁰ - c⁰ᵢ||"]
        S1 --> Q1["Find nearest centroid in codebook 1<br/>q₁ = argmin ||x¹ - c¹ᵢ||"]
        SM --> QM["Find nearest centroid in codebook M-1<br/>qₘ₋₁ = argmin ||xᴹ⁻¹ - cᴹ⁻¹ᵢ||"]
    end
    
    subgraph Output["PQ Code (M bytes)"]
        Q0 --> Code["[q₀, q₁, ..., qₘ₋₁]"]
        Q1 --> Code
        QM --> Code
    end
```

---

## 7. Sequence Diagrams

### 7.1 Training Sequence

```mermaid
sequenceDiagram
    participant User
    participant IndexIVFPQ
    participant Level1Quantizer as Coarse Quantizer
    participant ProductQuantizer as PQ
    participant Clustering
    
    User->>IndexIVFPQ: train(n, x)
    
    IndexIVFPQ->>Level1Quantizer: train_q1(n, x)
    Level1Quantizer->>Clustering: train k-means (nlist clusters)
    Clustering-->>Level1Quantizer: nlist centroids
    Level1Quantizer->>Level1Quantizer: quantizer.add(centroids)
    Level1Quantizer-->>IndexIVFPQ: Coarse quantizer ready
    
    IndexIVFPQ->>IndexIVFPQ: train_encoder(n, x, assign)
    
    alt by_residual = true
        IndexIVFPQ->>Level1Quantizer: assign(n, x)
        Level1Quantizer-->>IndexIVFPQ: assignments
        IndexIVFPQ->>IndexIVFPQ: compute_residuals()
    end
    
    IndexIVFPQ->>ProductQuantizer: train(n, residuals/vectors)
    
    loop For each subspace m = 0 to M-1
        ProductQuantizer->>ProductQuantizer: Extract sub-vectors
        ProductQuantizer->>Clustering: train k-means (ksub clusters)
        Clustering-->>ProductQuantizer: ksub centroids for subspace m
        ProductQuantizer->>ProductQuantizer: Store in centroids table
    end
    
    ProductQuantizer-->>IndexIVFPQ: PQ trained
    
    opt do_polysemous_training
        IndexIVFPQ->>IndexIVFPQ: Reorder centroids for Hamming
    end
    
    opt use_precomputed_table > 0
        IndexIVFPQ->>IndexIVFPQ: precompute_table()
    end
    
    IndexIVFPQ->>IndexIVFPQ: is_trained = true
    IndexIVFPQ-->>User: Training complete
```

### 7.2 Add Vectors Sequence

```mermaid
sequenceDiagram
    participant User
    participant IndexIVFPQ
    participant Quantizer
    participant ProductQuantizer as PQ
    participant InvertedLists
    
    User->>IndexIVFPQ: add(n, x)
    IndexIVFPQ->>IndexIVFPQ: add_with_ids(n, x, nullptr)
    IndexIVFPQ->>Quantizer: assign(n, x)
    Quantizer-->>IndexIVFPQ: coarse_idx[]
    
    IndexIVFPQ->>IndexIVFPQ: add_core_o(n, x, ids, nullptr, coarse_idx)
    
    alt by_residual = true
        IndexIVFPQ->>IndexIVFPQ: compute_residuals()
        Note over IndexIVFPQ: residuals[i] = x[i] - centroid[coarse_idx[i]]
    end
    
    IndexIVFPQ->>ProductQuantizer: compute_codes(residuals, codes, n)
    
    loop For each vector i
        ProductQuantizer->>ProductQuantizer: Split into M sub-vectors
        ProductQuantizer->>ProductQuantizer: Find nearest centroid in each subspace
        ProductQuantizer-->>IndexIVFPQ: M-byte code
    end
    
    loop For each vector i
        IndexIVFPQ->>InvertedLists: add_entry(list_no, id, code)
        InvertedLists-->>IndexIVFPQ: offset
        IndexIVFPQ->>IndexIVFPQ: direct_map.add(id, list_no, offset)
    end
    
    IndexIVFPQ->>IndexIVFPQ: ntotal += n
    IndexIVFPQ-->>User: Vectors added
```

### 7.3 Search Sequence

```mermaid
sequenceDiagram
    participant User
    participant IndexIVFPQ
    participant Quantizer
    participant Scanner as IVFPQScanner
    participant InvertedLists
    participant Heap as Result Heap
    
    User->>IndexIVFPQ: search(n, x, k, distances, labels)
    IndexIVFPQ->>Quantizer: search(n, x, nprobe)
    Quantizer-->>IndexIVFPQ: keys[], coarse_dis[]
    
    IndexIVFPQ->>IndexIVFPQ: search_preassigned()
    
    loop For each query (parallel)
        IndexIVFPQ->>Scanner: get_InvertedListScanner()
        IndexIVFPQ->>Scanner: set_query(query)
        Note over Scanner: Compute sim_table = M × ksub distances
        
        IndexIVFPQ->>Heap: Initialize(k)
        
        loop For each probe in nprobe
            IndexIVFPQ->>Scanner: set_list(key, coarse_dis)
            Note over Scanner: precompute_list_tables()
            
            IndexIVFPQ->>InvertedLists: get_codes(key), get_ids(key)
            InvertedLists-->>IndexIVFPQ: codes, ids
            
            IndexIVFPQ->>Scanner: scan_codes(list_size, codes, ids, heap)
            
            loop For each code (batched by 4)
                Scanner->>Scanner: Lookup sim_table[m][code[m]] for m=0..M-1
                Scanner->>Scanner: Sum distances
                alt distance < threshold
                    Scanner->>Heap: add_result(distance, id)
                    Heap->>Heap: Update threshold
                end
            end
        end
        
        IndexIVFPQ->>Heap: Extract sorted results
        Heap-->>IndexIVFPQ: distances[k], labels[k]
    end
    
    IndexIVFPQ-->>User: distances[], labels[]
```

### 7.4 Reconstruction Sequence

```mermaid
sequenceDiagram
    participant User
    participant IndexIVFPQ
    participant DirectMap
    participant InvertedLists
    participant ProductQuantizer as PQ
    participant Quantizer
    
    User->>IndexIVFPQ: reconstruct(key, recons)
    IndexIVFPQ->>DirectMap: get(key)
    DirectMap-->>IndexIVFPQ: list_no, offset
    
    IndexIVFPQ->>IndexIVFPQ: reconstruct_from_offset(list_no, offset, recons)
    IndexIVFPQ->>InvertedLists: get_single_code(list_no, offset)
    InvertedLists-->>IndexIVFPQ: PQ code
    
    IndexIVFPQ->>ProductQuantizer: decode(code, recons)
    Note over ProductQuantizer: For each m: recons[m*dsub:(m+1)*dsub] = centroid[m][code[m]]
    ProductQuantizer-->>IndexIVFPQ: decoded residual
    
    alt by_residual = true
        IndexIVFPQ->>Quantizer: reconstruct(list_no, centroid)
        Quantizer-->>IndexIVFPQ: centroid vector
        IndexIVFPQ->>IndexIVFPQ: recons += centroid
    end
    
    IndexIVFPQ-->>User: recons[] (reconstructed vector)
```

---

## 8. Distance Computation

### 8.1 Asymmetric Distance Computation (ADC)

In IVFPQ search, we use Asymmetric Distance Computation where:
- Query: full precision float vector
- Database: quantized PQ codes

```mermaid
flowchart TB
    subgraph Query["Query Processing"]
        Q["Query q"] --> DT["Compute Distance Table<br/>dis_table[m][j] = ||q_m - c_m,j||²"]
    end
    
    subgraph CodeDistance["Distance to PQ Code"]
        Code["PQ code = [q₀, q₁, ..., qₘ₋₁]"]
        Code --> Lookup["dis = Σ dis_table[m][code[m]]"]
    end
    
    DT --> Lookup
    Lookup --> Result["Approximate distance"]
```

### 8.2 Distance with Residuals (by_residual = true)

```mermaid
flowchart TB
    subgraph FullDistance["Full Distance Computation"]
        Formula["d(q, x) = ||q - centroid - residual||²"]
    end
    
    subgraph Decomposition["Decomposition"]
        Term1["Term 1: ||q - centroid||²<br/>(coarse distance from IVF)"]
        Term2["Term 2: ||residual||² + 2*(centroid|residual)<br/>(precomputed table)"]
        Term3["Term 3: -2*(q|residual)<br/>(query-specific table)"]
    end
    
    Formula --> Term1 & Term2 & Term3
    
    subgraph Computation["At Search Time"]
        Term1 --> CoarseDis["coarse_dis (from quantizer search)"]
        Term2 --> PrecompTable["Lookup precomputed_table[list_no]"]
        Term3 --> QueryTable["Lookup sim_table (computed per query)"]
        
        CoarseDis --> Total["Total = coarse_dis + precomp + query_table"]
        PrecompTable --> Total
        QueryTable --> Total
    end
```

### 8.3 Polysemous Filtering

Polysemous filtering uses the PQ codes as binary codes for fast Hamming distance pre-filtering:

```mermaid
flowchart TD
    Start[For each PQ code] --> ComputeHamming[Compute Hamming distance<br/>between query code and database code]
    ComputeHamming --> CheckHT{Hamming dist < threshold?}
    CheckHT -->|No| Skip[Skip code - likely not a good match]
    CheckHT -->|Yes| ComputeFull[Compute full PQ distance]
    ComputeFull --> AddResult[Add to results if good]
    Skip --> Next[Next code]
    AddResult --> Next
```

---

## 9. Data Structures

### 9.1 Memory Layout

```mermaid
graph TB
    subgraph IndexIVFPQ["IndexIVFPQ Memory Layout"]
        subgraph BaseFields["Base Fields"]
            d["d: dimension"]
            ntotal["ntotal: num vectors"]
            nlist["nlist: num clusters"]
        end
        
        subgraph PQFields["ProductQuantizer"]
            M["M: num sub-quantizers"]
            nbits["nbits: bits per index (8)"]
            dsub["dsub: d/M"]
            ksub["ksub: 2^nbits (256)"]
            centroids["centroids: M × ksub × dsub floats"]
        end
        
        subgraph InvLists["InvertedLists"]
            codes["codes[nlist]: PQ codes per list"]
            ids["ids[nlist]: vector IDs per list"]
        end
        
        subgraph PrecompTable["Precomputed Table (optional)"]
            table["nlist × M × ksub floats"]
        end
    end
```

### 9.2 PQ Codebook Layout

```mermaid
graph LR
    subgraph Centroids["centroids array: M × ksub × dsub"]
        subgraph M0["Subquantizer 0"]
            C0_0["c₀,₀ (dsub floats)"]
            C0_1["c₀,₁ (dsub floats)"]
            C0_255["c₀,₂₅₅ (dsub floats)"]
        end
        
        subgraph M1["Subquantizer 1"]
            C1_0["c₁,₀ (dsub floats)"]
            C1_1["c₁,₁ (dsub floats)"]
            C1_255["c₁,₂₅₅ (dsub floats)"]
        end
        
        subgraph MM["Subquantizer M-1"]
            CM_0["cₘ₋₁,₀ (dsub floats)"]
            CM_1["cₘ₋₁,₁ (dsub floats)"]
            CM_255["cₘ₋₁,₂₅₅ (dsub floats)"]
        end
    end
```

### 9.3 Inverted List Content for IVFPQ

```mermaid
graph TB
    subgraph InvertedList["Inverted List for Cluster i"]
        subgraph Codes["codes: n × M bytes"]
            Code0["Code 0: [q₀, q₁, ..., qₘ₋₁]"]
            Code1["Code 1: [q₀, q₁, ..., qₘ₋₁]"]
            CodeN["Code n-1: [q₀, q₁, ..., qₘ₋₁]"]
        end
        
        subgraph IDs["ids: n × 8 bytes"]
            ID0["Vector ID 0"]
            ID1["Vector ID 1"]
            IDN["Vector ID n-1"]
        end
        
        Code0 --- ID0
        Code1 --- ID1
        CodeN --- IDN
    end
```

### 9.4 Distance Table Layout

```mermaid
graph TB
    subgraph SimTable["sim_table: M × ksub floats"]
        subgraph Row0["Subquantizer 0"]
            D0_0["||q₀ - c₀,₀||²"]
            D0_1["||q₀ - c₀,₁||²"]
            D0_255["||q₀ - c₀,₂₅₅||²"]
        end
        
        subgraph Row1["Subquantizer 1"]
            D1_0["||q₁ - c₁,₀||²"]
            D1_1["||q₁ - c₁,₁||²"]
            D1_255["||q₁ - c₁,₂₅₅||²"]
        end
        
        subgraph RowM["Subquantizer M-1"]
            DM_0["||qₘ₋₁ - cₘ₋₁,₀||²"]
            DM_1["||qₘ₋₁ - cₘ₋₁,₁||²"]
            DM_255["||qₘ₋₁ - cₘ₋₁,₂₅₅||²"]
        end
    end
    
    subgraph Usage["Distance Computation"]
        CodeExample["Code = [42, 128, ..., 7]"]
        Computation["dis = sim_table[0][42] + sim_table[1][128] + ... + sim_table[M-1][7]"]
    end
    
    SimTable --> Computation
    CodeExample --> Computation
```

---

## 10. Performance Considerations

### 10.1 Complexity Analysis

```mermaid
graph TB
    subgraph Complexities["Operation Complexities"]
        subgraph Training["Training"]
            T1["Coarse: O(n × nlist × niter × d)"]
            T2["PQ: O(n × M × ksub × niter × dsub)"]
        end
        
        subgraph Adding["Adding n vectors"]
            A1["Coarse assign: O(n × nlist × d)"]
            A2["Residual: O(n × d)"]
            A3["PQ encode: O(n × M × ksub × dsub)"]
            A4["Storage: O(n × M) bytes"]
        end
        
        subgraph Search["Search (per query)"]
            S1["Coarse search: O(nlist × d)"]
            S2["Distance table: O(M × ksub × dsub)"]
            S3["Scan: O(nprobe × avg_list_size × M)"]
            S4["Table lookup: O(1) per code byte"]
        end
    end
```

### 10.2 Memory vs. Accuracy Trade-offs

```mermaid
graph LR
    subgraph Parameters["Parameter Impact"]
        M_high["↑ M (more sub-quantizers)"]
        M_low["↓ M (fewer sub-quantizers)"]
        
        nbits_high["↑ nbits (more centroids)"]
        nbits_low["↓ nbits (fewer centroids)"]
        
        nlist_high["↑ nlist (more clusters)"]
        nlist_low["↓ nlist (fewer clusters)"]
    end
    
    subgraph Effects["Effects"]
        M_high --> MH_acc["Better accuracy"]
        M_high --> MH_mem["More memory (M bytes/vec)"]
        
        M_low --> ML_acc["Lower accuracy"]
        M_low --> ML_mem["Less memory"]
        
        nbits_high --> NH_acc["Better subspace accuracy"]
        nbits_high --> NH_mem["Larger codebooks"]
        
        nlist_high --> NLH_speed["Faster search"]
        nlist_high --> NLH_acc["May hurt accuracy"]
    end
```

### 10.3 Optimal Parameter Selection

| Dataset Size | Recommended nlist | Recommended M | Memory per Vector |
|--------------|-------------------|---------------|-------------------|
| < 1M | 256 - 1024 | 8 - 16 | 8 - 16 bytes |
| 1M - 10M | 1024 - 4096 | 16 - 32 | 16 - 32 bytes |
| 10M - 100M | 4096 - 16384 | 32 - 64 | 32 - 64 bytes |
| > 100M | 16384 - 65536 | 64 | 64 bytes |

### 10.4 Search Speed Optimization

```mermaid
flowchart TD
    subgraph Optimizations["Speed Optimizations"]
        Precomp["use_precomputed_table = 1 or 2<br/>Pre-compute distance terms"]
        Batch["Process 4 codes at a time<br/>SIMD-friendly unrolling"]
        Polysemous["polysemous_ht > 0<br/>Hamming pre-filtering"]
        FastScan["IndexIVFPQFastScan<br/>SIMD-optimized scanning"]
    end
    
    subgraph WhenToUse["When to Use"]
        Precomp --> PrecompUse["L2 metric + by_residual<br/>Large nlist, small M×ksub×nlist"]
        Batch --> BatchUse["Always enabled internally"]
        Polysemous --> PolyUse["Very large lists<br/>Can tolerate some recall loss"]
        FastScan --> FastUse["4-bit PQ (nbits=4)<br/>Maximum throughput needed"]
    end
```

---

## Summary

IVFPQ is a sophisticated index that achieves remarkable compression and search speed through:

1. **Two-level Quantization**:
   - Coarse: IVF partitions space into nlist cells
   - Fine: PQ compresses residuals to M bytes

2. **Efficient Distance Computation**:
   - Precompute M × ksub distance table per query
   - O(M) lookups per code instead of O(d) computations

3. **Memory Efficiency**:
   - Original: d × 4 bytes per vector (e.g., 512 bytes for d=128)
   - IVFPQ: M bytes per vector (e.g., 16 bytes for M=16)
   - **32× compression** with reasonable accuracy

**Key Classes:**
- `IndexIVFPQ`: Main index combining IVF and PQ
- `ProductQuantizer`: Handles vector compression/decompression
- `IVFPQScanner`: Optimized distance computation with lookup tables
- `QueryTables`: Manages distance table computation

**Key Trade-offs:**
- `M`: More sub-quantizers = better accuracy but more memory
- `nbits`: Usually 8 (256 centroids) is optimal
- `nprobe`: More probes = better recall but slower
- `use_precomputed_table`: Trades memory for speed

IVFPQ is ideal for billion-scale search where memory is constrained but high recall is still needed.
