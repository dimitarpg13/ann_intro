# HNSW (Hierarchical Navigable Small World) Implementation in FAISS

## Table of Contents
1. [Introduction](#introduction)
2. [HNSW Algorithm Fundamentals](#hnsw-algorithm-fundamentals)
3. [Class Hierarchy](#class-hierarchy)
4. [Core Data Structures](#core-data-structures)
5. [Index Construction](#index-construction)
6. [Search Algorithm](#search-algorithm)
7. [Neighbor Selection Heuristics](#neighbor-selection-heuristics)
8. [Index Variants](#index-variants)
9. [Performance Considerations](#performance-considerations)
10. [Usage Examples](#usage-examples)

---

## Introduction

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor search algorithm that provides excellent query time performance with high recall. The FAISS implementation is based on the paper:

> *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"* by Yu. A. Malkov and D. A. Yashunin (arXiv 2017)

The implementation is heavily influenced by the NMSlib implementation by Yury Malkov and Leonid Boytsov.

### Key Characteristics
- **Graph-based indexing**: Vectors are organized as nodes in a multi-layer graph
- **Hierarchical structure**: Multiple layers with decreasing density from top to bottom
- **Small-world property**: Short-range and long-range connections for efficient navigation
- **Greedy search**: Navigates through the graph starting from entry point
- **Incremental construction**: Supports adding vectors one at a time

---

## HNSW Algorithm Fundamentals

### The Small World Phenomenon

HNSW exploits the "small world" graph property where most nodes can be reached from any other node through a small number of hops. This is achieved by combining:

1. **Local connections**: Links to nearby neighbors in the metric space
2. **Long-range connections**: Links that span larger distances for fast navigation

### Hierarchical Layer Structure

The index consists of multiple layers, where:
- **Layer 0 (base layer)**: Contains all vectors with the densest connections (2*M neighbors per node)
- **Higher layers**: Contain exponentially fewer nodes with M neighbors each
- **Entry point**: A node at the maximum level, used as the starting point for all searches

```mermaid
graph TB
    subgraph "Layer 2 (Sparse)"
        L2A((A))
        L2B((B))
        L2A --- L2B
    end
    
    subgraph "Layer 1 (Medium)"
        L1A((A))
        L1B((B))
        L1C((C))
        L1D((D))
        L1A --- L1B
        L1A --- L1C
        L1B --- L1D
        L1C --- L1D
    end
    
    subgraph "Layer 0 (Dense - All Vectors)"
        L0A((A))
        L0B((B))
        L0C((C))
        L0D((D))
        L0E((E))
        L0F((F))
        L0G((G))
        L0H((H))
        L0A --- L0B
        L0A --- L0C
        L0A --- L0E
        L0B --- L0D
        L0B --- L0F
        L0C --- L0D
        L0C --- L0G
        L0D --- L0H
        L0E --- L0F
        L0E --- L0G
        L0F --- L0H
        L0G --- L0H
    end
    
    L2A -.->|same node| L1A
    L2B -.->|same node| L1B
    L1A -.->|same node| L0A
    L1B -.->|same node| L0B
    L1C -.->|same node| L0C
    L1D -.->|same node| L0D
```

### Level Assignment

Each vector is randomly assigned a maximum level using an exponential distribution:

```
level = floor(-log(uniform_random(0,1)) * levelMult)
```

Where `levelMult = 1/log(M)` and M is the number of neighbors per level.

This ensures:
- Most vectors appear only in layer 0
- Progressively fewer vectors appear in higher layers
- The probability of being at level L is proportional to `exp(-L/levelMult)`

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
        +bool is_trained
        +add(n, x)*
        +search(n, x, k, distances, labels)*
        +train(n, x)*
        +reset()*
    }
    
    class HNSW {
        +storage_idx_t entry_point
        +int max_level
        +int efConstruction
        +int efSearch
        +bool check_relative_distance
        +bool search_bounded_queue
        +vector~double~ assign_probas
        +vector~int~ cum_nneighbor_per_level
        +vector~int~ levels
        +vector~size_t~ offsets
        +MaybeOwnedVector~storage_idx_t~ neighbors
        +RandomGenerator rng
        +set_default_probas(M, levelMult)
        +nb_neighbors(layer_no) int
        +neighbor_range(no, layer_no, begin, end)
        +random_level() int
        +add_with_locks(ptdis, pt_level, pt_id, locks, vt)
        +search(qdis, index, res, vt, params) HNSWStats
        +shrink_neighbor_list(qdis, input, output, max_size)
    }
    
    class MinimaxHeap {
        +int n
        +int k
        +int nvalid
        +vector~storage_idx_t~ ids
        +vector~float~ dis
        +push(i, v)
        +max() float
        +size() int
        +clear()
        +pop_min(vmin_out) int
        +count_below(thresh) int
    }
    
    class NodeDistCloser {
        +float d
        +int id
        +operator<(obj1) bool
    }
    
    class NodeDistFarther {
        +float d
        +int id  
        +operator<(obj1) bool
    }
    
    class IndexHNSW {
        +HNSW hnsw
        +bool own_fields
        +Index* storage
        +bool init_level0
        +bool keep_max_size_level0
        +add(n, x)
        +train(n, x)
        +search(n, x, k, distances, labels, params)
        +range_search(n, x, radius, result, params)
        +reconstruct(key, recons)
        +reset()
        +get_distance_computer() DistanceComputer*
    }
    
    class IndexHNSWFlat {
        <<IndexHNSW with IndexFlat storage>>
    }
    
    class IndexHNSWPQ {
        <<IndexHNSW with IndexPQ storage>>
        +train(n, x)
    }
    
    class IndexHNSWSQ {
        <<IndexHNSW with ScalarQuantizer storage>>
    }
    
    class IndexHNSW2Level {
        <<IndexHNSW with 2-level storage>>
        +flip_to_ivf()
        +search(n, x, k, distances, labels, params)
    }
    
    class IndexHNSWCagra {
        <<HNSW optimized for GPU CAGRA>>
        +bool base_level_only
        +int num_base_level_search_entrypoints
        +add(n, x)
        +search(n, x, k, distances, labels, params)
    }
    
    class IndexHNSWFlatPanorama {
        <<HNSW with Panorama progressive pruning>>
        +vector~float~ cum_sums
        +Panorama pano
        +size_t num_panorama_levels
        +add(n, x)
        +reset()
        +get_cum_sum(i) const float*
    }
    
    class SearchParametersHNSW {
        +int efSearch
        +bool check_relative_distance
        +bool bounded_queue
    }
    
    class HNSWStats {
        +size_t n1
        +size_t n2
        +size_t ndis
        +size_t nhops
        +reset()
        +combine(other)
    }
    
    class VisitedTable {
        +vector~uint8_t~ visited
        +uint8_t visno
        +set(no)
        +get(no) bool
        +advance()
    }
    
    Index <|-- IndexHNSW
    IndexHNSW <|-- IndexHNSWFlat
    IndexHNSW <|-- IndexHNSWPQ
    IndexHNSW <|-- IndexHNSWSQ
    IndexHNSW <|-- IndexHNSW2Level
    IndexHNSW <|-- IndexHNSWCagra
    IndexHNSWFlat <|-- IndexHNSWFlatPanorama
    
    IndexHNSW *-- HNSW : contains
    IndexHNSW o-- Index : storage
    
    HNSW *-- MinimaxHeap : uses
    HNSW *-- NodeDistCloser : uses
    HNSW *-- NodeDistFarther : uses
    
    SearchParameters <|-- SearchParametersHNSW
```

### Component Relationships

```mermaid
graph LR
    subgraph "Index Layer"
        IH[IndexHNSW]
        IHF[IndexHNSWFlat]
        IHPQ[IndexHNSWPQ]
        IHSQ[IndexHNSWSQ]
    end
    
    subgraph "Graph Structure"
        H[HNSW]
        MMH[MinimaxHeap]
        VT[VisitedTable]
    end
    
    subgraph "Storage Layer"
        IF[IndexFlat]
        IPQ[IndexPQ]
        ISQ[IndexScalarQuantizer]
    end
    
    subgraph "Distance Computation"
        DC[DistanceComputer]
        FDC[FlatCodesDistanceComputer]
    end
    
    IH --> H
    IH --> IF
    IHF --> IF
    IHPQ --> IPQ
    IHSQ --> ISQ
    
    H --> MMH
    H --> VT
    H --> DC
    
    IF --> FDC
```

---

## Core Data Structures

### HNSW Graph Storage

The HNSW structure stores the graph connectivity using compact arrays:

```mermaid
graph TB
    subgraph "HNSW Data Layout"
        subgraph "Level Information"
            levels["levels[]: Level of each vector<br/>e.g., [1, 3, 1, 2, 1, 1, ...]"]
        end
        
        subgraph "Offset Array"
            offsets["offsets[]: Offset into neighbors array<br/>offsets[i] = start position for vector i"]
        end
        
        subgraph "Neighbors Array"
            neighbors["neighbors[]: All neighbor links<br/>neighbors[offsets[i]:offsets[i+1]]<br/>= neighbors of vector i across all levels"]
        end
        
        subgraph "Cumulative Neighbors Per Level"
            cum["cum_nneighbor_per_level[]<br/>[0, 2*M, 2*M+M, 2*M+2*M, ...]"]
        end
    end
    
    levels --> offsets
    offsets --> neighbors
    cum --> neighbors
```

### Neighbor Storage Layout

For a vector at level L, its neighbors across all levels are stored contiguously:

```
neighbors[offsets[i] : offsets[i+1]]
= [level_0_neighbors..., level_1_neighbors..., ..., level_L_neighbors...]
```

Each level's neighbors occupy a fixed range:
- Level 0: `cum_nneighbor_per_level[0]` to `cum_nneighbor_per_level[1]` (2*M slots)
- Level k: `cum_nneighbor_per_level[k]` to `cum_nneighbor_per_level[k+1]` (M slots)

### MinimaxHeap Structure

The MinimaxHeap is a specialized heap that supports both min and max operations efficiently:

```mermaid
graph TB
    subgraph "MinimaxHeap"
        direction TB
        H["Max-Heap Structure<br/>(sorted by distance)"]
        
        subgraph "Operations"
            PUSH["push(id, distance)<br/>Insert or replace if full"]
            MAX["max()<br/>Return maximum distance (O(1))"]
            POPMIN["pop_min()<br/>Extract minimum distance (O(n))"]
            BELOW["count_below(thresh)<br/>Count elements below threshold"]
        end
        
        subgraph "Storage"
            IDS["ids[]: Vector IDs"]
            DIS["dis[]: Distances"]
            K["k: Current size"]
            N["n: Maximum size"]
        end
    end
    
    H --> PUSH
    H --> MAX
    H --> POPMIN
    H --> BELOW
    
    PUSH --> IDS
    PUSH --> DIS
```

### VisitedTable Structure

Efficient tracking of visited nodes during graph traversal:

```mermaid
graph LR
    subgraph "VisitedTable"
        V["visited[]: uint8_t array<br/>Size = total vectors"]
        VN["visno: Current visit number"]
        
        subgraph "Operations"
            SET["set(no): visited[no] = visno"]
            GET["get(no): return visited[no] == visno"]
            ADV["advance(): visno++<br/>(reset if visno >= 250)"]
        end
    end
    
    V --> SET
    V --> GET
    VN --> SET
    VN --> GET
    VN --> ADV
```

The VisitedTable uses a version number approach to avoid clearing the entire array between searches, making it O(1) to reset.

---

## Index Construction

### Add Operation Flowchart

```mermaid
flowchart TD
    START([Start: add n vectors]) --> STORE[Add vectors to storage index]
    STORE --> PREP[Prepare level table:<br/>Assign random levels to each vector]
    PREP --> SORT[Bucket sort vectors by level<br/>from highest to lowest]
    SORT --> INIT_LOCKS[Initialize OpenMP locks<br/>for all vectors]
    
    INIT_LOCKS --> LEVEL_LOOP{More levels<br/>to process?}
    LEVEL_LOOP -->|Yes| SHUFFLE[Randomly shuffle vectors<br/>at current level]
    SHUFFLE --> PAR_ADD[Parallel: For each vector]
    
    PAR_ADD --> GET_ENTRY[Get current entry point]
    GET_ENTRY --> CHECK_EMPTY{Graph empty?}
    CHECK_EMPTY -->|Yes| SET_ENTRY[Set as entry point]
    CHECK_EMPTY -->|No| GREEDY[Greedy descent from entry point<br/>to vector's level]
    
    SET_ENTRY --> NEXT_VEC
    
    GREEDY --> BUILD_LINKS[Build links at each level<br/>from vector's level down to 0]
    BUILD_LINKS --> UPDATE_ENTRY{Level > max_level?}
    UPDATE_ENTRY -->|Yes| NEW_ENTRY[Update entry point]
    UPDATE_ENTRY -->|No| NEXT_VEC
    NEW_ENTRY --> NEXT_VEC
    
    NEXT_VEC[Next vector] --> PAR_ADD
    PAR_ADD -->|Done| LEVEL_LOOP
    
    LEVEL_LOOP -->|No| DESTROY[Destroy locks]
    DESTROY --> END([End])
```

### add_with_locks Sequence Diagram

```mermaid
sequenceDiagram
    participant Main as add()
    participant HNSW as HNSW
    participant VT as VisitedTable
    participant DC as DistanceComputer
    participant Lock as OMP Locks
    
    Main->>HNSW: add_with_locks(ptdis, pt_level, pt_id, locks, vt)
    
    critical Get Entry Point
        HNSW->>HNSW: nearest = entry_point
        alt Entry point is -1
            HNSW->>HNSW: Set pt_id as entry_point
            HNSW-->>Main: Return (first vector)
        end
    end
    
    HNSW->>Lock: omp_set_lock(pt_id)
    
    HNSW->>DC: Compute distance to nearest
    
    loop For level = max_level down to pt_level+1
        HNSW->>HNSW: greedy_update_nearest()
        Note over HNSW: Find closer node at this level
    end
    
    loop For level = pt_level down to 0
        HNSW->>HNSW: add_links_starting_from()
        HNSW->>HNSW: search_neighbors_to_add()
        Note over HNSW: Find best candidates
        HNSW->>HNSW: shrink_neighbor_list()
        Note over HNSW: Apply neighbor selection heuristic
        
        loop For each selected neighbor
            HNSW->>HNSW: add_link(pt_id -> neighbor)
            HNSW->>Lock: omp_unset_lock(pt_id)
            HNSW->>Lock: omp_set_lock(neighbor)
            HNSW->>HNSW: add_link(neighbor -> pt_id)
            HNSW->>Lock: omp_unset_lock(neighbor)
            HNSW->>Lock: omp_set_lock(pt_id)
        end
    end
    
    HNSW->>Lock: omp_unset_lock(pt_id)
    
    alt pt_level > max_level
        HNSW->>HNSW: Update entry_point = pt_id
        HNSW->>HNSW: Update max_level = pt_level
    end
    
    HNSW-->>Main: Done
```

### Neighbor Search During Construction

```mermaid
flowchart TD
    START([search_neighbors_to_add]) --> INIT[Initialize candidates queue<br/>with entry point]
    INIT --> MARK[Mark entry point as visited]
    
    MARK --> LOOP{Candidates<br/>not empty?}
    LOOP -->|No| END([Return results])
    
    LOOP -->|Yes| POP[Pop nearest candidate]
    POP --> CHECK{Current distance ><br/>best result distance?}
    CHECK -->|Yes| END
    
    CHECK -->|No| GET_NEIGH[Get neighbors of current node]
    GET_NEIGH --> BATCH[Process neighbors in batches of 4]
    
    BATCH --> FOR_EACH{For each neighbor}
    FOR_EACH -->|Done| LOOP
    
    FOR_EACH -->|Next| VISITED{Already visited?}
    VISITED -->|Yes| FOR_EACH
    VISITED -->|No| SET_VISIT[Mark as visited]
    SET_VISIT --> COMPUTE[Compute distance to query]
    COMPUTE --> ADD_CHECK{Results size < efConstruction<br/>OR distance < worst result?}
    ADD_CHECK -->|Yes| ADD[Add to results and candidates]
    ADD_CHECK -->|No| FOR_EACH
    ADD --> TRIM{Results size > efConstruction?}
    TRIM -->|Yes| REMOVE[Remove worst result]
    TRIM -->|No| FOR_EACH
    REMOVE --> FOR_EACH
```

---

## Search Algorithm

### Search Flowchart

```mermaid
flowchart TD
    START([Start: search for k neighbors]) --> CHECK_EMPTY{Entry point<br/>exists?}
    CHECK_EMPTY -->|No| EMPTY_RESULT[Return empty results]
    CHECK_EMPTY -->|Yes| INIT[Initialize with entry point]
    
    INIT --> COMPUTE_ENTRY[Compute distance to entry point]
    COMPUTE_ENTRY --> UPPER_LOOP{level > 0?}
    
    UPPER_LOOP -->|Yes| GREEDY[greedy_update_nearest<br/>at current level]
    GREEDY --> DEC_LEVEL[level = level - 1]
    DEC_LEVEL --> UPPER_LOOP
    
    UPPER_LOOP -->|No| INIT_CANDIDATES[Initialize candidates heap<br/>with nearest node]
    INIT_CANDIDATES --> BOUNDED{Use bounded<br/>queue?}
    
    BOUNDED -->|Yes| BOUNDED_SEARCH[search_from_candidates<br/>with MinimaxHeap]
    BOUNDED -->|No| UNBOUNDED_SEARCH[search_from_candidate_unbounded<br/>with priority queue]
    
    BOUNDED_SEARCH --> COLLECT[Collect results from handler]
    UNBOUNDED_SEARCH --> TRIM[Trim to k results]
    TRIM --> EXTRACT[Extract results from queue]
    
    COLLECT --> ADV_VT[Advance VisitedTable]
    EXTRACT --> ADV_VT
    ADV_VT --> END([Return results])
```

### Search Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant IndexHNSW
    participant HNSW
    participant DC as DistanceComputer
    participant VT as VisitedTable
    participant RH as ResultHandler
    participant MMH as MinimaxHeap
    
    User->>IndexHNSW: search(n, x, k, distances, labels)
    
    loop For each query (parallel)
        IndexHNSW->>DC: set_query(query_vector)
        IndexHNSW->>HNSW: search(qdis, index, res, vt, params)
        
        Note over HNSW: Phase 1: Upper Layer Greedy Search
        HNSW->>HNSW: nearest = entry_point
        HNSW->>DC: d_nearest = distance(nearest)
        
        loop For level = max_level down to 1
            HNSW->>HNSW: greedy_update_nearest(level)
            
            loop Until no improvement
                HNSW->>HNSW: Get neighbors at level
                HNSW->>DC: Compute distances (batch of 4)
                HNSW->>HNSW: Update nearest if better found
            end
        end
        
        Note over HNSW: Phase 2: Layer 0 BFS Search
        HNSW->>MMH: Initialize with (nearest, d_nearest)
        
        HNSW->>HNSW: search_from_candidates()
        
        loop While candidates not empty
            HNSW->>MMH: pop_min() -> (v0, d0)
            
            alt check_relative_distance
                HNSW->>MMH: count_below(d0)
                Note over HNSW: Break if count >= efSearch
            end
            
            HNSW->>HNSW: Get neighbors of v0 at level 0
            
            loop For each neighbor v1 (batch of 4)
                HNSW->>VT: Check if visited
                alt Not visited
                    HNSW->>VT: Mark as visited
                    HNSW->>DC: Compute distance to v1
                    
                    alt distance < threshold
                        HNSW->>RH: add_result(distance, v1)
                    end
                    
                    HNSW->>MMH: push(v1, distance)
                end
            end
        end
        
        HNSW->>VT: advance()
        HNSW-->>IndexHNSW: Return stats
    end
    
    IndexHNSW-->>User: Return (distances, labels)
```

### Greedy Update Nearest

```mermaid
flowchart TD
    START([greedy_update_nearest]) --> INIT[Current: nearest, d_nearest]
    
    INIT --> LOOP{Start iteration}
    LOOP --> SAVE[prev_nearest = nearest]
    SAVE --> GET_NEIGH[Get neighbors at level]
    
    GET_NEIGH --> BATCH[Process in batches of 4]
    BATCH --> FOR_EACH{For each neighbor}
    
    FOR_EACH -->|Done| CHECK_IMPROVE{nearest ==<br/>prev_nearest?}
    FOR_EACH -->|Next| COMPUTE[Compute distance]
    COMPUTE --> COMPARE{distance < d_nearest?}
    COMPARE -->|Yes| UPDATE[nearest = neighbor<br/>d_nearest = distance]
    COMPARE -->|No| FOR_EACH
    UPDATE --> FOR_EACH
    
    CHECK_IMPROVE -->|Yes| END([Return - no improvement])
    CHECK_IMPROVE -->|No| LOOP
```

### Search from Candidates (Level 0)

```mermaid
flowchart TD
    START([search_from_candidates]) --> INIT_RES[Initialize results from candidates]
    INIT_RES --> MARK[Mark all candidates as visited]
    
    MARK --> MAIN_LOOP{Candidates<br/>not empty?}
    MAIN_LOOP -->|No| UPDATE_STATS[Update statistics]
    UPDATE_STATS --> END([Return nres])
    
    MAIN_LOOP -->|Yes| POP[pop_min() -> (v0, d0)]
    POP --> DIS_CHECK{check_relative_distance?}
    
    DIS_CHECK -->|Yes| COUNT[count = count_below(d0)]
    COUNT --> EF_CHECK{count >= efSearch?}
    EF_CHECK -->|Yes| UPDATE_STATS
    
    DIS_CHECK -->|No| STEP_CHECK{nstep > efSearch?}
    EF_CHECK -->|No| GET_NEIGH
    STEP_CHECK -->|Yes| UPDATE_STATS
    STEP_CHECK -->|No| GET_NEIGH[Get neighbors of v0]
    
    GET_NEIGH --> PREFETCH[Prefetch visited table entries]
    PREFETCH --> PROCESS_BATCH[Process neighbors in batches of 4]
    
    PROCESS_BATCH --> FOR_NEIGH{For each neighbor v1}
    FOR_NEIGH -->|Done| INC_STEP[nstep++]
    INC_STEP --> MAIN_LOOP
    
    FOR_NEIGH -->|Next| VISIT_CHECK{Already visited?}
    VISIT_CHECK -->|Yes| FOR_NEIGH
    VISIT_CHECK -->|No| MARK_VISIT[Mark as visited]
    MARK_VISIT --> COMPUTE_DIS[Compute distance]
    
    COMPUTE_DIS --> RESULT_CHECK{distance < threshold<br/>AND selector passes?}
    RESULT_CHECK -->|Yes| ADD_RESULT[Add to result handler]
    RESULT_CHECK -->|No| ADD_CAND
    ADD_RESULT --> UPDATE_THRESH[Update threshold]
    UPDATE_THRESH --> ADD_CAND[Add to candidates]
    ADD_CAND --> FOR_NEIGH
```

---

## Neighbor Selection Heuristics

### Shrink Neighbor List Algorithm

The HNSW algorithm uses a heuristic to select diverse neighbors rather than just the closest ones:

```mermaid
flowchart TD
    START([shrink_neighbor_list]) --> INIT[Sort candidates by distance<br/>nearest first]
    INIT --> OUTSIDERS[Initialize outsiders list]
    
    OUTSIDERS --> LOOP{Candidates<br/>not empty?}
    LOOP -->|No| FILL_CHECK{keep_max_size_level0<br/>AND output.size < max_size?}
    
    LOOP -->|Yes| POP[Pop nearest candidate v1]
    POP --> GOOD[good = true]
    
    GOOD --> CHECK_EXIST{For each v2<br/>in output}
    CHECK_EXIST -->|Done| GOOD_CHECK{good?}
    CHECK_EXIST -->|Next| DIST_COMPUTE[Compute dist(v1, v2)]
    DIST_COMPUTE --> CLOSER{dist(v1, v2) <<br/>dist(v1, query)?}
    CLOSER -->|Yes| BAD[good = false]
    CLOSER -->|No| CHECK_EXIST
    BAD --> GOOD_CHECK
    
    GOOD_CHECK -->|Yes| ADD_OUTPUT[Add v1 to output]
    GOOD_CHECK -->|No| KEEP_CHECK{keep_max_size_level0?}
    KEEP_CHECK -->|Yes| ADD_OUTSIDER[Add v1 to outsiders]
    KEEP_CHECK -->|No| LOOP
    ADD_OUTSIDER --> LOOP
    
    ADD_OUTPUT --> SIZE_CHECK{output.size >= max_size?}
    SIZE_CHECK -->|Yes| END([Return output])
    SIZE_CHECK -->|No| LOOP
    
    FILL_CHECK -->|Yes| FILL[Fill with outsiders]
    FILL_CHECK -->|No| END
    FILL --> END
```

### Neighbor Selection Rationale

```mermaid
graph TB
    subgraph "Simple Nearest Selection"
        Q1((Query))
        A1((A))
        B1((B))
        C1((C))
        
        Q1 -.->|d=1| A1
        Q1 -.->|d=2| B1
        Q1 -.->|d=3| C1
        
        Note1["Problem: A, B, C might<br/>all be in same direction"]
    end
    
    subgraph "HNSW Heuristic Selection"
        Q2((Query))
        A2((A))
        D2((D))
        E2((E))
        
        Q2 -.->|d=1| A2
        Q2 -.->|d=4| D2
        Q2 -.->|d=5| E2
        
        Note2["Better: A, D, E are in<br/>different directions<br/>(provides better coverage)"]
    end
    
    Note3["Selection Rule:<br/>Keep candidate only if it's<br/>not closer to any existing<br/>neighbor than to query"]
```

The heuristic ensures:
1. **Diversity**: Neighbors span different directions in the metric space
2. **Coverage**: Better exploration during search
3. **Quality**: Improved recall at the same search cost

---

## Index Variants

### IndexHNSWFlat

```mermaid
classDiagram
    class IndexHNSWFlat {
        <<Flat storage>>
        +IndexHNSWFlat()
        +IndexHNSWFlat(d, M, metric)
    }
    
    class IndexFlatL2 {
        +vector codes
        +add()
        +search()
        +reconstruct()
    }
    
    class IndexFlat {
        +vector codes
        +add()
        +search()
        +reconstruct()
    }
    
    IndexHNSWFlat *-- IndexFlatL2 : L2 metric
    IndexHNSWFlat *-- IndexFlat : other metrics
```

- **Storage**: Full precision vectors
- **Memory**: O(n * d * 4) bytes for vectors + graph overhead
- **Distance**: Exact distance computation
- **Use case**: When memory permits and accuracy is paramount

### IndexHNSWPQ

```mermaid
classDiagram
    class IndexHNSWPQ {
        <<Product Quantization storage>>
        +IndexHNSWPQ(d, pq_m, M, pq_nbits, metric)
        +train(n, x)
    }
    
    class IndexPQ {
        +ProductQuantizer pq
        +vector codes
        +train()
        +add()
        +search()
    }
    
    IndexHNSWPQ *-- IndexPQ
```

- **Storage**: PQ-compressed vectors
- **Memory**: O(n * pq_m * pq_nbits/8) bytes for vectors
- **Distance**: Approximate via lookup tables
- **Use case**: Large-scale scenarios requiring memory reduction

### IndexHNSWSQ

```mermaid
classDiagram
    class IndexHNSWSQ {
        <<Scalar Quantization storage>>
        +IndexHNSWSQ(d, qtype, M, metric)
    }
    
    class IndexScalarQuantizer {
        +ScalarQuantizer sq
        +vector codes
        +train()
        +add()
        +search()
    }
    
    IndexHNSWSQ *-- IndexScalarQuantizer
```

- **Storage**: Scalar-quantized vectors (e.g., fp16, int8)
- **Memory**: Depends on quantization type
- **Distance**: Approximate via quantized values
- **Use case**: Balance between memory and accuracy

### IndexHNSWFlatPanorama

```mermaid
classDiagram
    class IndexHNSWFlatPanorama {
        <<Progressive refinement>>
        +vector~float~ cum_sums
        +Panorama pano
        +size_t num_panorama_levels
        +add(n, x)
        +reset()
        +get_cum_sum(i) const float*
    }
    
    class Panorama {
        +size_t n_levels
        +size_t level_width_floats
        +compute_cumulative_sums()
        +compute_query_cum_sums()
    }
    
    IndexHNSWFlatPanorama *-- Panorama
```

- **Feature**: Progressive distance refinement with early pruning
- **Best for**: High-dimensional vectors (d > 512)
- **Tradeoff**: May have slightly different recall due to approximate beam ordering

---

## Performance Considerations

### Memory Layout

```mermaid
graph LR
    subgraph "Memory Components"
        subgraph "Graph Structure (HNSW)"
            L["levels: n * 4 bytes"]
            O["offsets: (n+1) * 8 bytes"]
            N["neighbors: ~n * (2M + L*M) * 4 bytes"]
        end
        
        subgraph "Vector Storage"
            V["Flat: n * d * 4 bytes"]
            VPQ["PQ: n * m bytes"]
            VSQ["SQ: n * d * qbytes"]
        end
    end
```

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| **M** | 32 | Number of neighbors per layer (higher = better recall, more memory) |
| **efConstruction** | 40 | Search depth during construction (higher = better graph, slower build) |
| **efSearch** | 16 | Search depth during query (higher = better recall, slower query) |

### Parameter Tuning Guidelines

```mermaid
graph TB
    subgraph "Construction Parameters"
        M["M (neighbors per level)<br/>- Higher: better recall<br/>- Lower: less memory, faster search"]
        EFC["efConstruction<br/>- Higher: better graph quality<br/>- Lower: faster build time"]
    end
    
    subgraph "Search Parameters"
        EFS["efSearch<br/>- Higher: better recall<br/>- Lower: faster queries"]
        BQ["search_bounded_queue<br/>- true: faster, slightly less accurate<br/>- false: more thorough search"]
    end
    
    subgraph "Trade-offs"
        T1["Memory vs Recall"]
        T2["Build Time vs Search Quality"]
        T3["Query Time vs Recall"]
    end
    
    M --> T1
    EFC --> T2
    EFS --> T3
```

### Parallelization

```mermaid
flowchart LR
    subgraph "Construction"
        C1[Parallel vector addition<br/>per level bucket]
        C2[OMP locks for<br/>concurrent updates]
    end
    
    subgraph "Search"
        S1[Parallel queries<br/>OMP for loop]
        S2[Per-thread VisitedTable<br/>and DistanceComputer]
    end
    
    subgraph "Optimizations"
        O1[Batch distance computation<br/>4 neighbors at a time]
        O2[Prefetching visited table]
        O3[SIMD MinimaxHeap operations<br/>AVX2/AVX512]
    end
```

### Complexity Analysis

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Add one vector | O(log n * M * efConstruction) | Average case |
| Search | O(log n + k * efSearch) | Average case |
| Memory | O(n * M * avg_level) | For graph structure |

---

## Usage Examples

### Basic Usage (C++)

```cpp
#include <faiss/IndexHNSW.h>

int main() {
    int d = 128;        // Dimension
    int M = 32;         // Number of neighbors
    int nb = 100000;    // Database size
    int k = 10;         // Number of results
    
    // Create index
    faiss::IndexHNSWFlat index(d, M);
    
    // Set construction parameter
    index.hnsw.efConstruction = 40;
    
    // Add vectors
    std::vector<float> xb(d * nb);
    // ... fill xb with data ...
    index.add(nb, xb.data());
    
    // Set search parameter
    index.hnsw.efSearch = 64;
    
    // Search
    std::vector<float> xq(d);
    // ... fill xq with query ...
    
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    
    index.search(1, xq.data(), k, distances.data(), labels.data());
    
    return 0;
}
```

### Using Search Parameters

```cpp
// Create custom search parameters
faiss::SearchParametersHNSW params;
params.efSearch = 128;
params.check_relative_distance = true;
params.bounded_queue = true;

// Search with parameters
index.search(nq, xq, k, distances, labels, &params);
```

### Python Usage

```python
import faiss
import numpy as np

d = 128
M = 32
nb = 100000
nq = 1000
k = 10

# Create index
index = faiss.IndexHNSWFlat(d, M)

# Set parameters
index.hnsw.efConstruction = 40
index.hnsw.efSearch = 64

# Add vectors
xb = np.random.random((nb, d)).astype('float32')
index.add(xb)

# Search
xq = np.random.random((nq, d)).astype('float32')
distances, labels = index.search(xq, k)
```

### Using Different Storage Types

```python
import faiss

d = 128
M = 32

# Flat storage (full precision)
index_flat = faiss.IndexHNSWFlat(d, M)

# PQ storage (compressed)
pq_m = 16  # Number of subquantizers
index_pq = faiss.IndexHNSWPQ(d, pq_m, M)
index_pq.train(training_data)

# Scalar quantizer storage
index_sq = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M)
index_sq.train(training_data)
```

---

## Summary

HNSW in FAISS provides a highly efficient approximate nearest neighbor search algorithm with:

1. **Multi-layer graph structure** for logarithmic search complexity
2. **Intelligent neighbor selection** using diversity heuristics
3. **Flexible storage options** (Flat, PQ, SQ) for different memory/accuracy trade-offs
4. **Highly optimized implementation** with SIMD operations and parallel processing
5. **Configurable parameters** for tuning recall/speed trade-offs

The implementation supports both construction-time and search-time parameter tuning, making it suitable for a wide range of use cases from high-accuracy scientific applications to large-scale production systems.
