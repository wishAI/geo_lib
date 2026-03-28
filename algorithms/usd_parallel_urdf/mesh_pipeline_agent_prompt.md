# Mesh Closing & Simplification Pipeline — Implementation Prompt

## Goal

Build a Python CLI tool that takes a set of open mesh fragments (STL files) extracted per-joint from an animation model, plus the original closed mesh, and produces one watertight simplified STL per joint.

## Context

The input meshes come from splitting a character/animation model by joint influence. Each piece is a subset of faces from the original closed mesh. These pieces are typically open (have boundary edges), may be ring/tube shaped (e.g. forearm segments with two open ends), and may have small floating fragments. The original complete mesh is available and is watertight — this is critical because it allows inside/outside queries to guide the repair.

## Inputs

- `--original` : Path to the original watertight STL mesh.
- `--parts-dir` : Directory containing per-joint STL files (open fragments).
- `--output-dir` : Directory for output watertight simplified STLs.
- `--target-ratio` : Float 0.0–1.0, fraction of faces to keep after simplification (default 0.25).
- `--max-hole-edges` : Max boundary loop size to attempt filling (default 500).

## Outputs

One STL per input part in `--output-dir`, each one watertight and simplified.

## Dependencies

Use the following Python libraries:

- `trimesh` — mesh loading, boundary detection, basic operations
- `numpy` — geometry math
- `scipy` — spatial queries (KDTree), sparse matrices for Laplacian smoothing
- `igl` (libigl Python bindings via `pip install libigl`) — winding number queries, quadric decimation
- `networkx` — boundary loop extraction from edge graphs (optional, can also do manually)

If `igl` is unavailable, fall back to trimesh's ray-based inside/outside test, but prefer `igl.fast_winding_number` when available.

## Algorithm — Step by Step

### Step 1: Load and Prepare the Original Mesh

```
Load original mesh with trimesh.
Compute an igl fast_winding_number precomputation structure on the original mesh.
Build a KDTree on the original mesh vertices for closest-point queries.
Create a function: is_inside(points) -> bool array
  Uses igl.fast_winding_number, threshold at 0.5.
Create a function: closest_on_surface(points) -> closest_points, distances
  Uses KDTree for approximate closest point on original mesh.
```

### Step 2: Per-Part Processing Loop

For each STL file in `--parts-dir`:

#### 2a. Load and Clean

```
Load the part mesh.
Merge duplicate vertices (trimesh.merge_vertices or equivalent) with a tolerance of 1e-6.
Remove degenerate faces (zero area or duplicate vertex indices).
Remove unreferenced vertices.
```

#### 2b. Remove Small Disconnected Components

```
Find connected components (trimesh.graph.connected_components).
Compute the surface area of each component.
Keep only components whose area is > 5% of the largest component's area.
This eliminates tiny floating fragments.
```

#### 2c. Find Boundary Loops

```
Find all boundary edges: edges that appear in exactly one face.
Build a graph of boundary edges.
Extract ordered boundary loops by walking the graph.
  - Start at any boundary vertex.
  - Walk to the next boundary vertex along a boundary edge.
  - Continue until returning to start.
  - Repeat for remaining unvisited boundary edges.
Each loop is an ordered list of vertex indices forming a closed polyline.
```

#### 2d. Fill Each Boundary Loop (Hole)

For each boundary loop with <= `--max-hole-edges` edges:

```
1. Compute loop centroid C = mean of loop vertex positions.
2. Compute loop average normal N:
   - For each consecutive triple of loop vertices, compute cross product.
   - Sum and normalize to get N.
3. Determine inward direction:
   - Test point: C_test = C + N * small_offset (use 0.1% of mesh bounding box diagonal).
   - If is_inside(C_test) is True, inward direction is +N.
   - Else try C_test = C - N * small_offset.
   - If that is inside, inward direction is -N.
   - If neither is clearly inside, use the direction where winding number is higher.
4. Classify the loop as roughly planar or non-planar:
   - Fit a best-fit plane to loop vertices (SVD of centered coordinates).
   - Compute max deviation of any loop vertex from the plane.
   - If max deviation < 5% of loop bounding box diagonal → planar.
5. Fill the hole:
   A) PLANAR CASE:
      - Project loop vertices onto the best-fit plane (2D).
      - Compute a constrained Delaunay triangulation of the 2D polygon.
      - Map resulting triangles back to 3D vertex indices.
      - Orient new faces so their normal agrees with the inward direction.
   B) NON-PLANAR CASE:
      - Use an ear-clipping approach on the 3D loop:
        - While loop has more than 3 vertices:
          - For each vertex v_i in the loop, consider triangle (v_{i-1}, v_i, v_{i+1}).
          - Compute triangle normal. Accept the ear if:
            - Normal roughly agrees with local mesh normal direction.
            - No other loop vertex falls inside the triangle.
            - Triangle centroid passes the is_inside test OR is close to original surface.
          - Score valid ears by triangle quality (aspect ratio). Pick the best.
          - Remove the ear vertex from the loop, add the triangle to the fill.
        - Add the final 3-vertex triangle.
      - If ear-clipping fails (no valid ear found), fall back to fan triangulation
        from the centroid, then project centroid onto original surface using
        closest_on_surface.
6. After filling, for each new fill vertex (if any were created, e.g. centroid):
   - Project to closest point on original mesh surface.
   - Verify is_inside is consistent (the point should be on or very near the surface).
```

#### 2e. Fix Normal Consistency

```
After all holes are filled:
1. Build face adjacency graph.
2. BFS/DFS from an arbitrary seed face.
3. For each neighbor: if the shared edge has consistent winding, keep the normal;
   if inconsistent, flip the neighbor face.
4. After consistent orientation: pick any face, compute its centroid,
   offset slightly along its normal, test is_inside.
   - If the offset point is INSIDE, all normals are pointing inward → flip ALL faces.
   - If OUTSIDE, normals are correct (pointing outward).
```

#### 2f. Verify Watertightness

```
Recompute boundary edges after filling.
If boundary edges remain (holes we skipped or filling failed):
  - Log a warning with the part filename and number of remaining boundary edges.
  - Attempt: for very small remaining loops (< 10 edges), force a fan fill from centroid.
Check: every edge should have exactly 2 adjacent faces.
Check: Euler characteristic V - E + F = 2 (for genus 0).
Log pass/fail per part.
```

#### 2g. Simplify with Quadric Error Metrics

```
If igl is available:
  Use igl.decimate with target face count = original_face_count * target_ratio.
  This implements QEM (Garland-Heckbert) decimation.

If igl is not available:
  Use trimesh.simplify_quadric_decimation (if available) or
  call meshlab's quadric edge collapse via pymeshlab.

After simplification:
  - Re-verify watertightness (edge collapse can occasionally create issues).
  - If broken, attempt trimesh.fill_holes as a last resort.
```

#### 2h. Post-Simplification Smoothing (Optional)

```
Apply 3-5 iterations of Laplacian smoothing ONLY to vertices that were part of
hole-fill faces (tag them during step 2d).
This blends the caps with the rest of the geometry.
Do NOT smooth original mesh vertices to preserve detail.
Use cotangent weights if available (igl.cotmatrix), else uniform weights.
Constrain smoothed vertices to stay within a distance threshold of original surface
using closest_on_surface.
```

#### 2i. Export

```
Save the final mesh as STL (binary) to output-dir/original_filename.stl.
```

### Step 3: Summary Report

```
After all parts are processed, print a summary table:
- Part filename
- Original face count
- Final face count
- Holes found / holes filled
- Watertight: yes/no
- Euler characteristic
- Any warnings
```

## Edge Cases to Handle

1. **Empty parts**: Some joints may have 0 faces. Skip and log.
2. **Already closed parts**: No boundary edges. Skip hole filling, go straight to simplification.
3. **Nested boundary loops**: A part might have holes within holes (genus > 0). Fill the largest loops first. Accept that the result may not be genus-0 in rare cases.
4. **Self-intersecting fill**: After filling, if possible, check for self-intersections. Log a warning but don't fail.
5. **Very thin parts**: Some joint regions may be nearly degenerate (e.g., a single strip of faces). Detect parts with near-zero volume after closing and log a warning.
6. **Non-manifold edges**: Edges shared by 3+ faces. Before any processing, split non-manifold edges by duplicating vertices until every edge has at most 2 faces.

## Testing Criteria

The implementation is correct if:

1. Running on a set of open fragments from a known character mesh produces all-watertight outputs.
2. Every output has Euler characteristic 2.
3. Every output has 0 boundary edges.
4. Face count is within ±10% of the target ratio.
5. No output mesh has inverted normals (all normals point outward, verifiable by checking that the signed volume is positive).
6. The tool handles edge cases (empty parts, already-closed parts) without crashing.

## Performance Notes

- For meshes under 100k faces per part, this should run in seconds per part.
- The winding number precomputation on the original mesh is the most expensive step — do it once and reuse.
- KDTree construction is also one-time.
- If processing 50+ parts, consider multiprocessing with a shared read-only original mesh structure.
