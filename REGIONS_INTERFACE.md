# Region Segmentation Interface Contract

## Purpose

This document defines the data interface between the **Region Segmentation** module and the **Descriptor Computation** module.
The goal is to allow both modules to be implemented independently while ensuring compatibility when integrated.

The segmentation module produces a **region map**, which the descriptor module consumes to compute region-level descriptors.

---

# 1. Input Image Format

All modules assume the input image has the following format:

* Type: `numpy.ndarray`
* Shape: `(H, W, C)`
* `H` = image height
* `W` = image width
* `C` = number of color channels (typically 3 for RGB)

Example:

```python
image.shape == (H, W, 3)
```

---

# 2. Segmentation Output Format

The segmentation module must return a **SegmentationResult** object containing:

```python
class SegmentationResult:
    region_map: np.ndarray
    num_regions: int
```

### region_map

A 2D array with shape:

```
(H, W)
```

Each pixel contains an **integer region ID**.

Example:

```
region_map

0 0 0 1 1
0 0 0 1 1
2 2 3 3 3
2 2 3 3 3
```

Interpretation:

* Pixels labeled `0` belong to Region 0
* Pixels labeled `1` belong to Region 1
* etc.

---

# 3. Region ID Rules

The segmentation module must follow these rules:

### Rule 1 — Contiguous IDs

Region IDs must be integers in the range:

```
0 to num_regions - 1
```

Valid example:

```
0,1,2,3,4
```

Invalid example:

```
0,2,7,9
```

---

### Rule 2 — Full Coverage

Every pixel in the image must belong to exactly **one region**.

No missing values or overlaps.

---

### Rule 3 — Shape Consistency

```
region_map.shape == image.shape[:2]
```

---

# 4. Segmentation Function Interface

The segmentation module must expose a function with the following signature:

```python
def segment_image(image: np.ndarray) -> SegmentationResult:
    """
    Performs region segmentation.

    Input:
        image: numpy array of shape (H, W, C)

    Output:
        SegmentationResult containing:
            region_map (H x W)
            num_regions
    """
```

The internal algorithm is flexible and may include:

* superpixels
* edge-based segmentation
* grid segmentation
* clustering
* saliency-based regions

Only the **output format must follow the contract**.

---

# 5. Descriptor Module Interface

The descriptor module consumes the segmentation result.

Expected interface:

```python
def compute_descriptors(
    image: np.ndarray,
    region_map: np.ndarray,
    num_regions: int
) -> np.ndarray:
```

Output:

```
descriptors.shape == (num_regions,)
```

Each entry corresponds to a descriptor value for one region.

Example:

```
descriptors = [0.23, 1.91, 0.55, 0.77]
```

Meaning:

```
descriptor[i] = descriptor value for region i
```

---

# 6. Mock Segmentation for Development

While the segmentation module is under development, descriptor development can use a temporary **mock segmentation**.

Example:

```python
def mock_segmentation(image):
    H, W, _ = image.shape
    region_map = np.zeros((H, W), dtype=int)

    grid_size = 8
    region_id = 0

    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * H // grid_size
            y1 = (i+1) * H // grid_size
            x0 = j * W // grid_size
            x1 = (j+1) * W // grid_size

            region_map[y0:y1, x0:x1] = region_id
            region_id += 1

    return SegmentationResult(region_map, region_id)
```

This allows descriptor development to proceed independently.

---

# 7. Future Extensions (Optional)

Segmentation may optionally provide additional metadata:

```
region_pixels: List[List[(y,x)]]
region_centroids
region_bounding_boxes
```

These are **optional optimizations** and not required for compatibility.

---

# 8. Integration Pipeline Example

Final pipeline usage:

```python
image = load_image("example.jpg")

seg = segment_image(image)

descriptors = compute_descriptors(
    image,
    seg.region_map,
    seg.num_regions
)
```

---

# Summary

Segmentation team guarantees:

```
image → region_map
```

Descriptor team guarantees:

```
image + region_map → descriptors
```

As long as the **region map contract is respected**, both modules remain compatible.
