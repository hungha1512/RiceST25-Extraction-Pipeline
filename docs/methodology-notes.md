# Methodology notes

- The final notebook DGCNN training cell uses 7D geometry, `k=20`, and 4096
  points. This is the default checkpoint contract.
- The notebook conclusion about XYZ-only features specifically discusses
  PointNet++; it does not redefine the DGCNN checkpoint input.
- The notebook CHM implementation uses a dynamic 8 mm grid, while the supplied
  publication method specifies a fixed 35 x 25 raster. Runtime follows the
  publication method.
- Notebook code sets Plant percentile to 100 although a nearby formula mentions
  99.9. Runtime defaults to the actual code value, 100; freeze one value before
  final experiments and keep manuscript/code consistent.
- Runtime uses full-cover voting for every pass. This implements the stated
  multi-pass intent and guarantees every point receives the same number of votes.

