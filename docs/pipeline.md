# Pipeline

1. Read a CloudCompare-preprocessed single plant-pot PLY/E57.
2. Remove invalid and duplicate points; run input QC.
3. Normalize the full cloud by centroid and maximum radial distance.
4. Build 7D features: XYZ, relative height, and normals (`k=20`).
5. Segment all points with pretrained DGCNN using full-cover chunk voting.
6. Export Pot and Plant point clouds.
7. Filter class outliers with DBSCAN in normalized coordinates.
8. Estimate plant height from the 14.7 cm Pot reference.
9. Build a 35 x 25 Plant CHM, smooth it, detect local maxima, and count peaks.
10. Export machine-readable results and QC figures.

Training, loss functions, ablation studies, and ground-truth evaluation are not
part of the runtime package.

