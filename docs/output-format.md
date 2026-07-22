# Output format

```text
outputs/<sample>/
├── result.json
├── measurements.csv
├── preprocessing_report.json
├── pipeline.log
├── point_clouds/
│   ├── segmented.ply
│   ├── pot.ply
│   └── plant.ply
├── arrays/
│   ├── chm_raw.npy
│   ├── chm_smoothed.npy
│   └── tiller_peaks.json
└── figures/
    ├── segmentation.png
    ├── height_measurement.png
    └── tiller_count.png
```

`result.json` includes the model provenance, complete public configuration,
quality warnings, class counts, phenotype results, and runtime timings.

