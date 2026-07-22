# Preprocessing contract

Version 0.1 expects one plant-pot ROI manually prepared in CloudCompare. Record
the CloudCompare version, ordered filters, filter parameters, point counts, and
crop rule for scientific reproducibility. The pipeline performs only safe
sanitization (invalid and duplicate removal) before inference.

Do not crop away the Pot bottom/rim or Plant top. The Z axis must represent
height. Use identical preprocessing rules for train/validation/test data.

