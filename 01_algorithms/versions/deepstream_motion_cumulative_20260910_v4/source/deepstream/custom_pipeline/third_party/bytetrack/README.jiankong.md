# Vendored ByteTrack

Source: https://github.com/FoundationVision/ByteTrack
Commit: d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
Subset: deploy/ncnn/cpp tracking files only. MIT license is in LICENSE.
No NCNN detector or appearance/ReID model is included.

Local adaptations:
- Configurable detection threshold (production default 0.25); low-score floor 0.10.
- Track IDs are per BYTETracker instance, not a process-global static counter.
- Preserve detection index through activation/update/reactivation.
- Expire lost tracks before association; clear removed tombstones each frame.
- Avoid importing the entire OpenCV namespace into the pipeline.

Eigen 3.4.0 is a build-time dependency, kept outside this repository.
Official source archive: https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
SHA256: 8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
Its source distribution retains its own license files.

Inherited implementation caveats: LAPJV allocations over-allocate by sizeof(element),
and its unreachable-in-normal-use error branches call system("pause")/exit(0).
These are not introduced by this integration. Do not accept unbounded/invalid
external geometry; the adapter rejects invalid, tiny and oversized pixel boxes.
