# Student Feature-Cache Reuse During Distillation

## Context

The live cached-teacher student process was spending its startup phase building
raw feature files that the completed transfer teacher had already built for
the same mixed corpus.

## Compatibility evidence

- The teacher cache contained 1,966 feature files and the student cache 450.
- Every student cache filename overlapped with the teacher cache.
- A sampled overlapping file had identical SHA-256 content in both locations.
- Feature-cache filenames are derived from the input SHA-256, code
  fingerprint, and feature-version identity; normalization is applied only
  after raw features are loaded.

## Action and verification

While the student process remained live, 1,506 missing teacher artifacts were
added to the student cache as hard links.  460 already-existing student files
were preserved, no link failed, the resulting student cache has 1,966 files,
and 1,506 files report a shared inode.  No model weight, dataset, source file,
or live process was replaced.

## Result

The student can now reuse the immutable teacher-preprocessed inputs rather
than regenerating those 1,506 feature artifacts.
