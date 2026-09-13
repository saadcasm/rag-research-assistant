# Phase 7.5: reproducible corpus acquisition

This phase turns the reviewed 57-record corpus definition into an auditable local
PDF collection. It deliberately stops before ingestion, chunking, embeddings,
indexes, or evaluation. The output is raw source material plus provenance, not a
new retrieval experiment.

## Run it

From the repository root:

```bash
rag-research-assistant corpus-build
```

Optional paths make a rerun or a separate local corpus explicit:

```bash
rag-research-assistant corpus-build \
  --proposal docs/proposed-phase-7.5-corpus.md \
  --corpus-dir data/corpus \
  --existing-papers-dir data/papers \
  --timeout 30
```

The command derives all 57 records from the reviewed Markdown table, so the
human-reviewed definition and the machine-readable state cannot silently drift.
It creates the following structure:

```text
data/corpus/
├── manifest.jsonl       # tracked provenance and acquisition state
└── pdfs/
    ├── rag_lewis_2020.pdf
    ├── dpr_karpukhin_2020.pdf
    └── …                # generated local PDFs; ignored by Git
```

The filenames are deterministic: every record is stored as `<paper_id>.pdf`.
Never infer provenance from a browser-provided filename.

## What is and is not downloaded

The 52 records whose reviewed canonical source is a direct public PDF are
eligible for download. The three existing papers (`rag_lewis_2020`,
`dpr_karpukhin_2020`, and `sgpt`) are first reconciled from `data/papers/`, then
copied under their deterministic corpus filenames only after validation. They
are not blindly fetched again.

The five held records (`bm25`, `deepct`, `rrf`, `product_quantization`, and
`mmr`) are always represented with `download_status` set to
`held_source_unverified`; the command never contacts a source for them. This
preserves the intended benchmark definition while avoiding an unreviewed
license/source decision.

## Validation and provenance

Every JSONL record preserves bibliographic information and the reviewed source,
then records `download_status`, `sha256`, `file_size_bytes`, content type,
resolved URL, timestamp, page count, and title-word overlap when a local PDF is
available. A response must satisfy all of the following before it becomes a
corpus PDF:

1. It is at least 256 bytes and begins with the PDF magic bytes (`%PDF-`).
2. `pypdf` can open it and find a first page.
3. At least 55% of normalized expected-title words occur on the extracted first
   page. This intentionally tolerates line wrapping and punctuation changes,
   while catching an obvious wrong paper or HTML error page.
4. Its SHA-256 and byte size are written into the manifest.

The validation is a guardrail, not proof of scholarly identity. A human should
still investigate any low-quality extraction, title mismatch, redirect to an
unexpected publisher, or duplicate checksum before treating the collection as a
benchmark input.

## Idempotency and failures

On a rerun, a valid existing `data/corpus/pdfs/<paper_id>.pdf` is revalidated
and reused; it is not downloaded again. If a local file fails validation, the
command records `validation_failed` and removes the newly created invalid target
rather than retaining an unsafe file. Download and validation errors are caught
per record, so one unavailable source does not stop the remaining acquisition.

Possible statuses are:

- `verified`: downloaded, PDF-validated, title-page checked, and checksummed.
- `existing_verified`: a valid pre-existing/reconciled file was checked locally.
- `download_failed`: network or HTTP acquisition did not succeed.
- `validation_failed`: bytes were fetched or found locally but did not satisfy
  the PDF/identity checks.
- `held_source_unverified`: deliberately held record; no request was made.
- `duplicate_checksum`: two identifiers produced the same PDF bytes; investigate
  the reported paper IDs instead of silently deduplicating them.

Checksums are content identities: if a source later replaces a PDF, a changed
SHA-256 makes that material change visible. Provenance links the exact local
artifact to its reviewed source and source version. Together, they let a future
evaluation rerun distinguish a changed corpus from a changed retrieval system.

## Before ingestion

Inspect `data/corpus/manifest.jsonl` and resolve every failure, identity
mismatch, redirect anomaly, and duplicate checksum. Only after all intended
PDFs are verified should the raw corpus be passed to the existing ingestion
pipeline. That separation prevents an incomplete or accidentally altered corpus
from quietly changing embeddings, indexes, and benchmark metrics.
