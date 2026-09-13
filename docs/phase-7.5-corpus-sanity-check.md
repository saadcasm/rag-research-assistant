# Phase 7.5: corpus extraction sanity check

Date: 2026-09-13

This is a lightweight pre-ingestion check of the 52 approved and locally
verified PDFs in `data/corpus/pdfs/`. It uses the project's current `pypdf`
parser and existing chunking implementations. It does not introduce a parser
health framework, a second parser, OCR, new indexes, or evaluation data.

## Extraction result

All **52 of 52 PDFs (100%)** opened and produced usable text. The extraction
covered **849 pages** and produced approximately **3,247,141 Unicode characters**
(about 3.25 million characters). No PDF failed, no PDF was empty, and none of the
849 pages was empty. A standalone extraction pass took approximately **18.96
seconds** on the development Mac.

The sanity check recorded per-document page count, extracted character count,
empty pages, replacement glyphs, embedded NUL characters, and printable/alphanumeric
ratios. The smallest paper still produced 13,469 characters over five pages.
There were no obviously broken or image-only PDFs and no reason to add OCR or a
new parsing dependency.

## Targeted anomaly review

- `monot5.pdf` and `realm.pdf` caused pypdf to warn that optional `fontTools`
  support would improve parsing of particular CFF Type1 encodings. Both papers
  extracted every page, contained no Unicode replacement characters, and had
  readable titles, prose, tables, and references in sampled output. The warning
  does not represent material text corruption in this corpus, so `fontTools` was
  not added merely to suppress it.
- `ance.pdf` reached pypdf's 5,000 Form XObject traversal safeguard on page 7.
  That page still produced 3,840 readable characters. A rendered review showed
  its tables, plots, captions, and surrounding prose intact enough for retrieval.
- `faiss.pdf` contains 654 Unicode replacement characters, all on pages 8 and 9
  (125 and 529 respectively). Rendered pages show readable two-column prose and
  captions; the damaged extraction is concentrated in plot/table symbol content.
  The paper otherwise produced 65,980 characters over 12 non-empty pages. This
  is a minor local loss, not a corpus-level extraction failure.
- `bright.pdf`, `m3_embedding.pdf`, `retro.pdf`, and `splade.pdf` contain 83 NUL
  characters in total. Sampled contexts place them in mathematical notation,
  not ordinary prose. They account for about 0.003% of extracted characters and
  do not justify parser changes for this phase.

The visual spot checks were limited to the pages implicated by extraction
warnings. This keeps the check proportional to its purpose: establish whether
the current parser is broadly usable, not perform layout reconstruction.

## Ingestion path check

The existing CLI already accepts an explicit PDF directory. The new corpus was
successfully ingested with commands of this form:

```bash
rag-research-assistant ingest \
  --input data/corpus/pdfs \
  --output data/processed/corpus-52/chunks-legacy.jsonl \
  --chunking-strategy legacy
```

Equivalent runs used `boundary`, `structural`, and `semantic`. No path or
configuration change was necessary. Outputs were placed under
`data/processed/corpus-52/` so the earlier three-paper experiment artifacts were
not overwritten. As generated data, those JSONL files remain ignored by Git.

## Chunking results

All strategies used their existing defaults: 1,200-character target size and
200-character overlap. Semantic chunking used the project's already-cached local
Sentence Transformers model. Runtime is wall-clock time from `/usr/bin/time` on
the same development Mac and includes PDF extraction on every run.

| Strategy | Documents | Chunks | Mean chars | Median chars | Min–max chars | Cross-page | Section labels | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 52 | 3,793 | 1,006.2 | 1,088 | 204–1,200 | 0 | 0 | 19.33s |
| boundary | 52 | 3,510 | 1,009.7 | 1,114 | 14–1,200 | 0 | 0 | 19.38s |
| structural | 52 | 2,939 | 1,099.8 | 1,145 | 52–1,200 | 815 | 2,939 | 19.54s |
| semantic | 52 | 5,124 | 563.1 | 486 | 2–1,200 | 740 | 5,124 | 93.49s |

The semantic strategy is about 4.8 times slower than the extraction-dominated
legacy run because it embeds sentence units. Its much smaller median chunk size
also explains why it creates the most chunks. The two-character semantic minimum
and 14-character boundary minimum are worth remembering when designing the
larger evaluation set, but changing chunking policy is outside this sanity check.

## Decision

The corpus is sufficiently healthy for the next phase. All intended public PDFs
produce retrievable text, the few warnings have localized and non-material impact,
and all four current strategies complete across every document. It is reasonable
to proceed to creation of the larger evaluation set while retaining `faiss.pdf`
pages 8–9 as a known minor extraction limitation.
