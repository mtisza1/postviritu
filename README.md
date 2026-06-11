# postviritu

`postviritu` is a Python CLI that post-processes [EsViritu](https://github.com/cmmr/EsViritu)
output. It re-aligns EsViritu's reconstructed consensus genomes against a large
NCBI nucleotide database (e.g. `core_nt`) using `mmseqs2`, re-derives taxonomy
from the NCBI taxonomy (via `taxonkit`), and rewrites EsViritu-format output
tables.

## Why

- The EsViritu database is curated and may be missing key genomes, which can
  cause mis-assignment in the taxonomic profile.
- Virus-based vectors can cause mis-assignment.
- Genome reconstructions can be equidistant (by ANI/length) to multiple taxa
  (often strains of the same species); EsViritu does not currently declare this
  ambiguity, but it should.

`postviritu` reuses EsViritu's quantitative metrics (read counts, RPKMF,
coverage, Pi, etc.) unchanged and only updates the **taxonomy** columns, adding
provenance columns that record the original vs. new assignment.

## Installation

```bash
conda env create -f environment.yml
conda activate postviritu
```

This installs `mmseqs2`, `taxonkit` + `pytaxonkit` (>= 0.10), `polars`,
`pandas`, and the `postviritu` CLI. Taxonomy/LCA resolution is done through the
[`pytaxonkit`](https://github.com/bioforensics/pytaxonkit) library, which wraps
the `taxonkit` binary.

## Usage

### 1. Build the search database (once)

```bash
postviritu setup-db \
  --fasta /path/to/core_nt.fasta \
  --taxdump /path/to/ncbi_taxdump_dir \
  --acc2taxid /path/to/nucl_gb.accession2taxid \
  --out /path/to/postviritu_db \
  --threads 16
```

This builds an `mmseqs2` database with taxonomy and writes a
`postviritu_db.json` manifest into the output directory.

### 2. Run on EsViritu output

Batch mode (default) processes every sample prefix found in the input directory:

```bash
postviritu run \
  --input-dir /path/to/esviritu_output \
  --db /path/to/postviritu_db \
  --outdir /path/to/postviritu_output \
  --threads 16
```

Single-sample mode:

```bash
postviritu run \
  --input-dir /path/to/esviritu_output \
  --db /path/to/postviritu_db \
  --outdir /path/to/postviritu_output \
  --sample_id AYWM5R.p2126
```

### Reassignment modes

- `--mode scratch` (default): re-derive every assembly's taxonomy purely from
  the new database alignments. Assemblies with no acceptable hit are marked
  fully unclassified.
- `--mode disagree`: keep EsViritu's call unless the top database hit clearly
  disagrees (override) or the best hits tie across taxa (assign the LCA and flag
  ambiguity).

### Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--min-identity` | `0.9` | Minimum fraction identity for an acceptable hit |
| `--min-aln-fraction` | `0.5` | Minimum query alignment fraction |
| `--max-evalue` | `1e-10` | Maximum e-value |
| `--bitscore-tie-frac` | `0.99` | Hits with bitscore ≥ frac × max are "tied" |
| `--threads` | `1` | Threads for mmseqs2 |
| `--keep-temp` | off | Keep intermediate files |

## Outputs

For each sample prefix, `postviritu` writes EsViritu-format tables with updated
taxonomy plus provenance columns:

- `{PREFIX}.detected_virus.info.tsv`
- `{PREFIX}.detected_virus.assembly_summary.tsv`
- `{PREFIX}.tax_profile.tsv`
- `{PREFIX}.virus_coverage_windows.tsv` (copied unchanged)

Provenance columns include `esviritu_species`, `esviritu_subspecies`,
`postviritu_hit_accession`, `postviritu_hit_taxid`, `postviritu_bitscore`,
`postviritu_pct_identity`, `postviritu_ambiguous`, and `postviritu_decision`.

## Status / out of scope

The initial version implements the `mmseqs2` backend (a pluggable interface
allows a future BLASTN backend), and does not regenerate HTML reports or
re-map reads.
