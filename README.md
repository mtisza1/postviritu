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

### Run with BLASTN -remote (no local DB)

You can also search NCBI's `nt` database remotely with `blastn` instead of
building a local mmseqs2 database. The input and output formats are identical
to `postviritu run`; only the search backend changes.

```bash
postviritu blastn \
  --input-dir /path/to/esviritu_output \
  --outdir /path/to/postviritu_output \
  --taxdump /path/to/ncbi_taxdump_dir \
  --threads 1
```

Remote searches are sent in batches of up to **3 query sequences at a time**
and each batch is awaited before the next is submitted. Use `--batch-size`
to change this default and `--max-target-seqs` to control how many subject
hits are reported per query.

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
| `--taxa-filter` | off | YAML file of taxa to include (see below) |

### Process only selected taxa

Because `postviritu` is most useful for a subset of predicted taxa, you can
provide a YAML include-list. Only assemblies whose original EsViritu taxonomy
matches an entry are re-aligned and reassigned; all others are copied through
unchanged.

Create a file such as `taxa_to_process.yaml`:

```yaml
species:
  - "s__betacoronavirus pandemicum"
  - "s__Human mastadenovirus A"
genus:
  - "g__Enterovirus"
```

Then pass it to `run` or `blastn`:

```bash
postviritu run \
  --input-dir /path/to/esviritu_output \
  --db /path/to/postviritu_db \
  --outdir /path/to/postviritu_output \
  --taxa-filter taxa_to_process.yaml
```

Rank keys may be any of `kingdom`, `phylum`, `class` (or `tclass`), `order`,
`family`, `genus`, `species`, or `subspecies`. Values may be given with or
without the EsViritu rank prefix (`s__...`) and are matched case-insensitively.
The filter uses OR semantics across rank entries. If any accession in an
assembly matches, `postviritu` processes that entire assembly. Each sample
reports the matching assembly count and warns if none match.

To guard against typos that would silently skip every assembly, `postviritu`
rejects a filter file that names ranks but lists no taxa under any of them
(`species:` with an empty or omitted list), one that is not a rank -> list
mapping (a bare top-level list), and entries filed under the wrong rank
(`genus: ["s__Human mastadenovirus A"]`). An empty file is still valid and
keeps all assemblies, as does omitting `--taxa-filter`.

## Outputs

For each sample prefix, `postviritu` writes EsViritu-format tables with updated
taxonomy plus provenance columns:

- `{PREFIX}.detected_virus.info.tsv`
- `{PREFIX}.detected_virus.assembly_summary.tsv`
- `{PREFIX}.tax_profile.tsv`
- `{PREFIX}.postviritu_report.html`
- `{PREFIX}.virus_coverage_windows.tsv` (copied unchanged)

The self-contained HTML report has one paginated view per query, shows the
original-to-proposed taxonomy change, and groups up to six database taxa into
two-column cards. Each card includes up to six reference alignments with ANI,
query coverage, score, e-value, and BLAST-style query/reference sequence blocks.
Use the Previous/Next controls or left/right arrow keys to move between queries.

Provenance columns include `esviritu_species`, `esviritu_subspecies`,
`postviritu_hit_accession`, `postviritu_hit_taxid`, `postviritu_bitscore`,
`postviritu_pct_identity`, `postviritu_ambiguous`, and `postviritu_decision`.
Assemblies skipped by `--taxa-filter` retain their original taxonomy and have
`postviritu_decision` set to `taxa_filtered`.

## Status / out of scope

The package implements both the `mmseqs2` and `blastn -remote` backends via a
pluggable aligner interface and does not re-map reads.
