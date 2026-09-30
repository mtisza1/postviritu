# postviritu

`postviritu` is a Python CLI that post-processes [EsViritu](https://github.com/cmmr/EsViritu) output. It re-aligns EsViritu's reconstructed consensus genomes against a large NCBI nucleotide database (e.g. `core_nt`) using `blastn` (remote query) or `mmseqs2`, (local query), re-derives taxonomy from the NCBI taxonomy (via `taxonkit`), supplements viral subspecies with NCBI Virus Variation genotypes when available, and provides a reviewable HTML report with proposed changes.

> [!NOTE]
> It is NOT recommended to use this tool to re-check all EsViritu taxonomy calls.
> Rather, it is most useful for checking a select set of viruses that may be:
> (A) public health concern.
> (B) virus-based vector or plasmid sequences.
>
> Therefore, only a limited set of taxa are processed by default.
>
> Also, NCBI servers will bounce you for too many requests, so be gentle.

## Why

- The EsViritu database is curated and may be missing key genomes, which can
  cause mis-assignment in the taxonomic profile.
- Virus-based vectors can cause mis-assignment.


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

### NCBI taxdump

`postviritu` resolves taxids to lineages with `taxonkit`, which needs a local
copy of the NCBI taxonomy dump (`names.dmp`, `nodes.dmp`, `delnodes.dmp`,
`merged.dmp`). Download and verify it, then extract those four files:

```bash
mkdir -p /path/to/ncbi_taxdump_dir
cd /path/to/ncbi_taxdump_dir
wget https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz \
     https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz.md5
md5sum -c taxdump.tar.gz.md5
tar -xzf taxdump.tar.gz names.dmp nodes.dmp delnodes.dmp merged.dmp
```

Pass this directory with `--taxdump /path/to/ncbi_taxdump_dir`. Alternatively,
extract the files into `~/.taxonkit`, which `taxonkit` uses when `--taxdump`
is omitted. The extended
[`new_taxdump`](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz)
archive also works, since it contains the same four files.

To check the dump, `echo 2697049 | taxonkit lineage --data-dir /path/to/ncbi_taxdump_dir`
should print the SARS-CoV-2 lineage.

NCBI updates the taxonomy continually, so refresh the dump periodically.
Otherwise hits to references newer than your dump may resolve to unknown
taxids, and older dumps use outdated names (e.g. pre-binomial virus species
names). Use the same dump for `setup-db` and for later runs.

## Usage

### Run with remote BLASTN (no local DB)

You can also search NCBI's `nt` database remotely through Biopython's QBLAST
client instead of building a local mmseqs2 database. The input and output
formats are identical to `postviritu run`; only the search backend changes.

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

### (local mmseqs) 1. Build the search database (once)

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

### (local mmseqs) 2. Run on EsViritu output

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

### Genotype enrichment (network)

For viral database hits, `postviritu` queries NCBI Virus Variation `vvsearch2`
by reference accession and uses a non-empty `Genotype` or `Lineage` as the subspecies. A
lookup is only made when the assignment is already a subspecies-level claim,
that is when it is unambiguous (not an LCA) and the hit identity is at or above
EsViritu's subspecies threshold. Non-viral hits are never queried, and results
are cached per accession for the run.

If no genotype/lineage is available, the taxdump-derived subspecies is kept. Failed
requests are retried with backoff and are *not* cached as "no genotype"; after
repeated consecutive failures the lookups are abandoned for the rest of the run
and a warning is emitted. Every run prints a tally of what the lookups did, for
example:

```
[postviritu] vvsearch2 genotype lookups: 42 queried, 17 genotyped, 25 without genotype, 0 failed, 0 skipped
```


### Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--min-identity` | `0.9` | Minimum fraction identity for an acceptable hit |
| `--min-aln-fraction` | `0.5` | Minimum fraction of canonical query bases (A, T, C, or G) aligned |
| `--max-evalue` | `1e-10` | Maximum e-value |
| `--bitscore-tie-frac` | `0.99` | Hits with bitscore ≥ frac × max are "tied" |
| `--threads` | `1` | Threads for mmseqs2 |
| `--keep-temp` | off | Keep intermediate files |
| `--taxa-filter` | high-concern list | YAML file of taxa to include, or `all` (see below) |
| `--vvsearch` / `--no-vvsearch` | on | Supplement viral subspecies with NCBI genotypes |
| `--vvsearch-email` | none | Contact address sent with `vvsearch2` requests |
| `--vvsearch-timeout` | `10` | Per-request timeout in seconds |

### Process only selected taxa

Because `postviritu` is most useful for a subset of predicted taxa, only
assemblies whose original EsViritu taxonomy matches an include-list are
re-aligned and reassigned; all others are copied through unchanged.

By default the built-in list of high-concern pathogens is used
(`HIGH_CONCERN_TAXA` in `src/postviritu/taxonomy.py`): e.g. SARS-CoV-2, MERS,
dengue, Zika, West Nile, chikungunya, mpox, variola, HIV, measles, mumps,
Lassa, CCHF, Rift Valley fever, the Ebola, Marburg, Henipa, hanta and
influenza A genera, polioviruses, EV-A71 and EV-D68. Pass `--taxa-filter all`
to process every assembly, or a YAML file to use your own list.

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
keeps all assemblies, as does `--taxa-filter all`.

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
two-column cards. Each card displays the rank of its best hit (with ties),
the species/subspecies of the reference taxon, and up to six reference
alignments with ANI, query coverage, score, e-value, and BLAST-style
query/reference sequence blocks. A searchable dropdown jumps directly to any
query of interest, and each page exposes the query's consensus FASTA sequence
in a dropdown. Assemblies skipped by `--taxa-filter` are omitted from the
report. Use the Previous/Next controls or left/right arrow keys to move between
queries.

Provenance columns include `esviritu_species`, `esviritu_subspecies`,
`postviritu_hit_accession`, `postviritu_hit_taxid`, `postviritu_bitscore`,
`postviritu_pct_identity`, `postviritu_ambiguous`, and `postviritu_decision`.
Assemblies skipped by `--taxa-filter` retain their original taxonomy and have
`postviritu_decision` set to `taxa_filtered`.

### EsViritu versions

Postviritu only accepts outputs from EsViritu versions v1 or greater.
Both the pre-1.3 and the EsViritu >= 1.3 output layouts are accepted. For 1.3+,
consensus headers of the form `{Accession}_{sample}_consensus` are mapped back
to the info-table Accession. The new `adj_taxonomy` and
`consensus_ref_identity` columns are also carried through to the rewritten
tables unchanged, as EsViritu provenance. When an assembly has no postviritu
hit, species/subspecies thresholds use `consensus_ref_identity`, falling back
to `avg_read_identity`, as EsViritu >= 1.3 does.

## Status / out of scope

The package implements both the `mmseqs2` and Biopython remote BLAST backends
via a pluggable aligner interface and does not re-map reads.
