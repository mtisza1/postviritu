"""NCBI taxonomy resolution via pytaxonkit, mapped onto EsViritu's 8 ranks.

EsViritu uses a fixed 8-rank lineage with GTDB-style prefixes:
``k__`` (kingdom), ``p__`` (phylum), ``c__`` (class -> ``tclass``), ``o__``
(order), ``f__`` (family), ``g__`` (genus), ``s__`` (species), ``t__``
(subspecies/strain). For viruses the "kingdom" slot is filled by NCBI's
top-level rank (``superkingdom`` / ``acellular root`` / ``domain`` = "Viruses").
Missing intermediate ranks are filled with ``unclassified_<nearest defined
ancestor>`` to mirror EsViritu's output style.

Lineage and LCA resolution use the `pytaxonkit <https://github.com/bioforensics/
pytaxonkit>`_ library (>= 0.10), which wraps the ``taxonkit`` binary and returns
pandas DataFrames.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import requests
import yaml

from . import __version__
from .io_esviritu import RANK_PREFIXES, TAX_RANKS

logger = logging.getLogger(__name__)

# Map each EsViritu rank to the NCBI rank name(s) that fill it, in priority order.
# The "kingdom" slot prefers the virus top-level rank ("Viruses"); newer NCBI
# taxdumps label this ``acellular root``/``domain`` rather than ``superkingdom``.
_NCBI_RANK_SOURCES = {
    "kingdom": ["superkingdom", "acellular root", "domain", "kingdom"],
    "phylum": ["phylum"],
    "tclass": ["class"],
    "order": ["order"],
    "family": ["family"],
    "genus": ["genus"],
    "species": ["species"],
    "subspecies": [
        "subspecies",
        "strain",
        "serotype",
        "terminal no rank",
        "no rank",
    ],
}

_DEFAULT_ROOT = "Viruses"

# ``vvsearch2`` is the Solr backend behind the NCBI Virus Variation web UI. It
# is not a versioned E-utilities endpoint and carries no stability contract, so
# every lookup is treated as best-effort: failures degrade to taxdump-derived
# taxonomy and are counted (see :class:`VVSearchStats`) rather than swallowed.
_VVSEARCH_URL = "https://www.ncbi.nlm.nih.gov/genomes/VirusVariation/vvsearch2/"
_ACCESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass
class VVSearchConfig:
    """Client policy for NCBI Virus Variation ``vvsearch2`` lookups.

    ``enabled=False`` makes the pipeline fully offline and deterministic. The
    rate limit and identification fields follow NCBI's usage guidance (no more
    than 3 requests/second, and identify the client).
    """

    enabled: bool = True
    timeout: float = 10.0
    min_interval: float = 0.34  # NCBI asks for <= 3 requests/second.
    max_attempts: int = 3  # Per accession, within one lookup.
    backoff: float = 1.0  # Seconds; doubled after each failed attempt.
    max_consecutive_failures: int = 5  # Then stop querying for the whole run.
    batch_size: int = 100  # Accessions per request (one Solr OR query).
    tool: str = "postviritu"
    email: Optional[str] = None


@dataclass
class VVSearchStats:
    """Tally of what the genotype lookups actually did during a run.

    Without this, an air-gapped run and a run where NCBI genuinely has no
    genotypes produce identical output, which makes the enrichment step
    impossible to audit after the fact.
    """

    attempted: int = 0  # Accessions we sent at least one request for.
    genotyped: int = 0  # Answered with a usable genotype.
    empty: int = 0  # Answered, but no genotype recorded.
    failed: int = 0  # Exhausted all attempts.
    skipped: int = 0  # Never attempted (disabled, circuit open, unparseable).
    circuit_open: bool = False  # Lookups abandoned after repeated failures.

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        """One-line, human-readable provenance for the run log."""
        parts = [
            f"{self.attempted} queried",
            f"{self.genotyped} genotyped",
            f"{self.empty} without genotype",
            f"{self.failed} failed",
            f"{self.skipped} skipped",
        ]
        text = "vvsearch2 genotype lookups: " + ", ".join(parts)
        if self.circuit_open:
            text += " (abandoned early after repeated failures)"
        return text


def _normalize_accession(value) -> Optional[str]:
    """Extract an accession.version from a database sequence identifier.

    Returns ``None`` for anything that does not look like an accession, so a
    malformed identifier degrades to "no genotype" instead of raising.
    """
    parts = str(value).split(maxsplit=1)
    if not parts:  # Empty or whitespace-only identifier.
        return None
    text = parts[0]
    fields = text.split("|")
    if len(fields) >= 4 and fields[0].lower() == "gi" and fields[1].isdigit():
        text = fields[3]
    elif len(fields) >= 2 and fields[0].lower() in {"gb", "emb", "dbj", "ref"}:
        text = fields[1]
    text = text.strip("|")
    return text if _ACCESSION_PATTERN.fullmatch(text) else None


def _sanitize_genotype(value) -> Optional[str]:
    """Collapse whitespace in an external genotype string, or return None.

    ``Genotype_s`` is untrusted external text that lands in a TSV cell, so any
    embedded tab or newline has to be neutralised before it reaches output.
    """
    if value is None:
        return None
    text = _WHITESPACE_RUN.sub(" ", str(value)).strip()
    return text or None


def map_ranks_to_esviritu(
    rank_to_name: Dict[str, str], root_default: str = _DEFAULT_ROOT
) -> Dict[str, str]:
    """Map a {ncbi_rank: name} dict to the 8 EsViritu ranks (with prefixes).

    Missing ranks are filled as ``<prefix>unclassified_<nearest ancestor>``.
    The ``subspecies`` slot uses an explicit sub-species rank or the queried
    terminal ``no rank`` node when it descends from a species.
    """
    out: Dict[str, str] = {}
    last_real: Optional[str] = None
    for rank in TAX_RANKS:
        prefix = RANK_PREFIXES[rank]
        name = None
        for src in _NCBI_RANK_SOURCES[rank]:
            if rank == "subspecies" and src == "no rank":
                continue  # avoid grabbing arbitrary 'no rank' nodes
            if src in rank_to_name and rank_to_name[src]:
                name = rank_to_name[src]
                break
        if name:
            out[rank] = prefix + name
            last_real = name
        elif rank == "subspecies":
            # EsViritu mirrors the species name at subspecies when no strain
            # rank exists (e.g. t__Human mastadenovirus B). Carry over the
            # species core (which may itself be 'unclassified_...').
            species_core = out.get("species", "s__").removeprefix(
                RANK_PREFIXES["species"]
            )
            out[rank] = prefix + species_core
        else:
            anchor = last_real if last_real else root_default
            out[rank] = prefix + "unclassified_" + anchor
    return out


def unclassified_lineage(root_default: str = _DEFAULT_ROOT) -> Dict[str, str]:
    """Return a fully-unclassified 8-rank lineage."""
    return {
        rank: RANK_PREFIXES[rank] + "unclassified_" + root_default
        for rank in TAX_RANKS
    }


def _is_missing(value) -> bool:
    """True if a pandas cell is NaN/None/empty (avoids importing pandas)."""
    if value is None:
        return True
    # pandas represents missing strings as float('nan'); NaN != NaN.
    if isinstance(value, float) and value != value:
        return True
    return value == ""


class Taxonomy:
    """Resolve taxids to lineages and compute LCAs using pytaxonkit (>= 0.10)."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        threads: Optional[int] = None,
        vvsearch: Optional[VVSearchConfig] = None,
    ):
        self.data_dir = data_dir
        self.threads = threads
        self.vvsearch = vvsearch if vvsearch is not None else VVSearchConfig()
        self.vvsearch_stats = VVSearchStats()
        self._lineage_cache: Dict[str, Dict[str, str]] = {}
        # Successful answers only. A transient failure must never be cached as
        # a negative, or it becomes indistinguishable from "no genotype exists"
        # for the rest of the run.
        self._genotype_cache: Dict[str, Optional[str]] = {}
        self._consecutive_failures = 0
        self._circuit_open = False
        self._last_request_at: Optional[float] = None

    @staticmethod
    def _pytaxonkit():
        try:
            import pytaxonkit
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pytaxonkit (>= 0.10) is required. Install it with "
                "`conda install -c bioconda pytaxonkit` or `pip install pytaxonkit`."
            ) from exc
        return pytaxonkit

    def rank_maps(self, taxids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        """Return {taxid: {ncbi_rank: name}} for the given taxids (cached).

        Uses ``pytaxonkit.lineage``; the ``FullLineage`` and ``FullLineageRanks``
        columns are semicolon-separated parallel lists of names and ranks.
        """
        unique = [str(t) for t in dict.fromkeys(taxids) if t and str(t) != "0"]
        missing = [t for t in unique if t not in self._lineage_cache]
        if missing:
            pt = self._pytaxonkit()
            df = pt.lineage(missing, data_dir=self.data_dir, threads=self.threads)
            for row in df.itertuples(index=False):
                taxid = str(row.TaxID)
                self._lineage_cache[taxid] = self._row_to_rank_map(
                    getattr(row, "FullLineage", None),
                    getattr(row, "FullLineageRanks", None),
                    getattr(row, "Name", None),
                    getattr(row, "Rank", None),
                )
            # Any taxid that produced no output gets an empty map.
            for t in missing:
                self._lineage_cache.setdefault(t, {})
        return {t: self._lineage_cache.get(t, {}) for t in unique}

    @staticmethod
    def _row_to_rank_map(
        names, ranks, terminal_name=None, terminal_rank=None
    ) -> Dict[str, str]:
        rank_map: Dict[str, str] = {}
        if not _is_missing(names) and not _is_missing(ranks):
            for name, rank in zip(str(names).split(";"), str(ranks).split(";")):
                if name and rank:
                    rank_map[rank] = name
        if not _is_missing(terminal_name) and not _is_missing(terminal_rank):
            terminal_name = str(terminal_name)
            terminal_rank = str(terminal_rank)
            if terminal_rank == "no rank" and "species" in rank_map:
                rank_map["terminal no rank"] = terminal_name
            elif terminal_rank != "no rank":
                rank_map[terminal_rank] = terminal_name
        return rank_map

    def _throttle(self) -> None:
        """Space requests out to honour the configured rate limit."""
        interval = self.vvsearch.min_interval
        if interval > 0 and self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < interval:
                time.sleep(interval - elapsed)
        self._last_request_at = time.monotonic()

    def _vvsearch_request(
        self, accessions: Sequence[str], label: str = "vvsearch2 request"
    ) -> Optional[Dict[str, Optional[str]]]:
        """Query vvsearch2 for a batch of accessions in a single request.

        Returns ``{accession: genotype}`` for every requested accession
        (``None`` where NCBI holds no genotype), or ``None`` when every attempt
        failed, which the caller must not confuse with an authoritative "these
        references have no genotype". ``label`` prefixes the progress log lines.
        """
        cfg = self.vvsearch
        terms = " OR ".join(f'"{accession}"' for accession in accessions)
        params = {
            "fq": 'SeqType_s:("Nucleotide")',
            "q": f"AccVer_s:({terms})",
            "fl": "AccVer_s,Genotype_s,Lineage_s",
            "wt": "json",
            "rows": len(accessions),
        }
        # NCBI asks callers to identify themselves so they can contact the
        # owner of a misbehaving client instead of blocking it outright.
        if cfg.tool:
            params["tool"] = cfg.tool
        if cfg.email:
            params["email"] = cfg.email
        headers = {"User-Agent": f"{cfg.tool or 'postviritu'}/{__version__}"}

        delay = cfg.backoff
        max_attempts = max(1, cfg.max_attempts)
        for attempt in range(1, max_attempts + 1):
            self._throttle()
            logger.info("%s (attempt %d/%d): sending", label, attempt, max_attempts)
            started = time.monotonic()
            try:
                response = requests.get(
                    _VVSEARCH_URL, params=params, headers=headers, timeout=cfg.timeout
                )
                response.raise_for_status()
                docs = response.json()["response"]["docs"]
                # SARS-CoV-2 records carry only a Pango lineage (Lineage_s);
                # records with both (e.g. mpox IIb / F.1) keep Genotype_s.
                found = {
                    doc["AccVer_s"]: _sanitize_genotype(doc.get("Genotype_s"))
                    or _sanitize_genotype(doc.get("Lineage_s"))
                    for doc in docs
                    if doc.get("AccVer_s")
                }
                answers = {accession: found.get(accession) for accession in accessions}
                logger.info(
                    "%s: answered in %.1fs, %d/%d with a genotype",
                    label,
                    time.monotonic() - started,
                    sum(1 for genotype in answers.values() if genotype),
                    len(accessions),
                )
                return answers
            except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                retry = (
                    f"retrying in {delay:.1f}s" if attempt < max_attempts else "giving up"
                )
                logger.warning(
                    "%s (attempt %d/%d) failed after %.1fs: %s: %s; %s",
                    label,
                    attempt,
                    max_attempts,
                    time.monotonic() - started,
                    type(exc).__name__,
                    exc,
                    retry,
                )
                if attempt < max_attempts:
                    if delay > 0:
                        time.sleep(delay)
                    delay *= 2
        return None

    def genotypes(
        self, accessions: Sequence[str], allow_lookup: bool = True
    ) -> Dict[str, Optional[str]]:
        """Return ``{accession: Virus Variation genotype}`` for reference ids.

        Keys are the identifiers as given (e.g. ``gi|…|gb|PQ065590.1|``); values
        are ``None`` when no genotype is known. Uncached accessions are looked
        up in batches of ``VVSearchConfig.batch_size`` per request. Always
        degrades to ``None`` rather than raising: the genotype is a supplement
        to taxdump-derived taxonomy, never a prerequisite for it. With
        ``allow_lookup=False`` only the cache is consulted, so read-only
        consumers cannot introduce a genotype that reassignment never resolved.
        """
        normalized = {raw: _normalize_accession(raw) for raw in dict.fromkeys(accessions)}
        if allow_lookup:
            unparseable = sum(acc is None for acc in normalized.values())
            self.vvsearch_stats.skipped += unparseable
            distinct = [acc for acc in dict.fromkeys(normalized.values()) if acc is not None]
            pending = [acc for acc in distinct if acc not in self._genotype_cache]
            if pending:
                self._lookup_pending(pending, cached=len(distinct) - len(pending), unparseable=unparseable)
        return {
            raw: self._genotype_cache.get(acc) if acc else None
            for raw, acc in normalized.items()
        }

    def _lookup_pending(self, pending: List[str], cached: int, unparseable: int) -> None:
        """Look up uncached accessions in batches, logging progress."""
        stats = self.vvsearch_stats
        if not self.vvsearch.enabled or self._circuit_open:
            reason = "disabled (--no-vvsearch)" if not self.vvsearch.enabled else (
                "abandoned after repeated failures"
            )
            logger.info(
                "vvsearch2: lookups %s; skipping %d reference accession(s)",
                reason,
                len(pending),
            )
            stats.skipped += len(pending)
            return

        batch_size = max(1, self.vvsearch.batch_size)
        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        logger.info(
            "vvsearch2: %d reference accession(s) to query in %d request(s) of <= %d "
            "(%d already cached, %d unparseable)",
            len(pending),
            len(batches),
            batch_size,
            cached,
            unparseable,
        )
        before = (stats.genotyped, stats.empty, stats.failed, stats.skipped)
        started = time.monotonic()
        queried = 0
        for index, batch in enumerate(batches, start=1):
            label = (
                f"vvsearch2 request {index}/{len(batches)}: {len(batch)} accession(s), "
                f"{queried + 1}-{queried + len(batch)} of {len(pending)} "
                f"({100 * (queried + len(batch)) / len(pending):.1f}% cumulative)"
            )
            self._lookup_batch(batch, label)
            queried += len(batch)
        genotyped, empty, failed, skipped = (
            now - then
            for now, then in zip(
                (stats.genotyped, stats.empty, stats.failed, stats.skipped), before
            )
        )
        logger.info(
            "vvsearch2: finished %d accession(s) in %.1fs: %d genotyped, %d without "
            "genotype, %d failed, %d skipped",
            len(pending),
            time.monotonic() - started,
            genotyped,
            empty,
            failed,
            skipped,
        )

    def _lookup_batch(self, batch: List[str], label: str = "vvsearch2 request") -> None:
        """Resolve one batch of normalized accessions into the genotype cache."""
        stats = self.vvsearch_stats
        if not self.vvsearch.enabled or self._circuit_open:
            stats.skipped += len(batch)
            return

        stats.attempted += len(batch)
        answers = self._vvsearch_request(batch, label)
        if answers is None:
            stats.failed += len(batch)
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.vvsearch.max_consecutive_failures:
                # Offline or blocked: stop paying the timeout on every
                # remaining reference and say so once.
                self._circuit_open = True
                stats.circuit_open = True
                logger.warning(
                    "Abandoning vvsearch2 genotype lookups after %d consecutive "
                    "failures; remaining assemblies keep taxdump-derived "
                    "subspecies. Use --no-vvsearch to make this explicit.",
                    self._consecutive_failures,
                )
            return

        self._consecutive_failures = 0
        for accession, genotype in answers.items():
            self._genotype_cache[accession] = genotype
            if genotype:
                stats.genotyped += 1
            else:
                stats.empty += 1

    def _vvsearch_genotype(
        self, accession: str, allow_lookup: bool = True
    ) -> Optional[str]:
        """Return the Virus Variation genotype for a single reference accession."""
        return self.genotypes([accession], allow_lookup=allow_lookup)[accession]

    def viral_taxids(self, taxids: Sequence[str]) -> set:
        """Return the subset of ``taxids`` whose top-level lineage is Viruses."""
        return {
            taxid
            for taxid, rmap in self.rank_maps(taxids).items()
            if rmap
            and map_ranks_to_esviritu(rmap)["kingdom"]
            .removeprefix(RANK_PREFIXES["kingdom"])
            .casefold()
            == _DEFAULT_ROOT.casefold()
        }

    def esviritu_lineage(
        self,
        taxid: str,
        accession: Optional[str] = None,
        allow_lookup: bool = True,
    ) -> Dict[str, str]:
        """Return the 8-rank EsViritu lineage for a taxid and optional accession.

        Passing ``accession`` supplements the subspecies slot with a Virus
        Variation genotype for viral taxa. ``allow_lookup=False`` restricts
        that to genotypes already resolved during this run, which is what
        read-only consumers (such as the HTML report) want: it keeps them
        consistent with the output tables and free of network traffic.
        """
        rmap = self.rank_maps([taxid]).get(str(taxid), {})
        if not rmap:
            return unclassified_lineage()
        lineage = map_ranks_to_esviritu(rmap)
        kingdom = lineage["kingdom"].removeprefix(RANK_PREFIXES["kingdom"])
        if accession and kingdom.casefold() == _DEFAULT_ROOT.casefold():
            genotype = self._vvsearch_genotype(accession, allow_lookup=allow_lookup)
            if genotype:
                lineage["subspecies"] = RANK_PREFIXES["subspecies"] + genotype
        return lineage

    def lca(self, taxids: Sequence[str]) -> Optional[str]:
        """Compute the lowest common ancestor taxid via ``pytaxonkit.lca``."""
        clean = [str(t) for t in dict.fromkeys(taxids) if t and str(t) != "0"]
        if not clean:
            return None
        if len(clean) == 1:
            return clean[0]
        pt = self._pytaxonkit()
        result = pt.lca(
            [int(t) for t in clean],
            skip_deleted=True,
            skip_unfound=True,
            data_dir=self.data_dir,
            threads=self.threads,
        )
        if not result or int(result) == 0:
            return None
        return str(result)


TAXA_FILTER_ALL = "all"

# Default include-list: pathogens of high public-health concern, matched
# against the original EsViritu lineage. Kept in code rather than a data file
# so it cannot go missing from an installed package.
HIGH_CONCERN_TAXA: Dict[str, List[str]] = {
    "species": [
        "s__betacoronavirus pandemicum",
        "s__Orthoflavivirus denguei",
        "s__Orthoflavivirus kyasanurense",
        "s__Orthoflavivirus nilense",
        "s__Orthoflavivirus zikaense",
        "s__Alphavirus chikungunya",
        "s__Orthobunyavirus oropoucheense",
        "s__Orthoflavivirus louisense",
        "s__Venezuelan equine encephalitis virus",
        "s__Eastern equine encephalitis virus",
        "s__Orthoflavivirus flavi",
        "s__Middle East respiratory syndrome-related coronavirus",
        "s__Orthonairovirus haemorrhagiae",
        "s__Mammarenavirus lassaense",
        "s__Phlebovirus riftense",
        "s__Lentivirus humimdef1",
        "s__Lentivirus humimdef2",
        "s__Hepacivirus hominis",
        "s__Morbillivirus hominis",
        "s__Orthorubulavirus parotitidis",
        "s__Orthopoxvirus variola",
        "s__Orthopoxvirus monkeypox",
    ],
    "genus": [
        "g__Ebolavirus",
        "g__Orthomarburgvirus",
        "g__Henipavirus",
        "g__Mammarenavirus",
        "g__Orthohantavirus",
        "g__Alphavirus",
        "g__Alphainfluenzavirus",
    ],
    "subspecies": [
        "t__Enterovirus A71",
        "t__enterovirus D68",
        "t__Poliovirus 1",
        "t__Poliovirus 2",
        "t__Poliovirus 3",
        "t__Human poliovirus 1",
        "t__Human poliovirus 2",
        "t__Human poliovirus 3",
    ],
}


class TaxaFilter:
    """Include-list filter for query assemblies based on original EsViritu taxonomy.

    The YAML file maps rank names (``species``, ``genus``, etc.) to lists of
    taxon strings. Values may be given with or without the EsViritu rank
    prefix (``s__Human mastadenovirus A`` or ``Human mastadenovirus A``);
    missing prefixes are added automatically from the rank key. An assembly is
    included if any of its original lineage ranks matches the include list.
    """

    def __init__(self, include: Optional[Dict[str, List[str]]] = None) -> None:
        self.include: Dict[str, set[str]] = {}
        if include is None:
            return
        if not isinstance(include, Mapping):
            raise ValueError(
                "Taxonomy filter must be a mapping of rank -> list of taxa, got "
                f"{type(include).__name__}. Example:\n  species:\n    - s__Human "
                "mastadenovirus A"
            )
        if not include:
            return
        for rank, values in include.items():
            rank_key = self._canonical_rank(rank)
            if rank_key not in TAX_RANKS:
                raise ValueError(
                    f"Unknown taxonomy rank '{rank}'. "
                    f"Use one of: {', '.join(TAX_RANKS)} (or 'class' for tclass)."
                )
            if values is None:
                values = []
            elif isinstance(values, str):
                values = [values]
            prefix = RANK_PREFIXES[rank_key]
            # Merge rather than assign: 'class' and 'tclass' canonicalize to the
            # same rank key, and assigning would silently drop one of them.
            self.include.setdefault(rank_key, set()).update(
                self._normalize(v, prefix, rank_key) for v in values
            )
        if not any(self.include.values()):
            # An include list that is present but lists no taxa would match
            # nothing at all, silently emptying every sample. That is nearly
            # always a typo, so fail loudly instead.
            raise ValueError(
                "Taxonomy filter lists no taxa for any rank "
                f"({', '.join(sorted(self.include))}). Add at least one entry, "
                "or omit the filter entirely to keep all assemblies."
            )
        # Drop ranks that ended up empty so `matches` never consults an empty set.
        self.include = {r: v for r, v in self.include.items() if v}

    @staticmethod
    def _canonical_rank(rank: str) -> str:
        """Map common rank aliases to the internal rank names."""
        rank = str(rank).strip().lower()
        if rank == "class":
            return "tclass"
        return rank

    @staticmethod
    def _normalize(
        value: object, prefix: str, rank_key: Optional[str] = None
    ) -> str:
        """Add the rank prefix (if absent) and lowercase for case-insensitive matching.

        ``rank_key`` is passed when normalizing user-supplied filter entries (not
        when normalizing lineage values). It enables a check that the entry does
        not carry a *different* rank's prefix, which would otherwise be prefixed
        again into something that can never match (e.g. ``genus: ["s__Foo"]``
        becoming ``g__s__foo``).
        """
        value = str(value).strip()
        lowered = value.lower()
        if not lowered.startswith(prefix.lower()):
            if rank_key is not None:
                for other, other_prefix in RANK_PREFIXES.items():
                    if other != rank_key and lowered.startswith(other_prefix.lower()):
                        raise ValueError(
                            f"Taxonomy filter entry '{value}' is listed under rank "
                            f"'{rank_key}' but carries the '{other_prefix}' "
                            f"({other}) prefix. Move it under '{other}', or use "
                            f"the '{prefix}' prefix (or none at all)."
                        )
            value = prefix + value
        return value.lower()

    @classmethod
    def from_yaml(cls, path: str) -> "TaxaFilter":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if data is None:
            data = {}
        return cls(data)

    @classmethod
    def from_spec(cls, spec: Optional[str]) -> Optional["TaxaFilter"]:
        """Resolve a ``--taxa-filter`` value.

        ``None`` selects the built-in :data:`HIGH_CONCERN_TAXA`, ``"all"``
        (case-insensitive) disables filtering, and anything else is a YAML path.
        """
        if spec is None:
            return cls(HIGH_CONCERN_TAXA)
        if spec.strip().lower() == TAXA_FILTER_ALL:
            return None
        return cls.from_yaml(spec)

    def matches(self, lineage: Dict[str, str]) -> bool:
        """Return True if ``lineage`` matches the include list."""
        if not self.include:
            return True
        for rank, values in self.include.items():
            value = lineage.get(rank)
            if value is not None and self._normalize(value, RANK_PREFIXES[rank]) in values:
                return True
        return False
