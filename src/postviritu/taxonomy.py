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

from typing import Dict, List, Optional, Sequence

import yaml

from .io_esviritu import RANK_PREFIXES, TAX_RANKS

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
    "subspecies": ["subspecies", "strain", "serotype", "no rank"],
}

_DEFAULT_ROOT = "Viruses"


def map_ranks_to_esviritu(
    rank_to_name: Dict[str, str], root_default: str = _DEFAULT_ROOT
) -> Dict[str, str]:
    """Map a {ncbi_rank: name} dict to the 8 EsViritu ranks (with prefixes).

    Missing ranks are filled as ``<prefix>unclassified_<nearest ancestor>``.
    The ``subspecies`` slot only uses an explicit sub-species rank; it is never
    auto-filled here (EsViritu derives ``t__`` from species via thresholding).
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

    def __init__(self, data_dir: Optional[str] = None, threads: Optional[int] = None):
        self.data_dir = data_dir
        self.threads = threads
        self._lineage_cache: Dict[str, Dict[str, str]] = {}

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
                )
            # Any taxid that produced no output gets an empty map.
            for t in missing:
                self._lineage_cache.setdefault(t, {})
        return {t: self._lineage_cache.get(t, {}) for t in unique}

    @staticmethod
    def _row_to_rank_map(names, ranks) -> Dict[str, str]:
        if _is_missing(names) or _is_missing(ranks):
            return {}
        rank_map: Dict[str, str] = {}
        for name, rank in zip(str(names).split(";"), str(ranks).split(";")):
            if name and rank:
                rank_map[rank] = name
        return rank_map

    def esviritu_lineage(self, taxid: str) -> Dict[str, str]:
        """Return the 8-rank EsViritu lineage for a single taxid."""
        rmap = self.rank_maps([taxid]).get(str(taxid), {})
        if not rmap:
            return unclassified_lineage()
        return map_ranks_to_esviritu(rmap)

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
                continue
            if isinstance(values, str):
                values = [values]
            prefix = RANK_PREFIXES[rank_key]
            self.include[rank_key] = {
                self._normalize(v, prefix) for v in values
            }

    @staticmethod
    def _canonical_rank(rank: str) -> str:
        """Map common rank aliases to the internal rank names."""
        rank = str(rank).strip().lower()
        if rank == "class":
            return "tclass"
        return rank

    @staticmethod
    def _normalize(value: object, prefix: str) -> str:
        """Add the rank prefix (if absent) and lowercase for case-insensitive matching."""
        value = str(value).strip()
        if not value.lower().startswith(prefix.lower()):
            value = prefix + value
        return value.lower()

    @classmethod
    def from_yaml(cls, path: str) -> "TaxaFilter":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if data is None:
            data = {}
        return cls(data)

    def matches(self, lineage: Dict[str, str]) -> bool:
        """Return True if ``lineage`` matches the include list."""
        if not self.include:
            return True
        for rank, values in self.include.items():
            value = lineage.get(rank)
            if value is not None and self._normalize(value, RANK_PREFIXES[rank]) in values:
                return True
        return False
