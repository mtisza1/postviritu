from postviritu.reassign import apply_identity_thresholds
from postviritu.taxonomy import map_ranks_to_esviritu, unclassified_lineage


def test_map_full_virus_lineage():
    rmap = {
        "superkingdom": "Viruses",
        "phylum": "Preplasmiviricota",
        "class": "Tectiliviricetes",
        "order": "Rowavirales",
        "family": "Adenoviridae",
        "genus": "Mastadenovirus",
        "species": "Human mastadenovirus F",
    }
    out = map_ranks_to_esviritu(rmap)
    assert out["kingdom"] == "k__Viruses"
    assert out["phylum"] == "p__Preplasmiviricota"
    assert out["tclass"] == "c__Tectiliviricetes"
    assert out["family"] == "f__Adenoviridae"
    assert out["species"] == "s__Human mastadenovirus F"
    # No strain rank -> subspecies mirrors species.
    assert out["subspecies"] == "t__Human mastadenovirus F"


def test_map_missing_mid_ranks_uses_unclassified_kingdom():
    rmap = {
        "superkingdom": "Viruses",
        "family": "Anelloviridae",
        "genus": "Gyrovirus",
        "species": "Gyrovirus chickenanemia",
    }
    out = map_ranks_to_esviritu(rmap)
    assert out["phylum"] == "p__unclassified_Viruses"
    assert out["tclass"] == "c__unclassified_Viruses"
    assert out["order"] == "o__unclassified_Viruses"
    assert out["family"] == "f__Anelloviridae"


def test_map_with_explicit_strain():
    rmap = {
        "superkingdom": "Viruses",
        "genus": "Mastadenovirus",
        "species": "Human mastadenovirus A",
        "serotype": "Human adenovirus 31",
    }
    out = map_ranks_to_esviritu(rmap)
    assert out["subspecies"] == "t__Human adenovirus 31"


def test_unclassified_lineage():
    out = unclassified_lineage()
    assert out["kingdom"] == "k__unclassified_Viruses"
    assert out["subspecies"] == "t__unclassified_Viruses"


def test_thresholds_demote_species_and_subspecies():
    lineage = {
        "kingdom": "k__Viruses",
        "phylum": "p__Duplornaviricota",
        "tclass": "c__Resentoviricetes",
        "order": "o__Reovirales",
        "family": "f__Sedoreoviridae",
        "genus": "g__Rotavirus",
        "species": "s__Rotavirus alphagastroenteritidis",
        "subspecies": "t__C2",
    }
    # identity below species cutoff (0.9) demotes both.
    out = apply_identity_thresholds(lineage, 0.8922, 0.90, 0.95)
    assert out["species"] == "s__unclassified_Rotavirus"
    assert out["subspecies"] == "t__unclassified_Rotavirus"


def test_thresholds_demote_only_subspecies():
    lineage = {
        "genus": "g__Mastadenovirus",
        "species": "s__Human mastadenovirus A",
        "subspecies": "t__Human adenovirus 31",
    }
    out = apply_identity_thresholds(lineage, 0.93, 0.90, 0.95)
    assert out["species"] == "s__Human mastadenovirus A"
    assert out["subspecies"] == "t__unclassified_Human mastadenovirus A"


def test_thresholds_no_change_when_high_identity():
    lineage = {
        "genus": "g__Mastadenovirus",
        "species": "s__Human mastadenovirus A",
        "subspecies": "t__Human adenovirus 31",
    }
    out = apply_identity_thresholds(lineage, 0.999, 0.90, 0.95)
    assert out == lineage
