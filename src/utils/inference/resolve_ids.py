"""UniProt ID resolution and sequence fetching for inference.

This module handles:
- Accepting Accession Numbers (P01308) or EntryIDs (INS_HUMAN)
- Resolving both formats using local SwissProt mapping or UniProt REST API
- Fetching amino acid sequences
- Returning structured ProteinInfo objects
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.utils.constants import (
    PID_MAPPING,
    SWISSPROT_FASTA,
    UNIPROT_JSON_URL,
)

logger = logging.getLogger(__name__)

# UniProt FASTA endpoint
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"


@dataclass
class ProteinInfo:
    """Resolved protein information for inference.

    Attributes:
        accession: UniProt accession number (e.g., P01308).
        entry_name: UniProt entry name / EntryID (e.g., INS_HUMAN).
        sequence: Full amino acid sequence.
        organism_id: NCBI taxonomy ID (for STRING-DB mapping). May be None.
        is_in_swissprot: Whether the protein exists in the local SwissProt 2024_01 release.
    """

    accession: str
    entry_name: str
    sequence: str
    organism_id: Optional[int] = None
    is_in_swissprot: bool = False


class IDResolver:
    """Resolves UniProt identifiers and fetches sequences.

    Uses a local SwissProt annotations TSV for fast O(1) lookups, falling
    back to the UniProt REST API for proteins not found locally.
    """

    def __init__(self, mapping_path: Path = PID_MAPPING, fasta_path: Path = SWISSPROT_FASTA):
        """Initialize the resolver.

        Args:
            mapping_path: Path to swissprot_2024_01_annotations.tsv with
                          columns [EntryID, Entry Name, term, Sequence].
            fasta_path: Path to the SwissProt FASTA (used for sequence lookup
                        when the annotations TSV does not contain sequences).
        """
        self._acc_to_entry: Dict[str, str] = {}
        self._entry_to_acc: Dict[str, str] = {}
        self._acc_to_seq: Dict[str, str] = {}

        if mapping_path.exists():
            logger.info("Loading local ID mapping from %s", mapping_path)
            df = pd.read_csv(mapping_path, sep="\t")
            # In swissprot_2024_01_annotations.tsv:
            # EntryID column contains Entry Names (e.g. AADK_BACSU)
            # Entry Name column contains Accessions (e.g. P17585)
            entry_series = df["EntryID"].astype(str)
            acc_series = df["Entry Name"].astype(str)
            self._acc_to_entry = dict(zip(acc_series, entry_series))
            self._entry_to_acc = dict(zip(entry_series, acc_series))
            if "Sequence" in df.columns:
                has_seq = df["Sequence"].notna()
                self._acc_to_seq = dict(
                    zip(acc_series[has_seq], df["Sequence"][has_seq].astype(str))
                )
            logger.info(
                "Loaded %d accession↔entry mappings", len(self._acc_to_entry)
            )
        else:
            logger.warning(
                "Local mapping file not found at %s; will rely on UniProt API",
                mapping_path,
            )

        # Try to index FASTA sequences if mapping didn't include sequences
        if not self._acc_to_seq and fasta_path.exists():
            self._load_fasta_index(fasta_path)

    def _load_fasta_index(self, fasta_path: Path) -> None:
        """Build an in-memory accession→sequence index from the SwissProt FASTA."""
        from Bio import SeqIO

        logger.info("Indexing FASTA sequences from %s (this may take a moment)...", fasta_path)
        count = 0
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            # FASTA headers: >sp|P01308|INS_HUMAN ... or just >P01308
            seq_id = record.id
            # Handle sp|ACC|ENTRY format
            parts = seq_id.split("|")
            if len(parts) >= 2:
                acc = parts[1]
            else:
                acc = seq_id
            self._acc_to_seq[acc] = str(record.seq)
            count += 1
        logger.info("Indexed %d sequences from FASTA", count)

    def _is_accession(self, identifier: str) -> bool:
        """Heuristic: accessions are alphanumeric without underscores."""
        return "_" not in identifier

    def resolve(
        self,
        identifier: str,
        require_sequence: bool = True,
        force_remote: bool = False,
    ) -> ProteinInfo:
        """Resolve a single UniProt identifier to a ProteinInfo.

        Args:
            identifier: Either an Accession Number (P01308) or
                        EntryID/Entry Name (INS_HUMAN).
            require_sequence: Whether amino acid sequence is strictly required.
                              If False, missing sequence defaults to "" without calling API.
            force_remote: If True, bypass local SwissProt TSV mapping and query
                          the UniProt REST API directly.

        Returns:
            Fully resolved ProteinInfo.

        Raises:
            ValueError: If the identifier cannot be resolved.
        """
        accession: Optional[str] = None
        entry_name: Optional[str] = None
        sequence: Optional[str] = None
        organism_id: Optional[int] = None
        is_local = False

        # ---- Local resolution ----
        if not force_remote:
            if self._is_accession(identifier):
                accession = identifier
                entry_name = self._acc_to_entry.get(accession)
                is_local = accession in self._acc_to_entry
            else:
                # Assume it's an EntryID / Entry Name
                entry_name = identifier
                accession = self._entry_to_acc.get(entry_name)
                is_local = entry_name in self._entry_to_acc

            # Try local sequence
            if accession and accession in self._acc_to_seq:
                sequence = self._acc_to_seq[accession]

        # ---- API fallback for missing info or forced remote ----
        if force_remote or accession is None or (sequence is None and require_sequence):
            logger.info(
                "Fetching %s from UniProt REST API (%s)",
                identifier,
                "forced remote lookup"
                if force_remote
                else ("found accession but missing sequence" if accession else "failed"),
            )
            api_info = self._fetch_from_api(identifier)
            if api_info is None:
                if accession is not None and not require_sequence:
                    sequence = ""
                else:
                    raise ValueError(
                        f"Could not resolve identifier '{identifier}' locally or via UniProt API"
                    )
            else:
                accession = accession or api_info["accession"]
                entry_name = entry_name or api_info.get("entry_name", identifier)
                sequence = sequence or api_info.get("sequence")
                organism_id = api_info.get("organism_id")
                is_local = accession in self._acc_to_entry

        if sequence is None:
            if require_sequence:
                raise ValueError(
                    f"Could not obtain sequence for '{identifier}' (accession={accession})"
                )
            sequence = ""

        # Fallback entry_name if still missing
        if entry_name is None:
            entry_name = accession

        return ProteinInfo(
            accession=accession,
            entry_name=entry_name,
            sequence=sequence,
            organism_id=organism_id,
            is_in_swissprot=is_local,
        )

    def _fetch_from_api(self, identifier: str) -> Optional[Dict]:
        """Fetch protein info from UniProt REST API.

        Args:
            identifier: UniProt accession or entry name.

        Returns:
            Dictionary with keys: accession, entry_name, sequence, organism_id.
            None if the API call fails.
        """
        # Try JSON endpoint for metadata
        try:
            json_resp = requests.get(
                UNIPROT_JSON_URL.format(uniprot_id=identifier), timeout=10
            )
            json_resp.raise_for_status()
            data = json_resp.json()

            accession = data.get("primaryAccession", identifier)
            entry_name = data.get("uniProtkbId", identifier)
            organism_id = (
                data.get("organism", {}).get("taxonId")
            )

            # Extract sequence
            seq_obj = data.get("sequence", {})
            sequence = seq_obj.get("value")

            return {
                "accession": accession,
                "entry_name": entry_name,
                "sequence": sequence,
                "organism_id": organism_id,
            }
        except requests.RequestException as exc:
            logger.warning("UniProt JSON API failed for %s: %s", identifier, exc)

        # Fallback: try FASTA endpoint for sequence only
        try:
            fasta_resp = requests.get(
                UNIPROT_FASTA_URL.format(uniprot_id=identifier), timeout=10
            )
            fasta_resp.raise_for_status()
            lines = fasta_resp.text.strip().split("\n")
            if lines:
                header = lines[0]
                sequence = "".join(lines[1:])
                # Parse header: >sp|P01308|INS_HUMAN ...
                parts = header.lstrip(">").split("|")
                acc = parts[1] if len(parts) >= 2 else identifier
                entry = parts[2].split()[0] if len(parts) >= 3 else identifier
                return {
                    "accession": acc,
                    "entry_name": entry,
                    "sequence": sequence,
                    "organism_id": None,
                }
        except requests.RequestException as exc:
            logger.warning("UniProt FASTA API failed for %s: %s", identifier, exc)

        return None


# Module-level singleton for convenience
_resolver: Optional[IDResolver] = None


def get_resolver(
    mapping_path: Path = PID_MAPPING,
    fasta_path: Path = SWISSPROT_FASTA,
) -> IDResolver:
    """Get or create the module-level IDResolver singleton.

    Args:
        mapping_path: Path to SwissProt annotations TSV.
        fasta_path: Path to SwissProt FASTA file.

    Returns:
        IDResolver instance.
    """
    global _resolver
    if _resolver is None:
        _resolver = IDResolver(mapping_path=mapping_path, fasta_path=fasta_path)
    return _resolver


def resolve_proteins(
    identifiers: List[str],
    mapping_path: Path = PID_MAPPING,
    fasta_path: Path = SWISSPROT_FASTA,
    require_sequence: bool = True,
    force_remote: bool = False,
) -> List[ProteinInfo]:
    """Resolve a list of UniProt identifiers to ProteinInfo objects.

    Args:
        identifiers: List of UniProt accessions or entry names.
        mapping_path: Path to SwissProt annotations TSV.
        fasta_path: Path to SwissProt FASTA file.
        require_sequence: Whether sequences are strictly required.
        force_remote: Whether to force UniProt REST API lookup.

    Returns:
        List of resolved ProteinInfo objects.

    Raises:
        ValueError: If any identifier cannot be resolved.
    """
    resolver = get_resolver(mapping_path=mapping_path, fasta_path=fasta_path)
    results = []
    for ident in identifiers:
        ident = ident.strip()
        if not ident:
            continue
        try:
            info = resolver.resolve(
                ident, require_sequence=require_sequence, force_remote=force_remote
            )
            logger.info(
                "Resolved %s → %s / %s (%d aa, in_swissprot=%s)",
                ident,
                info.accession,
                info.entry_name,
                len(info.sequence),
                info.is_in_swissprot,
            )
            results.append(info)
        except ValueError as exc:
            logger.error("Failed to resolve '%s': %s", ident, exc)
            raise
    return results
