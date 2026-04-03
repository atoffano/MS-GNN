"""SwissProt 2024_01 data download and preprocessing utilities.

This module prepares the data used by the MS-GNN SwissProt pipeline:
- downloads the 2024_01 SwissProt release archive
- exports a SwissProt FASTA file keyed by accession
- exports the accession -> entry name mapping used by the loaders
- parses GO annotations into EXP and CUR TSVs for BPO/CCO/MFO
- downloads AlphaFold structures and InterPro annotations
- builds a SwissProt-to-STRINGDB mapping and filtered STRINGDB edge file

The repository is wired for the 2024_01 SwissProt release only, so this
script keeps the release-specific filenames and paths centralized here.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import go3
import requests
from Bio import SeqIO
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.constants import (
    ALPHAFOLD_TAR_PATH,
    ALPHAFOLD_URL,
    ALPHAFOLD_PDB_DIR,
    GO_ANNOTATION_TEMPLATE,
    GO_OBO_URL,
    GO_EXP_ANNOTATION_TEMPLATE,
    GO_NAMESPACE_MAP,
    GO_ONTOLOGIES,
    GO_OBO_PATH,
    GO_ROOT_TERMS,
    INTERPRO_GZ_PATH,
    INTERPRO_URL,
    INTERPRO_TSV,
    LOG_FORMAT,
    PID_MAPPING,
    STRINGDB_LINKS_GZ_PATH,
    STRINGDB_LINKS_URL,
    STRINGDB_PATH,
    STRINGDB_SWISSPROT_MAPPING,
    SWISSPROT_DAT_GZ_PATH,
    SWISSPROT_DAT_PATH,
    SWISSPROT_FASTA,
    SWISSPROT_RELEASE,
    SWISSPROT_ROOT,
    SWISSPROT_TAR_PATH,
    SWISSPROT_TAR_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger(__name__)

_ONTOLOGY_LOADED = False
_RE_OLD = re.compile(r"DR\s+GO;\s+(GO:\d+);.*;\s+([A-Z]+)\.")
_RE_NEW = re.compile(r"DR\s+GO;\s*(GO:\d+);.*;\s*([A-Z]+)(?::|\.)")
_RE_STRING = re.compile(r"DR\s+STRING;\s*([^;]+);")
_ROOT_TERMS = set(GO_ROOT_TERMS)
_EXP_CODES = {
    "EXP",
    "IDA",
    "IPI",
    "IMP",
    "IGI",
    "IEP",
    "HDA",
    "HMP",
    "HGI",
    "HEP",
}


def download_file(url: str, destination: Path, chunk_size: int = 65536) -> None:
    """Download a file to destination with a progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, destination)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as handle, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def _extract_tar_gz(archive_path: Path, out_dir: Path) -> None:
    """Extract a .tar.gz archive into out_dir."""
    logger.info("Extracting %s", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=out_dir)


def _decompress_gz(src_path: Path, dst_path: Path) -> None:
    """Decompress src_path (.gz) into dst_path."""
    logger.info("Decompressing %s -> %s", src_path, dst_path)
    with gzip.open(src_path, "rb") as f_in, open(dst_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _cleanup_compressed_files(version_dir: Path) -> None:
    """Remove downloaded compressed artifacts to save disk space."""
    for root, _, files in os.walk(version_dir):
        for fname in files:
            if fname.endswith(".gz") or fname.endswith(".tar.gz"):
                try:
                    os.remove(Path(root) / fname)
                except OSError:
                    pass


def download_go_obo() -> None:
    """Download the fixed GO ontology release if it is not already present."""
    GO_OBO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if GO_OBO_PATH.exists():
        logger.info("go.obo already present at %s", GO_OBO_PATH)
        return
    download_file(GO_OBO_URL, GO_OBO_PATH)


def load_ontology() -> None:
    """Load GO terms into go3 once per process."""
    global _ONTOLOGY_LOADED
    if _ONTOLOGY_LOADED:
        return
    logger.info("Loading ontology index with go3")
    go3.load_go_terms(str(GO_OBO_PATH))
    _ONTOLOGY_LOADED = True


def _get_term(term_id: str):
    try:
        return go3.get_term_by_id(term_id)
    except Exception:
        return None


def propagate(term_ids: Iterable[str]) -> Set[str]:
    """Return the True Path Rule closure for a list of GO terms."""
    result: Set[str] = set()
    stack: List[str] = []

    for raw in term_ids:
        term = _get_term(raw)
        if term and not term.is_obsolete:
            stack.append(term.id)

    while stack:
        term_id = stack.pop()
        if term_id in result:
            continue

        term = _get_term(term_id)
        if term is None or term.is_obsolete:
            continue

        result.add(term.id)
        for parent in term.parents:
            if parent not in result:
                stack.append(parent)

    result -= _ROOT_TERMS
    return result


def get_namespace(term_id: str) -> Optional[str]:
    """Return the GO namespace of a term, or None if unavailable."""
    term = _get_term(term_id)
    if term is None or term.is_obsolete:
        return None
    return term.namespace


def _split_terms_by_namespace(terms: Iterable[str]) -> Dict[str, List[str]]:
    """Group GO terms by ontology short name (BPO/CCO/MFO)."""
    grouped: DefaultDict[str, List[str]] = defaultdict(list)
    for term_id in terms:
        namespace = get_namespace(term_id)
        short = GO_NAMESPACE_MAP.get(namespace) if namespace else None
        if short:
            grouped[short].append(term_id)
    return grouped


def _expected_release_files() -> List[Path]:
    """Return all SwissProt release outputs expected for 2024_01."""
    files = [SWISSPROT_FASTA, PID_MAPPING]
    for onto in GO_ONTOLOGIES:
        files.append(
            GO_ANNOTATION_TEMPLATE.with_name(
                GO_ANNOTATION_TEMPLATE.name.format(onto=onto)
            )
        )
        files.append(
            GO_EXP_ANNOTATION_TEMPLATE.with_name(
                GO_EXP_ANNOTATION_TEMPLATE.name.format(onto=onto)
            )
        )
    return files


def _release_outputs_exist() -> bool:
    return all(path.exists() for path in _expected_release_files())


def download_swissprot_release(skip_download: bool = False) -> Optional[Path]:
    """Download and extract the 2024_01 SwissProt release archive."""
    SWISSPROT_ROOT.mkdir(parents=True, exist_ok=True)
    if SWISSPROT_DAT_PATH.exists():
        return SWISSPROT_DAT_PATH

    if skip_download:
        logger.warning(
            "SwissProt dat file missing and download skipped: %s", SWISSPROT_DAT_PATH
        )
        return None

    if not SWISSPROT_TAR_PATH.exists():
        download_file(SWISSPROT_TAR_URL, SWISSPROT_TAR_PATH)
    else:
        logger.info("Using existing SwissProt archive: %s", SWISSPROT_TAR_PATH)

    _extract_tar_gz(SWISSPROT_TAR_PATH, SWISSPROT_ROOT)

    if not SWISSPROT_DAT_GZ_PATH.exists():
        logger.error("Expected %s inside SwissProt archive", SWISSPROT_DAT_GZ_PATH)
        return None

    if not SWISSPROT_DAT_PATH.exists():
        _decompress_gz(SWISSPROT_DAT_GZ_PATH, SWISSPROT_DAT_PATH)

    return SWISSPROT_DAT_PATH


def iter_swissprot_entries(
    dat_path: Path,
) -> Iterator[Tuple[str, str, List[str], List[str], str]]:
    """Yield parsed SwissProt entries from the .dat file."""
    use_old_re = SWISSPROT_RELEASE in {"1.0", "4.0", "7.0"}
    regex = _RE_OLD if use_old_re else _RE_NEW

    entry_name: Optional[str] = None
    accession: Optional[str] = None
    exp_raw: List[str] = []
    cur_raw: List[str] = []
    sequence_lines: List[str] = []
    in_seq = False

    def flush_entry() -> Optional[Tuple[str, str, List[str], List[str], str]]:
        nonlocal entry_name, accession, exp_raw, cur_raw, sequence_lines, in_seq
        if entry_name is None or accession is None:
            entry_name = None
            accession = None
            exp_raw = []
            cur_raw = []
            sequence_lines = []
            in_seq = False
            return None

        sequence = "".join(sequence_lines)
        item = (
            entry_name,
            accession,
            sorted(propagate(exp_raw)),
            sorted(propagate(cur_raw)),
            sequence,
        )

        entry_name = None
        accession = None
        exp_raw = []
        cur_raw = []
        sequence_lines = []
        in_seq = False
        return item

    with open(dat_path, "r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if line == "//":
                item = flush_entry()
                if item is not None:
                    yield item
                continue

            if in_seq:
                sequence_lines.append("".join(line.strip().split()))
                continue

            if line.startswith("ID "):
                entry_name = line.split()[1]
            elif line.startswith("AC ") and accession is None:
                accession = line.split()[1].strip(";")
            elif line.startswith("DR   GO;"):
                match = regex.match(line)
                if not match:
                    bare = re.match(r"DR\s+GO;\s*(GO:\d+);", line)
                    if bare:
                        go_id = bare.group(1)
                        if go_id not in cur_raw:
                            cur_raw.append(go_id)
                    continue

                go_id = match.group(1)
                evidence_code = match.group(2)

                if evidence_code in _EXP_CODES and go_id not in exp_raw:
                    exp_raw.append(go_id)
                if go_id not in cur_raw:
                    cur_raw.append(go_id)
            elif line.startswith("SQ "):
                in_seq = True

    item = flush_entry()
    if item is not None:
        yield item


def _write_fasta_record(handle, accession: str, entry_name: str, sequence: str) -> None:
    handle.write(f">{accession} {entry_name}\n")
    for start in range(0, len(sequence), 60):
        handle.write(sequence[start : start + 60] + "\n")


def build_release_assets(dat_path: Path) -> None:
    """Parse SwissProt release data and write FASTA, PID mapping and GO TSVs."""
    logger.info("Building SwissProt release assets from %s", dat_path)

    SWISSPROT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    PID_MAPPING.parent.mkdir(parents=True, exist_ok=True)

    go_paths = {
        "cur": {
            onto: GO_ANNOTATION_TEMPLATE.with_name(
                GO_ANNOTATION_TEMPLATE.name.format(onto=onto)
            )
            for onto in GO_ONTOLOGIES
        },
        "exp": {
            onto: GO_EXP_ANNOTATION_TEMPLATE.with_name(
                GO_EXP_ANNOTATION_TEMPLATE.name.format(onto=onto)
            )
            for onto in GO_ONTOLOGIES
        },
    }

    for kind_paths in go_paths.values():
        for path in kind_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        fasta_handle = stack.enter_context(open(SWISSPROT_FASTA, "w"))
        mapping_handle = stack.enter_context(open(PID_MAPPING, "w"))
        mapping_handle.write("EntryID\tEntry Name\n")

        annotation_handles: Dict[Tuple[str, str], object] = {}
        for kind, kind_paths in go_paths.items():
            for onto, path in kind_paths.items():
                handle = stack.enter_context(open(path, "w"))
                handle.write("EntryID\tterm\n")
                annotation_handles[(kind, onto)] = handle

        total_entries = 0
        kept_entries = 0
        for entry_name, accession, exp_terms, cur_terms, sequence in tqdm(
            iter_swissprot_entries(dat_path), desc="Parsing SwissProt entries"
        ):
            total_entries += 1
            if not accession or not entry_name or not sequence:
                continue

            kept_entries += 1
            _write_fasta_record(fasta_handle, accession, entry_name, sequence)
            mapping_handle.write(f"{accession}\t{entry_name}\n")

            term_sets = {
                "exp": _split_terms_by_namespace(exp_terms),
                "cur": _split_terms_by_namespace(cur_terms),
            }

            for kind, by_ontology in term_sets.items():
                for ontology in GO_ONTOLOGIES:
                    terms = by_ontology.get(ontology, [])
                    if not terms:
                        continue
                    annotation_handles[(kind, ontology)].write(
                        f"{accession}\t{'; '.join(terms)}\n"
                    )

    logger.info(
        "Parsed %d entries, kept %d SwissProt proteins", total_entries, kept_entries
    )


def _parse_string_mapping_from_dat(
    dat_path: Path, protein_ids: Set[str]
) -> Dict[str, str]:
    """Build SwissProt accession -> STRING protein id mapping from raw .dat entries."""
    mapping: Dict[str, str] = {}
    primary_accession: Optional[str] = None
    current_string_ids: List[str] = []

    def flush_entry() -> None:
        nonlocal primary_accession, current_string_ids
        if (
            primary_accession
            and primary_accession in protein_ids
            and current_string_ids
        ):
            mapping.setdefault(primary_accession, current_string_ids[0])
        primary_accession = None
        current_string_ids = []

    with open(dat_path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("//"):
                flush_entry()
                continue

            if line.startswith("AC ") and primary_accession is None:
                fields = line.split()
                if len(fields) > 1:
                    primary_accession = fields[1].strip(";")
                continue

            if line.startswith("DR   STRING;"):
                match = _RE_STRING.match(line)
                if match:
                    current_string_ids.append(match.group(1).strip())

    flush_entry()

    STRINGDB_SWISSPROT_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    with open(STRINGDB_SWISSPROT_MAPPING, "w") as handle:
        handle.write("From\tTo\n")
        for accession, string_id in sorted(mapping.items()):
            handle.write(f"{accession}\t{string_id}\n")

    logger.info("Wrote STRINGDB mapping for %d SwissProt proteins", len(mapping))
    return mapping


def get_swissprot_protein_ids(fasta_path: Path) -> Set[str]:
    """Extract all protein IDs from the SwissProt FASTA file."""
    logger.info("Reading protein IDs from %s", fasta_path)
    protein_ids: Set[str] = set()

    with open(fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            protein_ids.add(record.id)

    logger.info("Found %d protein IDs in FASTA file", len(protein_ids))
    return protein_ids


def extract_alphafold_structures(
    tar_path: Path, output_dir: Path, protein_ids: Set[str]
) -> None:
    """Extract and decompress AlphaFold PDB files for SwissProt proteins."""
    logger.info("Extracting AlphaFold structures from %s", tar_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0

    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting PDB files"):
            if not member.name.endswith(".pdb.gz"):
                continue

            filename = Path(member.name).name
            parts = filename.split("-")
            if len(parts) < 2:
                skipped_count += 1
                continue

            protein_id = parts[1]
            if protein_id not in protein_ids:
                skipped_count += 1
                continue

            gz_path = output_dir / filename
            tar.extract(member, path=output_dir)

            extracted_path = output_dir / member.name
            if extracted_path != gz_path:
                extracted_path.rename(gz_path)
                try:
                    extracted_path.parent.rmdir()
                except OSError:
                    pass

            pdb_path = gz_path.with_suffix("")
            with gzip.open(gz_path, "rb") as f_in, open(pdb_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            extracted_count += 1

    logger.info("Extracted %d PDB files, skipped %d", extracted_count, skipped_count)


def process_interpro_annotations(
    interpro_gz_path: Path, protein_ids: Set[str], output_path: Path
) -> None:
    """Process InterPro annotations, keeping only SwissProt proteins."""
    logger.info("Processing InterPro annotations from %s", interpro_gz_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched_lines = 0
    total_lines = 0

    with gzip.open(interpro_gz_path, "rt") as f_in, open(output_path, "w") as f_out:
        f_out.write("ID\tIPR\tdesc\tdb\tstart\tend\n")
        for line in tqdm(f_in, desc="Processing InterPro annotations"):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                continue

            protein_id = parts[0]
            if protein_id in protein_ids:
                f_out.write(line + "\n")
                matched_lines += 1

    logger.info(
        "Processed %d lines, kept %d matching SwissProt proteins",
        total_lines,
        matched_lines,
    )


def get_stringdb(protein_ids: Set[str], dat_path: Path) -> None:
    """Download and filter STRING interactions for the SwissProt protein set."""
    STRINGDB_LINKS_GZ_PATH.parent.mkdir(parents=True, exist_ok=True)

    mapping = _parse_string_mapping_from_dat(dat_path, protein_ids)
    stringdb_ids = set(mapping.values())

    if not STRINGDB_LINKS_GZ_PATH.exists():
        download_file(STRINGDB_LINKS_URL, STRINGDB_LINKS_GZ_PATH)

    with gzip.open(STRINGDB_LINKS_GZ_PATH, "rt") as fin, open(
        STRINGDB_PATH, "w"
    ) as fout:
        header = fin.readline()
        fout.write(header)
        kept = 0
        for line in tqdm(fin, desc="Filtering STRINGDB edges"):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            if cols[0] in stringdb_ids or cols[1] in stringdb_ids:
                fout.write(line)
                kept += 1

    logger.info(
        "Filtered STRINGDB interactions written to %s (%d edges)", STRINGDB_PATH, kept
    )


def _release_assets_missing() -> bool:
    return not _release_outputs_exist()


def main(skip_download: bool = False) -> None:
    """Download, parse and export the SwissProt 2024_01 dataset."""
    download_go_obo()
    load_ontology()

    dat_path = download_swissprot_release(skip_download=skip_download)
    if dat_path is None:
        if _release_assets_missing():
            logger.warning(
                "SwissProt release assets are incomplete and the dat file is unavailable."
            )
        return

    if not _release_outputs_exist():
        build_release_assets(dat_path)
    else:
        logger.info("SwissProt release assets already present under %s", SWISSPROT_ROOT)

    protein_ids = get_swissprot_protein_ids(SWISSPROT_FASTA)

    # AlphaFold structures
    alphafold_output_dir = ALPHAFOLD_PDB_DIR
    if not skip_download and not ALPHAFOLD_TAR_PATH.exists():
        download_file(ALPHAFOLD_URL, ALPHAFOLD_TAR_PATH)
    if ALPHAFOLD_TAR_PATH.exists() and (
        not alphafold_output_dir.exists() or not any(alphafold_output_dir.glob("*.pdb"))
    ):
        extract_alphafold_structures(
            ALPHAFOLD_TAR_PATH, alphafold_output_dir, protein_ids
        )

    # InterPro annotations
    if not skip_download and not INTERPRO_GZ_PATH.exists():
        download_file(INTERPRO_URL, INTERPRO_GZ_PATH)
    if INTERPRO_GZ_PATH.exists() and not INTERPRO_TSV.exists():
        process_interpro_annotations(INTERPRO_GZ_PATH, protein_ids, INTERPRO_TSV)

    # STRINGDB mapping and edge file
    if not STRINGDB_PATH.exists() or not STRINGDB_SWISSPROT_MAPPING.exists():
        get_stringdb(protein_ids, dat_path)

    _cleanup_compressed_files(SWISSPROT_ROOT)

    logger.info("All processing complete!")
    logger.info("FASTA: %s", SWISSPROT_FASTA)
    logger.info("PID mapping: %s", PID_MAPPING)
    logger.info("InterPro annotations: %s", INTERPRO_TSV)
    logger.info("STRINGDB edges: %s", STRINGDB_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess SwissProt 2024_01 data."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download steps and reuse already downloaded archives.",
    )
    args = parser.parse_args()
    main(skip_download=args.skip_download)
