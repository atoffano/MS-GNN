import requests
from src.utils.constants import (
    UNIPROT_JSON_URL,
    PDB_DOWNLOAD_URL,
    ALPHAFOLD_STRUCTURE_URL,
)


# Structure download utilities
def download_pdb(uniprot_id: str, dest_path: str) -> bool:
    """Try downloading PDB structure from RCSB."""
    try:
        response = requests.get(
            UNIPROT_JSON_URL.format(uniprot_id=uniprot_id), timeout=2
        )
        response.raise_for_status()
        data = response.json()

        for ref in data.get("uniProtKBCrossReferences", []):
            if ref.get("database") != "PDB":
                continue
            pdb_id = ref.get("id")
            if not pdb_id:
                continue

            pdb_resp = requests.get(PDB_DOWNLOAD_URL.format(pdb_id=pdb_id), timeout=2)
            pdb_resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(pdb_resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_alphafold(uniprot_id: str, dest_path: str) -> bool:
    """Try downloading structure from AlphaFold."""
    try:
        response = requests.get(
            ALPHAFOLD_STRUCTURE_URL.format(uniprot_id=uniprot_id), timeout=15
        )
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        return True
    except requests.RequestException:
        return False
