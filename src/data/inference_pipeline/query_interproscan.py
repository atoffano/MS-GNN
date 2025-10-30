"""InterProScan REST API client for protein domain annotation.

This module provides a client interface to the InterProScan 5 REST API for
querying protein domain and family annotations. InterProScan integrates
multiple protein signature databases (Pfam, SMART, ProSite, etc.) and
returns unified IPR accessions for detected domains.

The client handles job submission, polling, and result parsing for
on-demand protein annotation during inference.
"""

import requests
import time


def submit_interproscan(sequence, email=None, timeout=600, poll_interval=20):
    """
    Submit a protein sequence to the InterProScan 5 REST API and return IPR accessions.

    Args:
        sequence (str): Protein sequence (raw amino acid sequence, without FASTA header).
        email (str, optional): Email to provide to InterProScan (recommended).
        timeout (int, optional): Max wait time in seconds for job completion.
        poll_interval (int, optional): Time in seconds between status polls.

    Returns:
        list: List of unique IPR accessions found in the annotations.

    Raises:
        Exception: If job fails or timeout is reached.
    """
    base_url = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

    # Submit job
    submit_url = f"{base_url}/run"
    params = {
        "stype": "p",  # protein sequence
        "goterms": "false",
        "pathways": "false",
        "sequence": sequence,
    }
    if email:
        params["email"] = email

    response = requests.post(submit_url, data=params)

    if response.status_code != 200:
        raise Exception(f"Failed to submit sequence: {response.text}")

    job_id = response.text.strip()
    if not job_id:
        raise Exception("No job ID returned from InterProScan API.")

    print(f"Job submitted with ID: {job_id}")

    # Poll for job status
    status_url = f"{base_url}/status/{job_id}"
    elapsed = 0
    while elapsed < timeout:
        status_resp = requests.get(status_url)
        if status_resp.status_code != 200:
            raise Exception(f"Failed to get job status: {status_resp.text}")

        status = status_resp.text.strip()
        print(f"Job status: {status} (elapsed: {elapsed}s)")

        if status == "FINISHED":
            break
        elif status == "FAILED" or status == "ERROR":
            raise Exception("InterProScan job failed.")

        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        raise Exception("InterProScan job timed out.")

    # Fetch results in JSON format
    result_url = f"{base_url}/result/{job_id}/json"
    result_resp = requests.get(result_url)
    if result_resp.status_code != 200:
        raise Exception(f"Failed to retrieve results: {result_resp.text}")

    result_data = result_resp.json()

    # Extract IPR accessions
    ipr_accessions = set()
    for result in result_data.get("results", []):
        for match in result.get("matches", []):
            entry = match.get("signature", {}).get("entry")
            if entry and "accession" in entry:
                accession = entry["accession"]
                # Only keep IPR accessions (some entries might be None)
                if accession and accession.startswith("IPR"):
                    ipr_accessions.add(accession)

    return sorted(list(ipr_accessions))


# Example usage:
if __name__ == "__main__":
    sequence = "MSLEQKKGADIISKILQIQNSIGKTTSPSTLKTKLSEISRKEQENARIQSKLSDLQKKKIDIDNKLLKEKQNLIKEEILERKKLEVLTKKQQKDEIEHQKKLKREIDAIKASTQYITDVSISSYNNTIPETEPEYDLFISHASEDKEDFVRPLAETLQQLGVNVWYDEFTLKVGDSLRQKIDSGLRNSKYGTVVLSTDFIKKDWTNYELDGLVAREMNGHKMILPIWHKITKNDVLDYSPNLADKVALNTSVNSIEEIAHQLADVILNR"
    ipr_tags = submit_interproscan(sequence, email="antoine.toffano@lirmm.fr")
    print(f"\nFound IPR accessions: {ipr_tags}")
