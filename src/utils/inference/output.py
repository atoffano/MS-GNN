"""Prediction formatting and saving for inference.

Handles two output modes:
- Standard: TSV with columns (target_ID, term_ID, score)
- Barebones: TSV with columns (target_ID, term_ID) — leaf terms only

Also provides a human-readable console summary with GO term names.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _filter_leaf_terms(predicted_terms: List[str]) -> List[str]:
    """Return only leaf GO terms from predicted_terms.

    A term is a "leaf" within the predicted set when none of its
    direct GO children are also present in the predicted set.

    Args:
        predicted_terms: GO term IDs predicted for one protein.

    Returns:
        Subset containing only leaf terms.
    """
    import go3

    term_set = set(predicted_terms)
    leaf_terms = []
    for term_id in term_set:
        try:
            term = go3.get_term_by_id(term_id)
            if not any(child in term_set for child in term.children):
                leaf_terms.append(term_id)
        except Exception:
            leaf_terms.append(term_id)
    return leaf_terms


def _ensure_go_loaded() -> None:
    """Ensure go3 terms are loaded into memory."""
    try:
        import go3
        obo_path = Path("./data/go.obo")
        if obo_path.exists():
            go3.load_go_terms(str(obo_path))
        else:
            go3.load_go_terms()
    except Exception as exc:
        logger.warning("Could not load go.obo ontology: %s", exc)


def _get_term_name(term_id: str) -> str:
    """Return the name of a GO term, or the term ID if not found."""
    try:
        import go3
        t = go3.get_term_by_id(term_id)
        if hasattr(t, "name") and t.name:
            return t.name
    except Exception:
        pass
    return term_id


def save_inference_predictions(
    predictions: Dict[str, List[Tuple[str, float]]],
    subontology: str,
    output_dir: Path,
    tau: Optional[float] = None,
    barebones: bool = False,
    save_pkl: bool = False,
) -> Path:
    """Save inference predictions to a TSV file.

    Args:
        predictions: Dict mapping protein_id → [(go_term, score), ...].
        subontology: GO subontology name (MFO, BPO, CCO).
        output_dir: Directory for output files.
        tau: Score threshold for filtering. None = no filtering.
        barebones: If True (requires tau), write condensed leaf-terms-only output.
        save_pkl: If True, also save a pickled dictionary of predictions (.pkl).

    Returns:
        Path to the saved predictions file.
    """
    if barebones and tau is None:
        raise ValueError("Barebones output requires a tau threshold.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"predictions_{subontology}.tsv"

    _ensure_go_loaded()

    if barebones:
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write("target_ID\tterm_ID\n")
            for pid, term_scores in predictions.items():
                above_tau = [term for term, score in term_scores if score > tau]
                if above_tau:
                    leaves = _filter_leaf_terms(above_tau)
                    if leaves:
                        f.write(f"{pid}\t{'; '.join(leaves)}\n")
    else:
        name_cache: Dict[str, str] = {}
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write("target_ID\tterm_ID\tterm_name\tscore\n")
            for pid, term_scores in predictions.items():
                for term, score in term_scores:
                    if tau is not None and score <= tau:
                        continue
                    if score > 0:
                        if term not in name_cache:
                            name_cache[term] = _get_term_name(term)
                        f.write(f"{pid}\t{term}\t{name_cache[term]}\t{score:.6f}\n")

    logger.info("Predictions saved to %s", pred_path)
    if save_pkl:
        try:
            import pickle
            pkl_path = output_dir / f"predictions_{subontology}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(predictions, f)
            logger.info("Pickled predictions saved to %s", pkl_path)
        except Exception as exc:
            logger.warning("Could not save pickle: %s", exc)

    return pred_path


def print_summary(
    predictions: Dict[str, List[Tuple[str, float]]],
    subontology: str,
    output_path: Path,
    protein_names: Optional[Dict[str, str]] = None,
    tau: Optional[float] = None,
    top_k: int = 5,
) -> None:
    """Print a human-readable summary of predictions to stdout.

    Args:
        predictions: Dict mapping protein_id → [(go_term, score), ...].
        subontology: GO subontology name.
        output_path: Path to saved predictions file.
        protein_names: Optional mapping accession → entry name.
        tau: Model decision threshold (if None, looked up from DEFAULT_MODEL_THRESHOLDS).
        top_k: Number of top predictions to show per protein.
    """
    _ensure_go_loaded()

    # Determine effective threshold
    from src.utils.constants import DEFAULT_MODEL_THRESHOLDS
    default_thresh = DEFAULT_MODEL_THRESHOLDS.get(subontology)
    active_thresh = tau if tau is not None else default_thresh
    thresh_label = f"{active_thresh:.2f}" if active_thresh is not None else "N/A"
    if tau is None and default_thresh is not None:
        thresh_label += " (benchmark optimal)"

    sep = "═" * 68
    lines: List[str] = []
    lines.append(f"\n{sep}")
    lines.append(f"  MS-GNN Inference Results — {subontology} (Threshold: {thresh_label})")
    lines.append(sep)

    for pid, term_scores in predictions.items():
        name = protein_names.get(pid, "") if protein_names else ""
        header = f"  {pid}"
        if name:
            header += f" ({name})"
        header += ":"
        lines.append(header)

        # Sort by score descending
        sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)

        lines.append(f"    Top {min(top_k, len(sorted_terms))} predictions:")
        for term, score in sorted_terms[:top_k]:
            term_name = _get_term_name(term)
            if term_name and term_name != term:
                term_name_disp = term_name[:40]
                lines.append(f"      {term}  {term_name_disp:<42s} {score:.3f}")
            else:
                lines.append(f"      {term}  {score:.3f}")

        # Show leaf terms above threshold
        if active_thresh is not None:
            terms_above = [t for t, s in sorted_terms if s >= active_thresh]
            score_map = {t: s for t, s in sorted_terms}
            leaf_terms = _filter_leaf_terms(terms_above)
            # Sort leaf terms by score descending
            leaf_terms.sort(key=lambda t: score_map.get(t, 0.0), reverse=True)

            lines.append(f"    Leaf terms above threshold (τ >= {active_thresh:.2f}): {len(leaf_terms)}")
            if leaf_terms:
                for lt in leaf_terms:
                    lt_name = _get_term_name(lt)
                    lt_score = score_map.get(lt, 0.0)
                    if lt_name and lt_name != lt:
                        lt_disp = lt_name[:40]
                        lines.append(f"      ★ {lt}  {lt_disp:<42s} {lt_score:.3f}")
                    else:
                        lines.append(f"      ★ {lt}  {lt_score:.3f}")
            else:
                lines.append("      (None)")

        lines.append("")

    lines.append(f"  Predictions saved to: {output_path}")
    lines.append(f"{sep}\n")

    summary_text = "\n".join(lines)
    print(summary_text)
    logger.info("Summary of predictions for %s:\n%s", subontology, summary_text)

