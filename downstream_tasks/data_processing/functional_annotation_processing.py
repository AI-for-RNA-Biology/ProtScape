#!/usr/bin/env python3
"""Build frozen HPA localization and Reactome pathway annotation tables.

This module deliberately has no network client and no project configuration
dependency.  Every input must be an explicit local ``.tsv``/``.gmt`` file or
a ZIP archive containing exactly one such file (unless a member is selected
explicitly).

Both processors write the same canonical long table::

    protein,label,label_id,source

Rows are validated, de-duplicated, filtered on the number of unique positive
proteins per label, and sorted.  A deterministic JSON manifest records the
exact input SHA256, parameters, output SHA256, and row counts; it intentionally
contains no timestamp or machine-specific absolute path.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple, Union


CANONICAL_COLUMNS = ("protein", "label", "label_id", "source")
HPA_SOURCE = "HPA"
REACTOME_SOURCE = "Reactome"
HPA_LABEL_ID_POLICY = "source_scoped_location_label"
HPA_LABEL_ID_TEMPLATE = "HPA:<location label>"
REACTOME_UNIVERSE_MEMBERSHIP_POLICY = (
    "raw_trimmed_case_exact_before_gene_symbol_normalization"
)

# In the frozen HPA subcellular-location export, each of these columns contains
# semicolon-separated locations at the confidence level named by the column.
DEFAULT_HPA_EVIDENCE_COLUMNS = (
    "Enhanced",
    "Supported",
    "Approved",
    "Uncertain",
)
DEFAULT_HPA_ACCEPTED_RELIABILITY = ("Enhanced", "Supported", "Approved")

_REACTOME_HUMAN_STABLE_ID = re.compile(r"^R-HSA-[0-9]+(?:\.[0-9]+)?$")
_REMOTE_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_ENSEMBL_GENE_ID = re.compile(r"^ENSG[0-9]+(?:\.[0-9]+)?$", re.IGNORECASE)
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


PathLike = Union[str, Path]
Row = Dict[str, str]


def _as_local_file(path_value: PathLike, expected_suffix: str) -> Path:
    """Validate that *path_value* is an existing local file of an allowed type."""
    raw = str(path_value)
    if _REMOTE_PREFIX.match(raw):
        raise ValueError("Live or remote inputs are not supported: {}".format(raw))
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Input file not found: {}".format(path))
    suffix = path.suffix.lower()
    if suffix not in (expected_suffix, ".zip"):
        raise ValueError(
            "Expected a {} or .zip input, got: {}".format(expected_suffix, path)
        )
    return path


def sha256_file(path: PathLike) -> str:
    """Return the SHA256 digest of a local file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _select_zip_member(
    archive: zipfile.ZipFile,
    expected_suffix: str,
    requested_member: Optional[str],
) -> zipfile.ZipInfo:
    infos = [info for info in archive.infolist() if not info.is_dir()]
    if requested_member is not None:
        matches = [info for info in infos if info.filename == requested_member]
        if len(matches) != 1:
            raise ValueError(
                "ZIP member {!r} must occur exactly once; found {}".format(
                    requested_member, len(matches)
                )
            )
        selected = matches[0]
        if not selected.filename.lower().endswith(expected_suffix):
            raise ValueError(
                "ZIP member {!r} is not a {} file".format(
                    selected.filename, expected_suffix
                )
            )
    else:
        matches = [
            info
            for info in infos
            if info.filename.lower().endswith(expected_suffix)
            and not info.filename.startswith("__MACOSX/")
        ]
        if len(matches) != 1:
            raise ValueError(
                "ZIP input must contain exactly one {} file; found {}. "
                "Select one explicitly with archive_member/--archive-member.".format(
                    expected_suffix, len(matches)
                )
            )
        selected = matches[0]

    if selected.flag_bits & 0x1:
        raise ValueError("Encrypted ZIP members are not supported")
    if selected.file_size == 0:
        raise ValueError("Selected ZIP member is empty: {}".format(selected.filename))
    return selected


@contextlib.contextmanager
def _open_text_input(
    path_value: PathLike,
    expected_suffix: str,
    archive_member: Optional[str],
) -> Iterable[Tuple[TextIO, Optional[str]]]:
    """Open a validated UTF-8 text input and yield it with its ZIP member name."""
    path = _as_local_file(path_value, expected_suffix)
    if path.suffix.lower() != ".zip":
        if archive_member is not None:
            raise ValueError("archive_member is only valid for ZIP inputs")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle, None
        return

    if not zipfile.is_zipfile(str(path)):
        raise ValueError("Input has a .zip suffix but is not a valid ZIP: {}".format(path))
    with zipfile.ZipFile(str(path), "r") as archive:
        info = _select_zip_member(archive, expected_suffix, archive_member)
        with archive.open(info, "r") as raw_handle:
            with io.TextIOWrapper(
                raw_handle, encoding="utf-8-sig", errors="strict", newline=""
            ) as text_handle:
                yield text_handle, info.filename


def _validate_text(value: str, field: str, location: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Empty {} at {}".format(field, location))
    if _CONTROL_CHARACTER.search(cleaned):
        raise ValueError("Control character in {} at {}".format(field, location))
    return cleaned


def _validate_gene_symbol_token(value: str, location: str) -> str:
    """Validate one symbol token without changing its case."""
    symbol = _validate_text(value, "gene symbol", location)
    if any(character.isspace() for character in symbol):
        raise ValueError("Whitespace in gene symbol at {}: {!r}".format(location, value))
    if any(delimiter in symbol for delimiter in (";", ",", "|")):
        raise ValueError(
            "Multiple or malformed gene symbols at {}: {!r}".format(location, value)
        )
    return symbol


def _normalize_gene_symbol(value: str, location: str) -> str:
    return _validate_gene_symbol_token(value, location).upper()


def _load_protein_universe(
    path_value: PathLike,
) -> Tuple[set, Dict[str, object]]:
    """Load a strict one-column symbol list or two-column protein edge list.

    Schema detection is based on the first non-comment record and every later
    record must have the same width.  The returned provenance deliberately
    excludes the filename and absolute path so manifests can be reproduced
    after moving an otherwise identical frozen universe file.
    """
    raw_path = str(path_value)
    if _REMOTE_PREFIX.match(raw_path):
        raise ValueError(
            "Live or remote protein-universe inputs are not supported: {}".format(
                raw_path
            )
        )
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Protein-universe file not found: {}".format(path))

    proteins = set()
    expected_width = None
    record_count = 0
    member_count = 0
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if expected_width is None:
                expected_width = len(fields)
                if expected_width not in (1, 2):
                    raise ValueError(
                        "Protein universe must be a one-column symbol list or a "
                        "two-column edge list; found {} columns at line {}".format(
                            expected_width, line_number
                        )
                    )
            elif len(fields) != expected_width:
                raise ValueError(
                    "Inconsistent protein-universe schema at line {}: expected {} "
                    "columns, found {}".format(
                        line_number, expected_width, len(fields)
                    )
                )

            record_count += 1
            member_count += len(fields)
            for field_number, field in enumerate(fields, start=1):
                proteins.add(
                    _validate_gene_symbol_token(
                        field,
                        "{}:line {} field {}".format(
                            path, line_number, field_number
                        ),
                    )
                )

    if expected_width is None or not proteins:
        raise ValueError("Protein-universe file contains no symbols: {}".format(path))

    schema = "symbol_list" if expected_width == 1 else "two_column_edge_list"
    provenance = {
        "membership_policy": REACTOME_UNIVERSE_MEMBERSHIP_POLICY,
        "schema": schema,
        "sha256": sha256_file(path),
        "statistics": {
            "member_fields": member_count,
            "records": record_count,
            "unique_proteins": len(proteins),
        },
    }
    return proteins, provenance


def _validate_thresholds(
    min_positive_count: int, max_positive_count: Optional[int]
) -> None:
    if isinstance(min_positive_count, bool) or not isinstance(min_positive_count, int):
        raise TypeError("min_positive_count must be an integer")
    if min_positive_count < 1:
        raise ValueError("min_positive_count must be at least 1")
    if max_positive_count is not None:
        if isinstance(max_positive_count, bool) or not isinstance(max_positive_count, int):
            raise TypeError("max_positive_count must be an integer or None")
        if max_positive_count < min_positive_count:
            raise ValueError(
                "max_positive_count must be greater than or equal to min_positive_count"
            )


def _normalized_reliabilities(values: Sequence[str]) -> Tuple[str, ...]:
    normalized = []
    seen = set()
    for value in values:
        cleaned = str(value).strip().casefold()
        if not cleaned:
            raise ValueError("Accepted reliability values cannot be empty")
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    if not normalized:
        raise ValueError("At least one accepted reliability value is required")
    return tuple(sorted(normalized))


def _validate_column_names(columns: Sequence[str], parameter: str) -> Tuple[str, ...]:
    cleaned = tuple(str(column).strip() for column in columns)
    if not cleaned or any(not column for column in cleaned):
        raise ValueError("{} must contain non-empty column names".format(parameter))
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("{} contains duplicate column names".format(parameter))
    return cleaned


def _read_tsv_rows(handle: TextIO, input_name: str) -> Iterable[Tuple[int, Dict[str, str]]]:
    reader = csv.reader(handle, dialect="excel-tab", strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("HPA TSV is empty: {}".format(input_name))
    if not header or all(not field.strip() for field in header):
        raise ValueError("HPA TSV has an empty header: {}".format(input_name))
    header = [field.strip() for field in header]
    if any(not field for field in header):
        raise ValueError("HPA TSV contains an empty column name")
    if len(set(header)) != len(header):
        raise ValueError("HPA TSV contains duplicate column names")

    for line_number, values in enumerate(reader, start=2):
        if not values or all(value == "" for value in values):
            continue
        if len(values) != len(header):
            raise ValueError(
                "Expected {} TSV fields at line {}, found {}".format(
                    len(header), line_number, len(values)
                )
            )
        yield line_number, dict(zip(header, values))


def _hpa_evidence_reliability(
    evidence_columns: Sequence[str],
    evidence_reliability: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if evidence_reliability is None:
        return {column: column for column in evidence_columns}

    unknown = set(evidence_reliability) - set(evidence_columns)
    missing = set(evidence_columns) - set(evidence_reliability)
    if unknown or missing:
        raise ValueError(
            "evidence_reliability keys must exactly match evidence_columns; "
            "missing={}, unknown={}".format(sorted(missing), sorted(unknown))
        )
    result = {}
    for column in evidence_columns:
        reliability = str(evidence_reliability[column]).strip()
        if not reliability:
            raise ValueError("Empty reliability for evidence column {!r}".format(column))
        result[column] = reliability
    return result


def _parse_hpa(
    input_path: PathLike,
    gene_column: str,
    evidence_columns: Sequence[str],
    evidence_reliability: Optional[Mapping[str, str]],
    reliability_column: Optional[str],
    accepted_reliability: Sequence[str],
    include_uncertain: bool,
    archive_member: Optional[str],
) -> Tuple[List[Row], Optional[str], Dict[str, object], Dict[str, int]]:
    evidence_columns = _validate_column_names(evidence_columns, "evidence_columns")
    gene_column = str(gene_column).strip()
    if not gene_column:
        raise ValueError("gene_column cannot be empty")
    if reliability_column is not None:
        reliability_column = str(reliability_column).strip()
        if not reliability_column:
            raise ValueError("reliability_column cannot be empty")
        if evidence_reliability is not None:
            raise ValueError(
                "evidence_reliability cannot be combined with a row-level "
                "reliability_column"
            )

    accepted = set(_normalized_reliabilities(accepted_reliability))
    if include_uncertain:
        # The flag is the deliberate opt-in.  It is sufficient on its own so a
        # caller cannot accidentally request the flag while still inheriting a
        # default accepted-reliability list that silently omits the evidence.
        accepted.add("uncertain")
    else:
        accepted.discard("uncertain")
    if not accepted:
        raise ValueError(
            "No accepted reliability remains after excluding Uncertain; set "
            "include_uncertain=True to opt in"
        )

    per_column_reliability = None
    if reliability_column is None:
        per_column_reliability = _hpa_evidence_reliability(
            evidence_columns, evidence_reliability
        )

    annotations = []
    parsed_rows = 0
    accepted_rows = 0
    required = set(evidence_columns)
    required.add(gene_column)
    if reliability_column is not None:
        required.add(reliability_column)

    input_name = str(input_path)
    with _open_text_input(input_path, ".tsv", archive_member) as opened:
        handle, selected_member = opened
        iterator = _read_tsv_rows(handle, input_name)
        try:
            first_line, first_row = next(iterator)
        except StopIteration:
            raise ValueError("HPA TSV has a header but no data rows: {}".format(input_name))

        available = set(first_row)
        missing = required - available
        if missing:
            raise ValueError("HPA TSV is missing required columns: {}".format(sorted(missing)))

        def consume(line_number: int, row: Dict[str, str]) -> None:
            nonlocal parsed_rows, accepted_rows
            parsed_rows += 1
            location = "{}:line {}".format(input_name, line_number)
            protein = _normalize_gene_symbol(row[gene_column], location)

            columns_to_read = []
            if reliability_column is not None:
                reliability = _validate_text(
                    row[reliability_column], "reliability", location
                ).casefold()
                if reliability == "uncertain" and not include_uncertain:
                    return
                if reliability not in accepted:
                    return
                columns_to_read = list(evidence_columns)
            else:
                for column in evidence_columns:
                    reliability = per_column_reliability[column].strip().casefold()
                    if reliability == "uncertain" and not include_uncertain:
                        continue
                    if reliability in accepted:
                        columns_to_read.append(column)

            if columns_to_read:
                accepted_rows += 1
            for column in columns_to_read:
                raw_labels = row[column].strip()
                if not raw_labels:
                    continue
                for raw_label in raw_labels.split(";"):
                    label = _validate_text(raw_label, "HPA label", location)
                    if label.casefold() == "uncertain" and not include_uncertain:
                        continue
                    annotations.append(
                        {
                            "protein": protein,
                            "label": label,
                            # GO IDs are unsuitable as class identity here:
                            # GO:0005730 maps to both Nucleoli and Nucleoli rim,
                            # and two HPA locations have no GO ID.  Scope the
                            # complete location label to its source instead.
                            "label_id": "HPA:{}".format(label),
                            "source": HPA_SOURCE,
                        }
                    )

        consume(first_line, first_row)
        for current_line, current_row in iterator:
            consume(current_line, current_row)

    parameters = {
        "accepted_reliability": sorted(accepted),
        "evidence_columns": list(evidence_columns),
        "evidence_reliability": (
            None
            if per_column_reliability is None
            else [
                {
                    "column": column,
                    "reliability": per_column_reliability[column],
                }
                for column in evidence_columns
            ]
        ),
        "gene_column": gene_column,
        "include_uncertain": bool(include_uncertain),
        "go_id_class_identity": "not_used_non_bijective_or_missing",
        "label_id_policy": HPA_LABEL_ID_POLICY,
        "label_id_template": HPA_LABEL_ID_TEMPLATE,
        "label_delimiter": ";",
        "reliability_column": reliability_column,
    }
    parser_statistics = {
        "input_rows": parsed_rows,
        "rows_matching_reliability": accepted_rows,
    }
    return annotations, selected_member, parameters, parser_statistics


def _parse_reactome(
    input_path: PathLike,
    archive_member: Optional[str],
    protein_universe: Optional[set],
) -> Tuple[List[Row], Optional[str], Dict[str, int]]:
    annotations = []
    pathway_id_to_name = {}
    pathway_name_to_id = {}
    parsed_pathways = 0
    input_members = 0
    members_in_universe = 0
    members_outside_universe = 0
    members_rejected_case_mismatch = 0
    pathways_without_universe_members = 0
    casefolded_universe = (
        None
        if protein_universe is None
        else {protein.casefold() for protein in protein_universe}
    )
    input_name = str(input_path)

    with _open_text_input(input_path, ".gmt", archive_member) as opened:
        handle, selected_member = opened
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    "Reactome GMT line {} must contain a pathway name, stable ID, "
                    "and at least one gene".format(line_number)
                )
            location = "{}:line {}".format(input_name, line_number)
            pathway_name = _validate_text(fields[0], "pathway name", location)
            stable_id = _validate_text(fields[1], "Reactome stable ID", location)
            if not _REACTOME_HUMAN_STABLE_ID.fullmatch(stable_id):
                raise ValueError(
                    "Expected a human Reactome stable ID (R-HSA-...), got {!r} at {}".format(
                        stable_id, location
                    )
                )

            previous_name = pathway_id_to_name.get(stable_id)
            if previous_name is not None and previous_name != pathway_name:
                raise ValueError(
                    "Reactome stable ID {} maps to conflicting names {!r} and {!r}".format(
                        stable_id, previous_name, pathway_name
                    )
                )
            previous_id = pathway_name_to_id.get(pathway_name)
            if previous_id is not None and previous_id != stable_id:
                raise ValueError(
                    "Reactome pathway name {!r} maps to conflicting stable IDs {} and {}".format(
                        pathway_name, previous_id, stable_id
                    )
                )
            pathway_id_to_name[stable_id] = pathway_name
            pathway_name_to_id[pathway_name] = stable_id
            parsed_pathways += 1

            pathway_members_in_universe = 0
            for field_number, raw_gene in enumerate(fields[2:], start=3):
                input_members += 1
                if protein_universe is not None:
                    # Deliberately perform membership lookup before strict gene
                    # normalization.  Reactome infectious-disease pathways can
                    # contain entities such as "16S rRNA" or viral display
                    # names; these are safe to discard only because the caller
                    # supplied a frozen human-protein universe.
                    lookup_symbol = raw_gene.strip()
                    if lookup_symbol not in protein_universe:
                        members_outside_universe += 1
                        if lookup_symbol.casefold() in casefolded_universe:
                            members_rejected_case_mismatch += 1
                        continue
                    members_in_universe += 1
                    pathway_members_in_universe += 1
                protein = _normalize_gene_symbol(
                    raw_gene, "{} field {}".format(location, field_number)
                )
                if _ENSEMBL_GENE_ID.fullmatch(protein):
                    raise ValueError(
                        "Reactome GMT must contain human gene symbols, not Ensembl IDs; "
                        "got {!r} at {}".format(protein, location)
                    )
                annotations.append(
                    {
                        "protein": protein,
                        "label": pathway_name,
                        "label_id": stable_id,
                        "source": REACTOME_SOURCE,
                    }
                )
            if protein_universe is not None and pathway_members_in_universe == 0:
                pathways_without_universe_members += 1

    if parsed_pathways == 0:
        raise ValueError("Reactome GMT contains no pathway records: {}".format(input_name))
    parser_statistics = {
        "input_member_fields": input_members,
        "input_pathway_rows": parsed_pathways,
    }
    if protein_universe is not None:
        parser_statistics.update(
            {
                "member_fields_in_protein_universe": members_in_universe,
                "member_fields_outside_protein_universe": members_outside_universe,
                "member_fields_rejected_case_mismatch": (
                    members_rejected_case_mismatch
                ),
                "pathway_rows_without_protein_universe_members": (
                    pathways_without_universe_members
                ),
            }
        )
    return annotations, selected_member, parser_statistics


def _canonicalize_and_filter(
    rows: Iterable[Row],
    min_positive_count: int,
    max_positive_count: Optional[int],
) -> Tuple[List[Row], Dict[str, int]]:
    _validate_thresholds(min_positive_count, max_positive_count)
    unique = set()
    id_to_label = {}
    label_to_id = {}
    for row in rows:
        values = tuple(row[column] for column in CANONICAL_COLUMNS)
        unique.add(values)
        label = row["label"]
        label_id = row["label_id"]
        if label_id in id_to_label and id_to_label[label_id] != label:
            raise ValueError("Label ID {!r} maps to multiple labels".format(label_id))
        if label in label_to_id and label_to_id[label] != label_id:
            raise ValueError("Label {!r} maps to multiple label IDs".format(label))
        id_to_label[label_id] = label
        label_to_id[label] = label_id

    if not unique:
        raise ValueError("No annotations remain after input and reliability validation")

    proteins_by_label = {}
    for protein, label, label_id, source in unique:
        proteins_by_label.setdefault((label, label_id, source), set()).add(protein)
    kept_labels = {
        key
        for key, proteins in proteins_by_label.items()
        if len(proteins) >= min_positive_count
        and (max_positive_count is None or len(proteins) <= max_positive_count)
    }
    filtered = [
        dict(zip(CANONICAL_COLUMNS, values))
        for values in unique
        if (values[1], values[2], values[3]) in kept_labels
    ]
    filtered.sort(key=lambda row: tuple(row[column] for column in CANONICAL_COLUMNS))
    if not filtered:
        maximum = "unbounded" if max_positive_count is None else str(max_positive_count)
        raise ValueError(
            "No labels remain after positive-count filtering (min={}, max={})".format(
                min_positive_count, maximum
            )
        )

    before_proteins = {values[0] for values in unique}
    after_proteins = {row["protein"] for row in filtered}
    statistics = {
        "annotations_after_filter": len(filtered),
        "annotations_before_filter": len(unique),
        "labels_after_filter": len(kept_labels),
        "labels_before_filter": len(proteins_by_label),
        "proteins_after_filter": len(after_proteins),
        "proteins_before_filter": len(before_proteins),
    }
    return filtered, statistics


def _csv_bytes(rows: Sequence[Row]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=CANONICAL_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=".{}-".format(path.name),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_result(
    input_path: PathLike,
    output_csv: PathLike,
    manifest_path: Optional[PathLike],
    processor: str,
    source: str,
    selected_member: Optional[str],
    parameters: Dict[str, object],
    rows: Sequence[Row],
    statistics: Dict[str, int],
    force: bool,
    protein_universe_path: Optional[PathLike] = None,
    protein_universe_provenance: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    input_file = Path(input_path).expanduser()
    output_file = Path(output_csv).expanduser()
    if manifest_path is None:
        manifest_file = output_file.with_suffix(".manifest.json")
    else:
        manifest_file = Path(manifest_path).expanduser()

    resolved_input = os.path.abspath(str(input_file))
    resolved_output = os.path.abspath(str(output_file))
    resolved_manifest = os.path.abspath(str(manifest_file))
    protected_inputs = {resolved_input}
    if protein_universe_path is not None:
        protected_inputs.add(os.path.abspath(str(Path(protein_universe_path).expanduser())))
    if (
        resolved_output == resolved_manifest
        or resolved_output in protected_inputs
        or resolved_manifest in protected_inputs
    ):
        raise ValueError("Input, output CSV, and manifest paths must be distinct")
    existing = [path for path in (output_file, manifest_file) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing output(s) without force=True: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    csv_content = _csv_bytes(rows)
    manifest = {
        "format_version": 1,
        "input": {
            "archive_member": selected_member,
            "container": "zip" if input_file.suffix.lower() == ".zip" else "plain",
            "sha256": sha256_file(input_file),
        },
        "output": {
            "columns": list(CANONICAL_COLUMNS),
            "csv_sha256": hashlib.sha256(csv_content).hexdigest(),
        },
        "parameters": parameters,
        "processor": processor,
        "source": source,
        "statistics": statistics,
    }
    if protein_universe_provenance is not None:
        manifest["protein_universe"] = protein_universe_provenance
    manifest_content = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(output_file, csv_content)
    _atomic_write(manifest_file, manifest_content)
    return manifest


def process_hpa(
    input_path: PathLike,
    output_csv: PathLike,
    manifest_path: Optional[PathLike] = None,
    gene_column: str = "Gene name",
    evidence_columns: Sequence[str] = DEFAULT_HPA_EVIDENCE_COLUMNS,
    evidence_reliability: Optional[Mapping[str, str]] = None,
    reliability_column: Optional[str] = None,
    accepted_reliability: Sequence[str] = DEFAULT_HPA_ACCEPTED_RELIABILITY,
    include_uncertain: bool = False,
    min_positive_count: int = 1,
    max_positive_count: Optional[int] = None,
    archive_member: Optional[str] = None,
    force: bool = False,
) -> Dict[str, object]:
    """Process a frozen HPA subcellular-location TSV (or ZIP).

    By default, the official confidence-specific columns ``Enhanced``,
    ``Supported``, ``Approved``, and ``Uncertain`` are used, with each column
    name interpreted as its reliability.  Only the first three are accepted.

    ``evidence_reliability`` can map custom evidence column names to confidence
    values.  Alternatively, set ``reliability_column`` to apply one row-level
    confidence value to every configured evidence column.  These two modes are
    intentionally mutually exclusive.  Uncertain evidence requires the
    explicit ``include_uncertain=True`` opt-in.

    HPA ``GO id`` values are deliberately not used as class IDs because their
    mapping to location labels is non-bijective and incomplete (for example,
    ``GO:0005730`` identifies both ``Nucleoli`` and ``Nucleoli rim``).  The
    stable class identity is therefore the explicit ``HPA:<location label>``.
    """
    rows, selected_member, parameters, parser_statistics = _parse_hpa(
        input_path=input_path,
        gene_column=gene_column,
        evidence_columns=evidence_columns,
        evidence_reliability=evidence_reliability,
        reliability_column=reliability_column,
        accepted_reliability=accepted_reliability,
        include_uncertain=include_uncertain,
        archive_member=archive_member,
    )
    filtered, statistics = _canonicalize_and_filter(
        rows, min_positive_count, max_positive_count
    )
    statistics.update(parser_statistics)
    parameters.update(
        {
            "max_positive_count": max_positive_count,
            "min_positive_count": min_positive_count,
        }
    )
    return _write_result(
        input_path=input_path,
        output_csv=output_csv,
        manifest_path=manifest_path,
        processor="hpa_subcellular_localization",
        source=HPA_SOURCE,
        selected_member=selected_member,
        parameters=parameters,
        rows=filtered,
        statistics=statistics,
        force=force,
    )


def process_reactome(
    input_path: PathLike,
    output_csv: PathLike,
    manifest_path: Optional[PathLike] = None,
    min_positive_count: int = 1,
    max_positive_count: Optional[int] = None,
    archive_member: Optional[str] = None,
    force: bool = False,
    protein_universe_path: Optional[PathLike] = None,
) -> Dict[str, object]:
    """Process a frozen human Reactome GMT (or ZIP) of gene symbols.

    Production CLI use requires ``protein_universe_path``.  The Python API
    keeps it optional for strict validation and backwards compatibility: with
    no universe every member is validated as a gene symbol, while with a
    universe raw members are filtered against it before validation and before
    label positive-count thresholds are applied.
    """
    protein_universe = None
    universe_provenance = None
    if protein_universe_path is not None:
        protein_universe, universe_provenance = _load_protein_universe(
            protein_universe_path
        )
    rows, selected_member, parser_statistics = _parse_reactome(
        input_path=input_path,
        archive_member=archive_member,
        protein_universe=protein_universe,
    )
    filtered, statistics = _canonicalize_and_filter(
        rows, min_positive_count, max_positive_count
    )
    statistics.update(parser_statistics)
    parameters = {
        "human_stable_id_pattern": _REACTOME_HUMAN_STABLE_ID.pattern,
        "max_positive_count": max_positive_count,
        "min_positive_count": min_positive_count,
        "protein_identifier": "human_gene_symbol",
        "protein_universe_filter": protein_universe_path is not None,
        "protein_universe_membership_policy": (
            REACTOME_UNIVERSE_MEMBERSHIP_POLICY
            if protein_universe_path is not None
            else None
        ),
    }
    return _write_result(
        input_path=input_path,
        output_csv=output_csv,
        manifest_path=manifest_path,
        processor="reactome_pathways",
        source=REACTOME_SOURCE,
        selected_member=selected_member,
        parameters=parameters,
        rows=filtered,
        statistics=statistics,
        force=force,
        protein_universe_path=protein_universe_path,
        protein_universe_provenance=universe_provenance,
    )


def _parse_evidence_arguments(
    values: Optional[Sequence[str]],
    reliability_column: Optional[str],
) -> Tuple[Tuple[str, ...], Optional[Dict[str, str]]]:
    if not values:
        return DEFAULT_HPA_EVIDENCE_COLUMNS, None
    columns = []
    mapping = {}
    saw_mapping = False
    for value in values:
        if "=" in value:
            if reliability_column is not None:
                raise ValueError(
                    "RELIABILITY=COLUMN evidence syntax cannot be combined with "
                    "--reliability-column"
                )
            reliability, column = value.split("=", 1)
            reliability = reliability.strip()
            column = column.strip()
            if not reliability or not column:
                raise ValueError("Evidence mappings must use RELIABILITY=COLUMN")
            saw_mapping = True
            columns.append(column)
            mapping[column] = reliability
        else:
            columns.append(value)
    if saw_mapping and len(mapping) != len(columns):
        raise ValueError("Either map every evidence column or none of them")
    return tuple(columns), mapping if saw_mapping else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical labels from frozen HPA or Reactome files."
    )
    subparsers = parser.add_subparsers(dest="dataset")

    hpa = subparsers.add_parser("hpa", help="Process HPA subcellular locations")
    hpa.add_argument("--input", required=True, type=Path)
    hpa.add_argument("--output-csv", required=True, type=Path)
    hpa.add_argument("--manifest", type=Path)
    hpa.add_argument("--archive-member")
    hpa.add_argument("--gene-column", default="Gene name")
    hpa.add_argument(
        "--evidence-column",
        action="append",
        help=(
            "Repeat a column name, or RELIABILITY=COLUMN for custom per-column "
            "confidence. Defaults to the four official HPA confidence columns."
        ),
    )
    hpa.add_argument(
        "--reliability-column",
        help="Optional row-level confidence column (alternative to mapped evidence columns).",
    )
    hpa.add_argument(
        "--accepted-reliability",
        action="append",
        help="Repeat accepted confidence values (default: Enhanced, Supported, Approved).",
    )
    hpa.add_argument("--include-uncertain", action="store_true")
    hpa.add_argument("--min-positive-count", type=int, default=1)
    hpa.add_argument("--max-positive-count", type=int)
    hpa.add_argument("--force", action="store_true")

    reactome = subparsers.add_parser("reactome", help="Process human Reactome GMT")
    reactome.add_argument("--input", required=True, type=Path)
    reactome.add_argument("--output-csv", required=True, type=Path)
    reactome.add_argument(
        "--protein-universe",
        required=True,
        type=Path,
        help=(
            "Frozen one-column human protein-symbol list or two-column protein "
            "edge list used to remove non-protein GMT members."
        ),
    )
    reactome.add_argument("--manifest", type=Path)
    reactome.add_argument("--archive-member")
    reactome.add_argument("--min-positive-count", type=int, default=1)
    reactome.add_argument("--max-positive-count", type=int)
    reactome.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dataset is None:
        parser.error("choose a dataset: hpa or reactome")

    if args.dataset == "hpa":
        try:
            evidence_columns, evidence_reliability = _parse_evidence_arguments(
                args.evidence_column, args.reliability_column
            )
        except ValueError as exc:
            parser.error(str(exc))
        manifest = process_hpa(
            input_path=args.input,
            output_csv=args.output_csv,
            manifest_path=args.manifest,
            gene_column=args.gene_column,
            evidence_columns=evidence_columns,
            evidence_reliability=evidence_reliability,
            reliability_column=args.reliability_column,
            accepted_reliability=(
                args.accepted_reliability or DEFAULT_HPA_ACCEPTED_RELIABILITY
            ),
            include_uncertain=args.include_uncertain,
            min_positive_count=args.min_positive_count,
            max_positive_count=args.max_positive_count,
            archive_member=args.archive_member,
            force=args.force,
        )
    else:
        manifest = process_reactome(
            input_path=args.input,
            output_csv=args.output_csv,
            protein_universe_path=args.protein_universe,
            manifest_path=args.manifest,
            min_positive_count=args.min_positive_count,
            max_positive_count=args.max_positive_count,
            archive_member=args.archive_member,
            force=args.force,
        )

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
