import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from downstream_tasks.data_processing import functional_annotation_processing as annotations


class FunctionalAnnotationProcessingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _read_csv(self, path):
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_hpa_uses_gene_symbols_confidence_columns_and_filters_after_dedup(self):
        hpa = self.root / "subcellular_location.tsv"
        hpa.write_text(
            "Gene\tGene name\tReliability\tEnhanced\tSupported\tApproved\tUncertain\n"
            "ENSG2\tgeneb\tEnhanced\tNucleoplasm; Cytosol\t\tNucleoplasm\tVesicles\n"
            "ENSG1\tGENEA\tSupported\tCytosol\t\t\tMitochondria\n"
            "ENSG3\tGENEC\tUncertain\t\tCytosol\t\tPlasma membrane\n",
            encoding="utf-8",
        )
        output = self.root / "hpa.csv"

        manifest = annotations.process_hpa(
            hpa,
            output,
            min_positive_count=2,
            max_positive_count=3,
        )

        # Duplicate GENEB/Nucleoplasm evidence is collapsed before counting;
        # uncertain-column locations never enter the default output.
        self.assertEqual(
            self._read_csv(output),
            [
                {
                    "protein": "GENEA",
                    "label": "Cytosol",
                    "label_id": "HPA:Cytosol",
                    "source": "HPA",
                },
                {
                    "protein": "GENEB",
                    "label": "Cytosol",
                    "label_id": "HPA:Cytosol",
                    "source": "HPA",
                },
                {
                    "protein": "GENEC",
                    "label": "Cytosol",
                    "label_id": "HPA:Cytosol",
                    "source": "HPA",
                },
            ],
        )
        self.assertEqual(manifest["statistics"]["labels_before_filter"], 2)
        self.assertEqual(manifest["statistics"]["labels_after_filter"], 1)
        self.assertEqual(manifest["statistics"]["annotations_before_filter"], 4)
        self.assertEqual(
            manifest["parameters"]["label_id_policy"],
            "source_scoped_location_label",
        )
        self.assertEqual(
            manifest["parameters"]["label_id_template"], "HPA:<location label>"
        )
        self.assertEqual(
            manifest["parameters"]["go_id_class_identity"],
            "not_used_non_bijective_or_missing",
        )
        self.assertNotIn("Vesicles", output.read_text(encoding="utf-8"))
        self.assertNotIn("Mitochondria", output.read_text(encoding="utf-8"))

    def test_hpa_supports_custom_evidence_mapping_and_explicit_uncertain_opt_in(self):
        hpa = self.root / "custom.tsv"
        hpa.write_text(
            "symbol\tgood_locations\tlow_confidence\n"
            "abc1\tNucleus;Cytosol\tVesicles\n",
            encoding="utf-8",
        )
        output = self.root / "custom.csv"

        annotations.process_hpa(
            hpa,
            output,
            gene_column="symbol",
            evidence_columns=("good_locations", "low_confidence"),
            evidence_reliability={
                "good_locations": "Supported",
                "low_confidence": "Uncertain",
            },
            accepted_reliability=("Supported",),
            include_uncertain=True,
        )

        self.assertEqual(
            [row["label"] for row in self._read_csv(output)],
            ["Cytosol", "Nucleus", "Vesicles"],
        )

    def test_hpa_class_ids_do_not_collapse_nonbijective_or_missing_go_ids(self):
        hpa = self.root / "hpa_go_ids.tsv"
        hpa.write_text(
            "Gene name\tEnhanced\tSupported\tApproved\tUncertain\tGO id\n"
            "GENEA\tNucleoli;Nucleoli rim\t\t\t\tGO:0005730\n"
            "GENEB\tCytosol\t\t\t\t\n",
            encoding="utf-8",
        )
        output = self.root / "hpa_go_ids.csv"

        annotations.process_hpa(hpa, output)

        self.assertEqual(
            [(row["label"], row["label_id"]) for row in self._read_csv(output)],
            [
                ("Nucleoli", "HPA:Nucleoli"),
                ("Nucleoli rim", "HPA:Nucleoli rim"),
                ("Cytosol", "HPA:Cytosol"),
            ],
        )

    def test_hpa_supports_row_level_reliability_and_rejects_missing_columns(self):
        hpa = self.root / "row_confidence.tsv"
        hpa.write_text(
            "symbol\tconfidence\tlocations\n"
            "ABC1\tEnhanced\tNucleus;Cytosol\n"
            "ABC2\tUncertain\tVesicles\n",
            encoding="utf-8",
        )
        output = self.root / "row_confidence.csv"
        annotations.process_hpa(
            hpa,
            output,
            gene_column="symbol",
            evidence_columns=("locations",),
            reliability_column="confidence",
        )
        self.assertEqual(
            [row["protein"] for row in self._read_csv(output)], ["ABC1", "ABC1"]
        )

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            annotations.process_hpa(
                hpa,
                self.root / "missing.csv",
                gene_column="not_a_column",
                evidence_columns=("locations",),
                reliability_column="confidence",
            )

    def test_reactome_zip_is_deduplicated_sorted_and_has_stable_ids(self):
        archive = self.root / "ReactomePathways.gmt.zip"
        with zipfile.ZipFile(str(archive), "w", zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("README.txt", "frozen fixture")
            handle.writestr(
                "ReactomePathways.gmt",
                "Z pathway\tR-HSA-200.1\tgeneB\tGENEA\tGENEA\n"
                "A pathway\tR-HSA-100\tGENEC\tGENEA\n",
            )
        output = self.root / "reactome.csv"

        manifest = annotations.process_reactome(
            archive,
            output,
            min_positive_count=2,
            max_positive_count=2,
        )

        self.assertEqual(
            self._read_csv(output),
            [
                {
                    "protein": "GENEA",
                    "label": "A pathway",
                    "label_id": "R-HSA-100",
                    "source": "Reactome",
                },
                {
                    "protein": "GENEA",
                    "label": "Z pathway",
                    "label_id": "R-HSA-200.1",
                    "source": "Reactome",
                },
                {
                    "protein": "GENEB",
                    "label": "Z pathway",
                    "label_id": "R-HSA-200.1",
                    "source": "Reactome",
                },
                {
                    "protein": "GENEC",
                    "label": "A pathway",
                    "label_id": "R-HSA-100",
                    "source": "Reactome",
                },
            ],
        )
        self.assertEqual(manifest["input"]["archive_member"], "ReactomePathways.gmt")
        self.assertEqual(
            manifest["input"]["sha256"],
            hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["statistics"]["annotations_before_filter"], 4)

    def test_reactome_filters_nonprotein_members_through_frozen_edge_universe(self):
        reactome = self.root / "pathways.gmt"
        reactome.write_text(
            "Kept pathway\tR-HSA-100\tGENEA\t16S rRNA\tGENEB\n"
            "Dropped pathway\tR-HSA-200\tGENEA\tSARS-CoV-2 nsp1\n",
            encoding="utf-8",
        )
        universe = self.root / "global_ppi_edgelist.txt"
        universe.write_text(
            "GENEA GENEB\nGENEB GENEC\n",
            encoding="utf-8",
        )
        output = self.root / "filtered.csv"

        manifest = annotations.process_reactome(
            reactome,
            output,
            protein_universe_path=universe,
            min_positive_count=2,
        )

        self.assertEqual(
            self._read_csv(output),
            [
                {
                    "protein": "GENEA",
                    "label": "Kept pathway",
                    "label_id": "R-HSA-100",
                    "source": "Reactome",
                },
                {
                    "protein": "GENEB",
                    "label": "Kept pathway",
                    "label_id": "R-HSA-100",
                    "source": "Reactome",
                },
            ],
        )
        self.assertEqual(
            manifest["protein_universe"],
            {
                "membership_policy": (
                    "raw_trimmed_case_exact_before_gene_symbol_normalization"
                ),
                "schema": "two_column_edge_list",
                "sha256": hashlib.sha256(universe.read_bytes()).hexdigest(),
                "statistics": {
                    "member_fields": 4,
                    "records": 2,
                    "unique_proteins": 3,
                },
            },
        )
        self.assertEqual(manifest["statistics"]["input_member_fields"], 5)
        self.assertEqual(
            manifest["statistics"]["member_fields_in_protein_universe"], 3
        )
        self.assertEqual(
            manifest["statistics"]["member_fields_outside_protein_universe"], 2
        )
        self.assertEqual(
            manifest["statistics"]["member_fields_rejected_case_mismatch"], 0
        )
        self.assertEqual(manifest["statistics"]["labels_before_filter"], 2)
        self.assertEqual(manifest["statistics"]["labels_after_filter"], 1)

        # Without the explicit universe, the same non-protein member remains a
        # strict validation error instead of being silently accepted.
        with self.assertRaisesRegex(ValueError, "Whitespace in gene symbol"):
            annotations.process_reactome(
                reactome,
                self.root / "strict_without_universe.csv",
                min_positive_count=2,
            )

    def test_manifest_is_deterministic_and_contains_no_machine_path_or_timestamp(self):
        reactome = self.root / "pathways.gmt"
        reactome.write_text(
            "Pathway\tR-HSA-123\tGENEA\tGENEB\n", encoding="utf-8"
        )
        first = self.root / "one.csv"
        second = self.root / "two.csv"
        first_universe = self.root / "universe_one.txt"
        second_universe = self.root / "universe_two.txt"
        universe_content = "GENEA\nGENEB\n"
        first_universe.write_text(universe_content, encoding="utf-8")
        second_universe.write_text(universe_content, encoding="utf-8")

        annotations.process_reactome(
            reactome, first, protein_universe_path=first_universe
        )
        annotations.process_reactome(
            reactome, second, protein_universe_path=second_universe
        )

        first_manifest = first.with_suffix(".manifest.json").read_bytes()
        second_manifest = second.with_suffix(".manifest.json").read_bytes()
        self.assertEqual(first_manifest, second_manifest)
        manifest = json.loads(first_manifest.decode("utf-8"))
        self.assertNotIn(str(self.root), first_manifest.decode("utf-8"))
        self.assertNotIn("timestamp", manifest)
        self.assertEqual(
            manifest["output"]["columns"],
            ["protein", "label", "label_id", "source"],
        )
        self.assertEqual(manifest["protein_universe"]["schema"], "symbol_list")
        self.assertEqual(
            manifest["protein_universe"]["membership_policy"],
            "raw_trimmed_case_exact_before_gene_symbol_normalization",
        )
        self.assertEqual(
            manifest["protein_universe"]["statistics"],
            {"member_fields": 2, "records": 2, "unique_proteins": 2},
        )

    def test_protein_universe_schema_is_strict_and_reactome_cli_requires_it(self):
        malformed = self.root / "mixed_universe.txt"
        malformed.write_text("GENEA\nGENEA GENEB\n", encoding="utf-8")
        reactome = self.root / "pathways.gmt"
        reactome.write_text("Pathway\tR-HSA-123\tGENEA\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Inconsistent protein-universe schema"):
            annotations.process_reactome(
                reactome,
                self.root / "malformed.csv",
                protein_universe_path=malformed,
            )

        parser = annotations._build_parser()
        reactome_subparser = next(
            action
            for action in parser._actions
            if getattr(action, "dest", None) == "dataset"
        ).choices["reactome"]
        universe_action = next(
            action
            for action in reactome_subparser._actions
            if getattr(action, "dest", None) == "protein_universe"
        )
        self.assertTrue(universe_action.required)

    def test_reactome_universe_membership_rejects_pathogen_case_collisions(self):
        human_symbols = ("DLAT", "FES", "FGD1", "MIP", "MSRA", "PTPA", "TAT")
        pathogen_tokens = ("dlaT", "fes", "fgd1", "mip", "msrA", "ptpA", "tat")
        universe = self.root / "human_symbols.txt"
        universe.write_text("\n".join(human_symbols) + "\n", encoding="utf-8")
        reactome = self.root / "case_collisions.gmt"
        reactome.write_text(
            "Case collision pathway\tR-HSA-999\t"
            + "\t".join(human_symbols + pathogen_tokens)
            + "\n",
            encoding="utf-8",
        )
        output = self.root / "case_exact.csv"

        manifest = annotations.process_reactome(
            reactome,
            output,
            protein_universe_path=universe,
            min_positive_count=len(human_symbols),
            max_positive_count=len(human_symbols),
        )

        self.assertEqual(
            [row["protein"] for row in self._read_csv(output)],
            sorted(human_symbols),
        )
        self.assertEqual(
            manifest["statistics"]["member_fields_in_protein_universe"], 7
        )
        self.assertEqual(
            manifest["statistics"]["member_fields_outside_protein_universe"], 7
        )
        self.assertEqual(
            manifest["statistics"]["member_fields_rejected_case_mismatch"], 7
        )
        self.assertEqual(
            manifest["parameters"]["protein_universe_membership_policy"],
            "raw_trimmed_case_exact_before_gene_symbol_normalization",
        )

    def test_reactome_rejects_nonhuman_ids_ensembl_genes_and_ambiguous_zips(self):
        invalid_id = self.root / "nonhuman.gmt"
        invalid_id.write_text(
            "Mouse pathway\tR-MMU-123\tGENEA\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "human Reactome stable ID"):
            annotations.process_reactome(invalid_id, self.root / "nonhuman.csv")

        ensembl = self.root / "ensembl.gmt"
        ensembl.write_text(
            "Human pathway\tR-HSA-123\tENSG00000123456\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "gene symbols, not Ensembl"):
            annotations.process_reactome(ensembl, self.root / "ensembl.csv")

        ambiguous = self.root / "ambiguous.zip"
        with zipfile.ZipFile(str(ambiguous), "w") as handle:
            handle.writestr("one.gmt", "One\tR-HSA-1\tGENEA\n")
            handle.writestr("two.gmt", "Two\tR-HSA-2\tGENEB\n")
        with self.assertRaisesRegex(ValueError, "exactly one .gmt"):
            annotations.process_reactome(ambiguous, self.root / "ambiguous.csv")

        annotations.process_reactome(
            ambiguous,
            self.root / "selected.csv",
            archive_member="two.gmt",
        )
        self.assertEqual(self._read_csv(self.root / "selected.csv")[0]["protein"], "GENEB")

    def test_strict_threshold_and_overwrite_validation(self):
        reactome = self.root / "pathways.gmt"
        reactome.write_text("Pathway\tR-HSA-123\tGENEA\n", encoding="utf-8")
        output = self.root / "labels.csv"

        with self.assertRaisesRegex(ValueError, "No labels remain"):
            annotations.process_reactome(
                reactome, output, min_positive_count=2
            )
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            annotations.process_reactome(
                reactome,
                output,
                min_positive_count=2,
                max_positive_count=1,
            )

        annotations.process_reactome(reactome, output)
        with self.assertRaisesRegex(FileExistsError, "force=True"):
            annotations.process_reactome(reactome, output)
        annotations.process_reactome(reactome, output, force=True)

    def test_remote_inputs_are_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "remote inputs"):
            annotations.process_reactome(
                "https://reactome.org/ReactomePathways.gmt",
                self.root / "labels.csv",
            )


if __name__ == "__main__":
    unittest.main()
