import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

from downstream_tasks.topology_baselines import (
    build_graph,
    compute_metrics,
    load_split_data,
    load_task_data,
    make_seed_matrix,
    propagate_scores,
    read_undirected_edges,
    run_baselines,
)


class TopologyBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_task(self, genes):
        path = self.root / "task.csv"
        rows = ["protein,label,label_id,source"]
        for index, gene in enumerate(genes):
            label = "Alpha" if index % 2 == 0 else "Beta"
            label_id = "L:A" if index % 2 == 0 else "L:B"
            rows.append(f"{gene},{label},{label_id},fixture")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def _write_split(self, genes):
        path = self.root / "split_indices.npz"
        arrays = {
            "genes": np.asarray(genes),
            "split_stratification": np.asarray("fixture"),
        }
        for fold in range(6):
            arrays[f"fold_{fold}"] = np.asarray(
                [2 * fold, 2 * fold + 1], dtype=np.int64
            )
        np.savez_compressed(path, **arrays)
        return path

    def test_unknown_and_held_out_vertices_are_never_seeded(self):
        graph = build_graph(
            name="global_binary",
            universe="full",
            reference_edges=(("A", "X"), ("B", "X")),
            task_genes=("A", "B", "C"),
        )
        labels = np.asarray([[1.0], [1.0], [0.0]], dtype=np.float32)
        seed, train_mask = make_seed_matrix(graph, labels, np.asarray([0, 2]))
        graph_index = {gene: index for index, gene in enumerate(graph.genes)}

        self.assertEqual(float(seed[graph_index["A"], 0]), 1.0)
        self.assertEqual(float(seed[graph_index["B"], 0]), 0.0)
        self.assertEqual(float(seed[graph_index["X"], 0]), 0.0)
        self.assertEqual(float(train_mask[graph_index["B"]]), 0.0)
        self.assertEqual(seed.shape[1], 1)

    def test_full_graph_allows_unknown_bridge_but_annotated_graph_does_not(self):
        edges = (("A", "X"), ("B", "X"))
        labels = np.asarray([[1.0], [1.0], [0.0]], dtype=np.float32)
        full = build_graph(
            name="global_binary",
            universe="full",
            reference_edges=edges,
            task_genes=("A", "B", "C"),
        )
        annotated = build_graph(
            name="global_binary",
            universe="annotated",
            reference_edges=edges,
            task_genes=("A", "B", "C"),
        )
        arguments = dict(
            labels=labels,
            train_indices=np.asarray([0, 2]),
            evaluation_indices=np.asarray([1]),
            method="label_propagation",
            normalization="random_walk",
            depth=2,
        )
        full_score, full_metadata = propagate_scores(graph=full, **arguments)
        annotated_score, annotated_metadata = propagate_scores(
            graph=annotated, **arguments
        )

        self.assertGreater(float(full_score[0, 0]), 0.0)
        self.assertEqual(float(annotated_score[0, 0]), 0.0)
        self.assertEqual(full_metadata["reachable_coverage"], 1.0)
        self.assertEqual(annotated_metadata["reachable_coverage"], 0.0)

    def test_held_out_label_value_cannot_change_direct_neighbor_score(self):
        graph = build_graph(
            name="global_binary",
            universe="annotated",
            reference_edges=(("A", "B"), ("B", "C")),
            task_genes=("A", "B", "C"),
        )
        base = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)
        changed = base.copy()
        changed[1, 0] = 1.0
        arguments = dict(
            graph=graph,
            train_indices=np.asarray([0, 2]),
            evaluation_indices=np.asarray([1]),
            method="direct_neighbor",
            normalization="none",
            depth=None,
        )
        first, _ = propagate_scores(labels=base, **arguments)
        second, _ = propagate_scores(labels=changed, **arguments)
        np.testing.assert_array_equal(first, second)

    def test_direct_neighbor_is_fraction_over_training_neighbours(self):
        graph = build_graph(
            name="global_binary",
            universe="annotated",
            reference_edges=(
                ("A", "X"),
                ("B", "X"),
                ("C", "X"),
                ("U", "X"),
            ),
            task_genes=("A", "B", "C", "X", "U"),
        )
        labels = np.asarray(
            [[1.0], [0.0], [1.0], [0.0], [1.0]], dtype=np.float32
        )

        score, _ = propagate_scores(
            graph=graph,
            labels=labels,
            train_indices=np.asarray([0, 1, 2]),
            evaluation_indices=np.asarray([3]),
            method="direct_neighbor",
            normalization="none",
            depth=None,
        )

        self.assertAlmostEqual(float(score[0, 0]), 2.0 / 3.0)

    def test_random_walk_label_propagation_is_finite_depth(self):
        graph = build_graph(
            name="global_binary",
            universe="annotated",
            reference_edges=(("A", "B"), ("B", "C")),
            task_genes=("A", "B", "C"),
        )
        labels = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)

        score, metadata = propagate_scores(
            graph=graph,
            labels=labels,
            train_indices=np.asarray([0]),
            evaluation_indices=np.asarray([1, 2]),
            method="label_propagation",
            normalization="random_walk",
            depth=1,
        )

        np.testing.assert_allclose(
            score[:, 0],
            np.asarray([1.0 / 2.0, 0.0]),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(metadata["reachable_coverage"], 1.0)

    def test_symmetric_label_propagation_matches_closed_form(self):
        graph = build_graph(
            name="global_binary",
            universe="annotated",
            reference_edges=(("A", "B"), ("B", "C")),
            task_genes=("A", "B", "C"),
        )
        labels = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)

        score, metadata = propagate_scores(
            graph=graph,
            labels=labels,
            train_indices=np.asarray([0]),
            evaluation_indices=np.asarray([1, 2]),
            method="label_propagation",
            normalization="symmetric",
            depth=1,
        )

        np.testing.assert_allclose(
            score[:, 0],
            np.asarray([1.0 / np.sqrt(2.0), 0.0]),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(metadata["reachable_coverage"], 1.0)

    def test_label_propagation_uses_exact_walk_length(self):
        graph = build_graph(
            name="global_binary",
            universe="annotated",
            reference_edges=(("A", "B"), ("B", "C")),
            task_genes=("A", "B", "C"),
        )
        labels = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)

        score, _ = propagate_scores(
            graph=graph,
            labels=labels,
            train_indices=np.asarray([0]),
            evaluation_indices=np.asarray([1, 2]),
            method="label_propagation",
            normalization="random_walk",
            depth=2,
        )

        np.testing.assert_allclose(score[:, 0], np.asarray([0.0, 0.5]))

    def test_held_out_label_value_cannot_change_propagated_score(self):
        graph = build_graph(
            name="global_binary",
            universe="annotated",
            reference_edges=(("A", "B"), ("B", "C")),
            task_genes=("A", "B", "C"),
        )
        labels = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)
        changed = labels.copy()
        changed[1, 0] = 1.0
        arguments = dict(
            graph=graph,
            train_indices=np.asarray([0, 2]),
            evaluation_indices=np.asarray([1]),
            method="label_propagation",
            normalization="random_walk",
            depth=1,
        )

        first, _ = propagate_scores(labels=labels, **arguments)
        second, _ = propagate_scores(labels=changed, **arguments)
        np.testing.assert_array_equal(first, second)

    def test_end_to_end_outputs_are_frozen_and_validation_selected(self):
        genes = tuple(f"G{index:02d}" for index in range(12))
        task_path = self._write_task(genes)
        split_path = self._write_split(genes)
        ppi_path = self.root / "global_ppi.txt"
        ppi_path.write_text(
            "\n".join(
                [f"{genes[index]} {genes[(index + 2) % len(genes)]}" for index in range(12)]
                + ["G00 BRIDGE", "BRIDGE G02"]
            )
            + "\n",
            encoding="utf-8",
        )
        output = self.root / "output"

        paths = run_baselines(
            task_csv=task_path,
            split_indices=split_path,
            global_ppi=ppi_path,
            output_dir=output,
            universes=("annotated", "full"),
            depths=(1, 2, 3),
            save_predictions=True,
        )

        for path in paths.values():
            self.assertTrue(path.is_file(), path)
        validation = paths["validation"].read_text(encoding="utf-8")
        validation_summary = paths["validation_summary"].read_text(encoding="utf-8")
        selected = paths["selection"].read_text(encoding="utf-8")
        test = paths["test"].read_text(encoding="utf-8")
        per_label = paths["per_label"].read_text(encoding="utf-8")
        ensemble = paths["ensemble"].read_text(encoding="utf-8")
        ensemble_per_label = paths["ensemble_per_label"].read_text(encoding="utf-8")
        manifest = paths["manifest"].read_text(encoding="utf-8")
        self.assertIn("validation_auprc_macro_mean", selected)
        self.assertIn("normalization", selected.splitlines()[0])
        self.assertIn("depth", selected.splitlines()[0])
        self.assertIn("validation_auprc_macro_mean", validation_summary)
        self.assertIn("test_fold", test)
        self.assertIn("all-zero graph state", manifest)
        self.assertNotIn("test_fold", validation.splitlines()[0])
        self.assertIn("average_precision", per_label.splitlines()[0])
        self.assertIn("auprc_macro", ensemble.splitlines()[0])
        self.assertIn("average_precision", ensemble_per_label.splitlines()[0])

        with paths["ensemble"].open(encoding="utf-8", newline="") as handle:
            ensemble_rows = list(csv.DictReader(handle))
        self.assertEqual(len(ensemble_rows), 4)
        self.assertTrue(all(int(row["n_ensemble_members"]) == 5 for row in ensemble_rows))

        with paths["selection"].open(encoding="utf-8", newline="") as handle:
            selection_rows = list(csv.DictReader(handle))
        self.assertEqual(len(selection_rows), 4)
        propagated = [
            row for row in selection_rows if row["method"] == "label_propagation"
        ]
        self.assertEqual(len(propagated), 2)
        self.assertTrue(
            all(
                row["normalization"] in {"random_walk", "symmetric"}
                for row in propagated
            )
        )
        self.assertTrue(
            all(int(float(row["depth"])) in {1, 2, 3} for row in propagated)
        )

        prefix = "global_binary__annotated__direct_neighbor__none__none"
        with np.load(paths["predictions"], allow_pickle=False) as archive:
            member_scores = np.stack(
                [archive[f"{prefix}__{rotation}__scores"] for rotation in range(5)],
                axis=0,
            )
            ensemble_scores = archive[f"{prefix}__ensemble__scores"]
            truth = archive["y_true"]
        np.testing.assert_allclose(
            ensemble_scores,
            member_scores.astype(np.float64).mean(axis=0).astype(np.float32),
        )
        expected_metrics = compute_metrics(truth, ensemble_scores)
        direct_neighbor_ensemble = next(
            row
            for row in ensemble_rows
            if row["graph"] == "global_binary"
            and row["universe"] == "annotated"
            and row["method"] == "direct_neighbor"
        )
        for metric in ("auprc_macro", "auprc_micro", "auroc_macro"):
            self.assertAlmostEqual(
                float(direct_neighbor_ensemble[metric]), expected_metrics[metric]
            )

    def test_edge_reader_is_undirected_deduplicated_and_strict(self):
        path = self.root / "edges.txt"
        path.write_text("B A\nA B\nA A\nB C\n", encoding="utf-8")
        self.assertEqual(read_undirected_edges(path), (("A", "B"), ("B", "C")))
        path.write_text("A B unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected two columns"):
            read_undirected_edges(path)

    def test_split_alignment_rejects_non_partition(self):
        genes = tuple(f"G{index:02d}" for index in range(12))
        task = load_task_data(self._write_task(genes))
        split = self._write_split(genes)
        loaded = load_split_data(split, task)
        self.assertEqual(loaded.genes, genes)

        arrays = {"genes": np.asarray(genes)}
        for fold in range(6):
            arrays[f"fold_{fold}"] = np.asarray([0, 1], dtype=np.int64)
        np.savez_compressed(split, **arrays)
        with self.assertRaisesRegex(ValueError, "partition"):
            load_split_data(split, task)


if __name__ == "__main__":
    unittest.main()
