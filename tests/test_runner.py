from branchmem.evaluation.runner import run_full_benchmark, write_results
from branchmem.llm.mock_backend import MockBackend


def test_run_full_benchmark_end_to_end_with_mock_backend(tmp_path):
    backend = MockBackend(
        canned_responses={
            "resolutions": (
                '{"resolutions": []}'
            )
        }
    )
    results, detector_scores, all_pairs = run_full_benchmark(
        backend=backend, n_scenarios=6, seed=1, divergence_spans=[4.0, 10.0]
    )
    assert len(results) == 6
    assert all(r.n_questions > 0 for r in results)
    assert set(detector_scores.keys()) == {"embedding_threshold", "nli", "llm_judge"}
    assert len(all_pairs) > 0

    write_results(results, detector_scores, tmp_path, run_metadata={"seed": 1})
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    csv_text = (tmp_path / "results.csv").read_text()
    assert "three_way_llm" in csv_text
    assert csv_text.count("\n") == 6  # header + 6 scenario rows, no trailing newline
