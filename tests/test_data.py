import tempfile

import mlflow
import numpy as np


def test_warcraft_attack_data_from_mlflow():
    # First get all the runs in "AdversarialAttacks" experiment
    runs = mlflow.search_runs(experiment_names=["AdversarialAttacks"])
    # Now filter for the warcraft attack data
    runs = runs[runs["params.attacked_models_experiment"] == "Warcraft_Models"]
    # Only successful runs
    runs = runs[runs["status"] == "FINISHED"]
    # Now get a list of the run IDs
    print(f"Found {len(runs)} runs for Warcraft attack data")
    run_ids = runs["run_id"].tolist()
    total_runs = len(run_ids)
    for i, run_id in enumerate(run_ids):
        print(f"[{i}/{total_runs}] Testing run {run_id}")
        # Get the run data
        run_data = mlflow.get_run(run_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="adv_samples/adv_samples.npz", dst_path=tmpdir
            )
            # Now load the data
            data = np.load(path)
            Z = data["Z"]
            Z_adv = data["Z_adv"]
            c = data["c"]
            true_dec = data["dec"]
            # try to load the denormalized data directly, else recompute it
            try:
                Z_den = data["Z_den"]
            except Exception:
                Z_den = (Z * data["z_score_std"]) + data["z_score_mean"]
            try:
                Z_adv_den = data["Z_adv_den"]
            except Exception:
                Z_adv_den = (Z_adv * data["z_score_std"]) + data["z_score_mean"]
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="adv_samples/adv_decisions.npz", dst_path=tmpdir
            )
            data = np.load(path)
            dec_hat = data["dec_hat"]
            dec_adv_hat = data["dec_adv_hat"]
        # Now verify the data
        assert c.shape == true_dec.shape
        assert Z_den.shape == Z_adv_den.shape == Z.shape == Z_adv.shape
        assert dec_hat.shape == dec_adv_hat.shape == c.shape
        # The decision should only contain 0 and 1
        assert np.all(np.isin(true_dec, [0, 1]))
        assert np.all(np.isin(dec_hat, [0, 1]))
        assert np.all(np.isin(dec_adv_hat, [0, 1]))
        # The data should at most be epsilon away from the original data
        epsilon = run_data.data.params.get("epsilon")
        epsilon = float(epsilon)
        allowed_diff = 1e-5
        assert epsilon is not None, f"Epsilon value not found in run data for run {run_id}"
        diff = np.abs(Z_den - Z_adv_den).max(axis=1)
        assert np.all(diff <= epsilon + allowed_diff), (
            f"Max difference {diff.max()} is greater than epsilon {epsilon} for run {run_id}"
        )
        # Also all the data should be in the range of 0 to 255 for each rgb channel
        all_pixels = Z_adv_den.flatten()
        assert ((all_pixels >= 0 - allowed_diff) & (all_pixels <= 255 + allowed_diff)).all(), (
            f"Some pixel values are out of range for run {run_id}"
        )
        print(f"[{i}/{total_runs}] ✓ Run {run_id} passed validation")
    print(f"\n🎉 Successfully validated the data of {total_runs} attackers!")


def test_knapsack_attack_data_from_mlflow():
    # First get all the runs in "AdversarialAttacks" experiment
    runs = mlflow.search_runs(experiment_names=["AdversarialAttacks"])
    # Now filter for the warcraft attack data
    runs = runs[runs["params.attacked_models_experiment"] == "Knapsack_Models"]
    # Only successful runs
    runs = runs[runs["status"] == "FINISHED"]
    runs = runs[runs["params.mask"] == "provided"]
    # Now get a list of the run IDs
    print(f"Found {len(runs)} runs for Warcraft attack data")
    run_ids = runs["run_id"].tolist()
    total_runs = len(run_ids)
    for i, run_id in enumerate(run_ids):
        print(f"[{i}/{total_runs}] Testing run {run_id}")
        # Get the run data
        run_data = mlflow.get_run(run_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="adv_samples/adv_samples.npz", dst_path=tmpdir
            )
            # Now load the data
            data = np.load(path)
            Z = data["Z"]
            Z_adv = data["Z_adv"]
            c = data["c"]
            true_dec = data["dec"]
            # try to load the denormalized data directly, else recompute it
            try:
                Z_den = data["Z_den"]
            except Exception:
                try:
                    Z_den = (Z * data["z_score_std"]) + data["z_score_mean"]
                except Exception:
                    Z_den = None
            try:
                Z_adv_den = data["Z_adv_den"]
            except Exception:
                try:
                    Z_adv_den = (Z_adv * data["z_score_std"]) + data["z_score_mean"]
                except Exception:
                    Z_adv_den = None
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="adv_samples/adv_decisions.npz", dst_path=tmpdir
            )
            data = np.load(path)
            dec_hat = data["dec_hat"]
            dec_adv_hat = data["dec_adv_hat"]
        # Now verify the data
        assert c.shape == true_dec.shape
        if Z_den is None or Z_adv_den is None:
            assert Z.shape == Z_adv.shape
        else:
            assert Z_den.shape == Z_adv_den.shape == Z.shape == Z_adv.shape
        assert dec_hat.shape == dec_adv_hat.shape == c.shape
        # The decision should only contain 0 and 1
        assert np.all(np.isin(true_dec, [0, 1]))
        assert np.all(np.isin(dec_hat, [0, 1]))
        assert np.all(np.isin(dec_adv_hat, [0, 1]))
        # The data should at most be epsilon away from the original data
        epsilon = run_data.data.params.get("epsilon")
        epsilon = float(epsilon)
        allowed_diff = 1e-5
        assert epsilon is not None, f"Epsilon value not found in run data for run {run_id}"
        diff = np.abs(Z - Z_adv).max(axis=1)
        assert np.all(diff <= epsilon + allowed_diff), (
            f"Max difference {diff.max()} is greater than epsilon {epsilon} for run {run_id}"
        )
        # Also the first 4 features should not be changed
        first_4_feat_diff = np.abs(Z[:, :, :4] - Z_adv[:, :, :4])
        first_4_feat = np.allclose(first_4_feat_diff, 0, atol=allowed_diff)
        assert first_4_feat, (
            f"First 4 features changed for run {run_id}, max diff: {first_4_feat_diff.max()}"
        )
        # Also the values should not go negative
        if Z_den is None or Z_adv_den is None:
            print(f"Denormalized data not found for run {run_id}, skipping negative check")
        else:
            assert np.all(Z_adv_den[:, :, -4:] >= 0 - allowed_diff), (
                f"Some features are negative for run {run_id}, Min: {Z_adv_den[:, :, -4:].min()}, Max: {Z_adv_den.max()}"
            )
            assert np.all(Z_den[:, :, -4:] >= 0 - allowed_diff), (
                f"Some features are negative for run {run_id}, Min: {Z_den[:, :, -4:].min()}, Max: {Z_den.max()}"
            )
        print(f"[{i}/{total_runs}] ✓ Run {run_id} passed validation")
    print(f"\n🎉 Successfully validated the data of {total_runs} attackers!")
