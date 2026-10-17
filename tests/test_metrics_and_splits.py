"""Unit tests for the parts that decide every reported number. Run: make test"""
import numpy as np
import pandas as pd

from arthritis import metrics as M
from arthritis.splits import attach, make_folds


def _toy(n_pat=80, joints=16, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        frailty = rng.normal()
        for v in range(2):
            for j in range(joints):
                y = int(rng.random() < 1 / (1 + np.exp(-(frailty - 2))))
                rows.append({"patient_id": f"P{i}", "case": f"P{i}_{v}", "joint": j, "label": y,
                             "center": f"C{i % 4}", "split": "test" if i % 5 == 0 else "train"})
    return pd.DataFrame(rows)


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert np.isnan(M.auroc([1, 1], [0.2, 0.3]))


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500); p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    thr = M.threshold_for_sensitivity(y, p, 0.9)
    assert M.confusion_at(y, p, thr)["sensitivity"] >= 0.9


def test_sens_spec_operating_points():
    y = np.array([0] * 10 + [1] * 10); p = np.linspace(0, 1, 20)
    assert M.sens_at_spec(y, p, 0.9) == 1.0 and M.spec_at_sens(y, p, 0.9) == 1.0


def test_platt_fixes_prevalence_shift_temperature_cannot():
    rng = np.random.default_rng(2)
    y = (rng.random(4000) < 0.1).astype(int)
    z = 1.5 * (y - 0.5) + rng.normal(0, 1, 4000) + 2.0     # every logit shifted up, as after balanced sampling
    p = 1 / (1 + np.exp(-z))
    p_platt = M.apply_platt(p, M.fit_platt(y, p))
    p_temp = M.apply_temperature(p, M.fit_temperature(y, p))
    assert M.ece(y, p_platt) < 0.03
    assert M.ece(y, p_platt) < M.ece(y, p_temp)


def test_cluster_bootstrap_ci_contains_point_and_is_wider_than_naive():
    df = _toy()
    p = np.clip(df.label * 0.4 + np.random.default_rng(3).random(len(df)) * 0.6, 0, 1)
    clus = M.cluster_bootstrap(df.label, p, df.patient_id, M.auroc, 400)
    naive = M.cluster_bootstrap(df.label, p, np.arange(len(df)), M.auroc, 400)
    assert clus["ci_low"] <= clus["value"] <= clus["ci_high"]
    assert (clus["ci_high"] - clus["ci_low"]) >= (naive["ci_high"] - naive["ci_low"]) * 0.9


def test_hand_aggregation_max():
    cases, g, y, p = M.aggregate_to_hand(["a", "a", "b"], ["P1", "P1", "P2"], [0, 1, 0], [0.2, 0.7, 0.4])
    assert list(cases) == ["a", "b"] and list(y) == [1, 0] and np.allclose(p, [0.7, 0.4])


def test_folds_are_patient_disjoint_and_test_is_sealed():
    man = _toy()
    folds = make_folds(man, 5, 0, "test")
    m = attach(man, folds)
    assert m.groupby("patient_id").fold.nunique().max() == 1
    sealed = set(man.loc[man.split == "test", "patient_id"])
    assert set(m.loc[m.role == "sealed", "patient_id"]) == sealed
    assert not sealed & set(m.loc[m.role == "dev", "patient_id"])


def test_leaked_patient_is_moved_to_sealed():
    man = _toy()
    leak = man[(man.patient_id == "P0")].copy()
    leak["split"] = "train"; leak["case"] = leak["case"] + "_dup"
    m = attach(pd.concat([man, leak]), make_folds(pd.concat([man, leak]), 5, 0, "test"))
    assert (m.loc[m.patient_id == "P0", "role"] == "sealed").all()


def test_study_stem_and_file_patient_token():
    from arthritis.manifest import _file_patient_token, _study_stem
    assert _study_stem("JP_HMCRD_P0185_20150602_0498_R") == "JP_HMCRD_P0185_20150602_0498"
    assert _study_stem("JP_HMCRD_P0185_20150602_0498") == "JP_HMCRD_P0185_20150602_0498"
    assert _file_patient_token("JP_SARC_P0001_20100423_7383_R") == ("SARC", 1)


def test_same_patient_number_in_two_centers_is_two_patients(tmp_path):
    """Patient numbers restart per center in RAM-H1200; HMCRD P0001 and SARC P0001 are different people."""
    import json
    from arthritis.manifest import _join_metadata
    meta = pd.DataFrame({"Mapped Image Stem": ["JP_HMCRD_P0001_20200101_1111", "JP_SARC_P0001_20200101_2222"],
                         "Normalized PatientID": [1, 1], "Center": ["HMCRD", "SARC"], "isRA": [1, 1]})
    meta.to_excel(tmp_path / "Metadata.xlsx", index=False)
    df = pd.DataFrame({"case": ["JP_HMCRD_P0001_20200101_1111_L", "JP_SARC_P0001_20200101_2222_R"],
                       "joint": ["R", "R"], "score": [0, 1], "label": [0, 1], "split": ["train", "train"]})
    cfg = {"data": {"metadata_file": str(tmp_path / "Metadata.xlsx"), "metadata_image_col": "auto",
                    "metadata_patient_col": "auto", "metadata_center_col": "auto", "metadata_isra_col": "auto",
                    "patient_key": "auto"}}
    out = _join_metadata(tmp_path, df, cfg)
    assert out.patient_id.nunique() == 2
