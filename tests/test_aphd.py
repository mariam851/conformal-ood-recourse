import numpy as np
import pytest

from conformal_trees.signals import aphd, leaf_mass, reference_score, tood_batch_score


def _brute_force_aphd(query, reference, exclude_self=False):
    """Textbook definition: average over reference rows of the Hamming distance."""
    out = []
    for i, q in enumerate(query):
        dists = []
        for j, r in enumerate(reference):
            if exclude_self and i == j:
                continue
            dists.append(np.mean(q != r))          # fraction of trees in different leaves
        out.append(np.mean(dists))
    return np.array(out)


@pytest.fixture
def leaves():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 6, size=(40, 12))
    qry = rng.integers(0, 8, size=(15, 12))        # leaf ids 6, 7 never seen in the reference
    return qry, ref


def test_leaf_mass_identity_matches_pairwise_definition(leaves):
    qry, ref = leaves
    assert np.allclose(aphd(qry, ref), _brute_force_aphd(qry, ref))
    assert np.allclose(leaf_mass(qry, ref), 1.0 - _brute_force_aphd(qry, ref))


def test_leave_one_out_matches_pairwise_definition(leaves):
    _, ref = leaves
    assert np.allclose(aphd(ref, ref, query_is_reference=True), _brute_force_aphd(ref, ref, exclude_self=True))
    assert np.allclose(tood_batch_score(ref), leaf_mass(ref, ref, query_is_reference=True))


def test_scores_live_in_unit_interval_and_unseen_leaves_are_maximally_far(leaves):
    qry, ref = leaves
    s = reference_score(qry, ref)
    assert ((s >= 0) & (s <= 1)).all()
    never_seen = np.full((1, ref.shape[1]), 99)
    assert reference_score(never_seen, ref)[0] == 1.0       # shares no leaf with any ID point


def test_points_sharing_leaves_get_higher_ood_score():
    # 30 "ID" rows spread over many leaves, 10 "OOD" rows all in the same leaf of every tree.
    rng = np.random.default_rng(1)
    id_rows = rng.integers(0, 50, size=(30, 20))
    ood_rows = np.full((10, 20), 77)
    batch = np.vstack([id_rows, ood_rows])
    s = tood_batch_score(batch)
    assert s[30:].min() > s[:30].max()             # clustering OOD points look "more OOD" in TOOD


def test_shape_mismatch_rejected(leaves):
    qry, ref = leaves
    with pytest.raises(ValueError):
        leaf_mass(qry[:, :5], ref)


def test_reference_and_batch_scores_have_opposite_sign_conventions():
    """The inversion bug: far-from-ID points must score HIGH under the reference score."""
    rng = np.random.default_rng(2)
    ref = rng.integers(0, 10, size=(200, 30))               # ID reference occupies leaves 0..9
    id_q = rng.integers(0, 10, size=(50, 30))
    far_q = rng.integers(100, 110, size=(50, 30))           # leaves the reference never reaches
    assert reference_score(far_q, ref).min() > reference_score(id_q, ref).max()
    