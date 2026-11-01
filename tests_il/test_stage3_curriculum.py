from imitation.collect_curriculum import TRAIN_ONLY, TEST_ONLY
from imitation.robust_suite import SCENARIOS

def test_future_test_conditions_are_not_training_names():
    assert not (set(TRAIN_ONLY)&set(TEST_ONLY))
    assert TEST_ONLY.issubset(SCENARIOS)
    for name in TRAIN_ONLY:
        assert name not in TEST_ONLY
