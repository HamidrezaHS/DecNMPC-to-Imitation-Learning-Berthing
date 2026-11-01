from meta_rl_bridge.tasks import generate

def test_future_tasks_reproducible_and_disjoint():
    a,b=generate(77,3,2,2),generate(77,3,2,2)
    assert a==b
    ids=[t['task_id'] for t in a['tasks']]
    assert len(set(ids))==len(ids)
    assert all(t['teacher_model_access'] is False for t in a['tasks'])
    assert all(t['wind_mps']>9 for t in a['tasks'] if t['partition']=='ood')

def test_future_tasks_valid_counts():
    import pytest
    with pytest.raises(ValueError):generate(n_train=0)
