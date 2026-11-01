import json
from pathlib import Path
import pytest
from research_protocol.runner import load_protocol,make_command,metrics_to_record,checkpoint_digests
from research_protocol.report import summarize
from meta_rl_bridge.episode import export_episode

P=Path(__file__).resolve().parents[1]/'experiments/PROSPECTIVE_PROTOCOL_V1.json'

def test_protocol_has_new_scenarios_and_no_overlap_with_paper():
    p=load_protocol(P);old=json.loads((P.parents[1]/'PAPER_PROTOCOL.json').read_text())
    assert not (set(p['tasks'])&set(old['groups']['holdout']))
    assert len(p['tasks'])>=8
    assert set(p['methods'])=={'routed_il','baseline_il','teacher_nmpc'}

def test_same_task_command_and_mass_unsupported(tmp_path):
    p=load_protocol(P);task=p['tasks']['g01_w70c19_120_65']
    il,err=make_command('routed_il',task,10,tmp_path/'il',100,6)
    nm,err2=make_command('teacher_nmpc',task,10,tmp_path/'nmpc',100,6)
    assert err is None and err2 is None
    assert '--safety-filter' in il and '--progress-assist' in il
    assert '--query-expert' not in il
    assert '--maxiter' in nm and 'ship_state_local' in nm
    bad,reason=make_command('teacher_nmpc',p['tasks']['g07_w72c26_260_250_shiftmass'],10,tmp_path,100,6)
    assert bad is None and 'mass' in reason

def test_new_status_never_inferred_from_small_error():
    m={'status':'TIMEOUT','success':False,'unsafe':False,'position_error_m':0.1}
    r=metrics_to_record(m,'routed_il','x',1,'abcd')
    assert r['success'] is False
    vals=summarize([r]);assert vals['per_method']['routed_il']['successes']==0

def test_unknown_parameters_rejected(tmp_path):
    p=load_protocol(P)
    p['tasks']['g01_w70c19_120_65']['invented']=5
    f=tmp_path/'bad.json';f.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Unknown task attributes'):
        load_protocol(f)


def test_meta_history_causal_keys(tmp_path):
    import csv
    run=tmp_path/'run';run.mkdir()
    keys=['time_s','x_m','y_m','psi_deg','u_mps','v_mps','r_deg_s']
    for i in range(1,5):
        keys += [f'tug{i}_{q}' for q in ('x_m','y_m','psi_deg','thrust_N','fraction','offset_deg')]
    records=[]
    for j in range(2):
        d={k:0.0 for k in keys};d['time_s']=float(j*10);d['x_m']=float(j)
        records.append(d)
    with (run/'trajectory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    paths=export_episode(run,tmp_path/'export')
    assert len(paths)==4
    info=json.loads(Path(paths[0]).read_text().strip())
    assert info['t_observation_s']==0.0
    assert info['t_next_observation_s']==10.
    assert 'a_posteriori' in info['label_visibility']
    assert 'peer_tug_measurements' not in info


def test_latex_summary_displays_zero_success_for_failure():
    from research_protocol.latex_summary import make_table
    r=dict(method='routed_il',task='a_1',seed=1,status='TIMEOUT',success=False,
           unsafe=False,time_s=6500.,position_error_m=1.2)
    latex=make_table([r])
    assert '0/1' in latex
    assert 'a\\_1' in latex
    assert '6500' not in latex  # never calls a failed run a successful completion time
