"""Independently compare every exported row to the pinned official only_final loader."""
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

B=Path(__file__).resolve().parents[1]
P=B/'task_package'
module_path=B/'materials/upstream_transition1x_dataloader.py'
spec=importlib.util.spec_from_file_location('official_t1x_loader',module_path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mapping=json.loads((B/'build/source_id_mapping.json').read_text())
by_source={r['source_path']:r for r in mapping}
assert len(by_source)==len(mapping)==10073
allowed={'reaction_id','atomic_numbers','reactant_positions_angstrom','product_positions_angstrom'}
train_extra={'forward_barrier_eV','transition_state_positions_angstrom','reactant_energy_eV','product_energy_eV','transition_state_energy_eV'}

def jsonl(path):
    with gzip.open(path,'rt') as stream:return [json.loads(s) for s in stream]

def csv_rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))

hidden={r['reaction_id']:r for r in jsonl(P/'verifier/test/data/reference_states.jsonl.gz')}
result={'official_loader_sha256':hashlib.sha256(module_path.read_bytes()).hexdigest(),'split_checks':{}}
for split,filename in [('train','train'),('val','validation'),('test','test')]:
    records=jsonl(P/f'input/{filename}.jsonl.gz')
    public={r['reaction_id']:r for r in records};assert len(public)==len(records)
    expected_keys=allowed|train_extra if split=='train' else allowed
    assert all(set(r)==expected_keys for r in records)
    labels={}
    if split!='train':
        source_labels=csv_rows(P/f'verifier/test/data/{filename}_labels.csv')
        labels={r['reaction_id']:float(r['forward_barrier_eV']) for r in source_labels}
        assert len(labels)==len(source_labels)==len(records)
        template=csv_rows(P/('input/validation_predictions.csv' if split=='val' else 'input/predictions.csv'))
        assert {r['reaction_id'] for r in template}==public.keys()
        assert all(r['forward_barrier_eV']=='' for r in template)
    seen=set()
    for reaction in module.Dataloader(B/'materials/Transition1x.h5',datasplit=split,only_final=True):
        r,p,t=(reaction[s] for s in ['reactant','product','transition_state'])
        source_path=f"/data/{r['formula']}/{reaction['rxn']}"
        rid=by_source[source_path]['reaction_id'];assert by_source[source_path]['split']==split
        assert rid not in seen;seen.add(rid)
        out=public[rid]
        assert np.array_equal(out['atomic_numbers'],r['atomic_numbers'])
        assert np.array_equal(out['reactant_positions_angstrom'],r['positions'])
        assert np.array_equal(out['product_positions_angstrom'],p['positions'])
        assert np.array_equal(r['atomic_numbers'],p['atomic_numbers']) and np.array_equal(r['atomic_numbers'],t['atomic_numbers'])
        barrier=t['wB97x_6-31G(d).energy']-r['wB97x_6-31G(d).energy'];assert math.isfinite(barrier)
        target=out if split=='train' else hidden[rid]
        assert target['forward_barrier_eV']==barrier
        assert np.array_equal(target['transition_state_positions_angstrom'],t['positions'])
        for name,state in [('reactant',r),('product',p),('transition_state',t)]:
            assert target[name+'_energy_eV']==state['wB97x_6-31G(d).energy']
        if split!='train':assert labels[rid]==barrier
    assert seen==public.keys()
    result['split_checks'][split]={'count':len(seen),'all_fields_exactly_match_official_only_final_loader':True,'input_field_allowlist_passed':True,'full_id_coverage':True}
    print('Checked official loader',split,len(seen),flush=True)
result['passed']=True
(B/'build/official_loader_export_check.json').write_text(json.dumps(result,indent=2)+'\n')
