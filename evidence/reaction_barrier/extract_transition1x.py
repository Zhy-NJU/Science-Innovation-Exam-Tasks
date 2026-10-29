"""Extract the official Transition1x v4 endpoint/barrier task without resplitting.

The source HDF5 must first pass a complete byte-count/MD5/SHA256 audit.
Only explicit endpoint fields are exported; source identifiers and audit evidence
stay under build/. The author-released converged dataset is not re-certified by
Hessian/IRC calculations, and no per-record convergence flag is invented.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform

import h5py
import numpy as np

BASE = Path(__file__).resolve().parents[1]
BUILD = BASE / 'build'
INPUT = BASE / 'task_package/input'
HIDDEN = BASE / 'task_package/verifier/test/data'
SOURCE = BASE / 'materials/Transition1x.h5'
EXPECTED_BYTES = 6623929000
EXPECTED_MD5 = 'faab0413cf6b042b6d50e6272dea9a8c'
ENERGY = 'wB97x_6-31G(d).energy'
STATES = ('reactant', 'product', 'transition_state')
SPLITS = ('train', 'val', 'test')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hash_file(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            md5.update(data); sha.update(data)
    return {'bytes': path.stat().st_size, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}


def compact(data):
    return json.dumps(data, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()


def sign(data):
    return hashlib.sha256(compact(data)).hexdigest()


def geometry_signatures(numbers, positions):
    exact = sign([numbers.tolist(), positions.tolist()])
    # This signature is an invariant candidate test, not a unique graph identifier.
    # Exact elemental composition disjointness is the stronger cross-split test.
    by_pair = defaultdict(list)
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            pair = tuple(sorted((int(numbers[i]), int(numbers[j]))))
            by_pair[pair].append(round(float(np.linalg.norm(positions[i]-positions[j])), 8))
    invariant = sign([sorted(Counter(numbers.tolist()).items()),
                      [(list(pair), sorted(ds)) for pair, ds in sorted(by_pair.items())]])
    return exact, invariant


def write_jsonl(path, rows):
    # Blank gzip filename and fixed timestamp prevent path/time information in headers.
    digest = hashlib.sha256()
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=9) as stream:
            for row in rows:
                line = compact(row) + b'\n'
                digest.update(line); stream.write(line)
    return {'path': str(path.relative_to(BASE)), 'rows': len(rows),
            'bytes': path.stat().st_size, 'sha256': hash_file(path)['sha256'],
            'uncompressed_jsonl_sha256': digest.hexdigest()}


def write_labels(path, rows, blank=False):
    with path.open('w', newline='') as stream:
        out = csv.writer(stream, lineterminator='\n')
        out.writerow(['reaction_id', 'forward_barrier_eV'])
        out.writerows((r['reaction_id'], '' if blank else repr(r['barrier'])) for r in rows)
    return {'path':str(path.relative_to(BASE)), 'rows':len(rows),
            'bytes':path.stat().st_size, 'sha256':hash_file(path)['sha256']}


def cross_groups(index):
    return [{'signature': key, 'members': members}
            for key, members in index.items() if len({x['split'] for x in members}) > 1]


def main():
    source_hash = hash_file(SOURCE)
    assert source_hash['bytes'] == EXPECTED_BYTES, source_hash
    assert source_hash['md5'] == EXPECTED_MD5, source_hash
    downloader = json.loads((BUILD/'source_hashes.json').read_text())
    assert source_hash['sha256'] == downloader['sha256']
    print('Source integrity verified', source_hash, flush=True)
    indexes = {name: defaultdict(list) for name in ['raw_source_hash', 'exact_geometry', 'distance_candidate', 'reaction_object', 'directed_pair', 'unordered_pair']}
    formula_sets = {}; composition_sets = {}; rows = []; exclusions = []
    source_shapes = {}; attrs = {}
    with h5py.File(SOURCE, 'r') as h5:
        assert set(h5) == {'data', *SPLITS}, list(h5)
        attrs = {str(k): str(v) for k,v in h5.attrs.items()}
        data_path_count = sum(len(g) for g in h5['data'].values())
        for split in SPLITS:
            formula_sets[split] = set(h5[split])
            composition_sets[split] = set()
            split_rows = []
            for formula, group in sorted(h5[split].items()):
                if group is None:
                    raise RuntimeError(f'Broken official HDF5 link: {split}/{formula}')
                for name, reaction in sorted(group.items()):
                    source_path = f'/data/{formula}/{name}'
                    meta = {'split': split, 'source_path': source_path}
                    try:
                        states = {}
                        for state in STATES:
                            sg = reaction[state]
                            source_z = np.asarray(sg['atomic_numbers'])
                            assert np.isfinite(source_z).all() and np.array_equal(source_z, np.floor(source_z)), 'noninteger atomic numbers'
                            z = np.asarray(source_z, dtype=np.int64)
                            xyz = np.asarray(sg['positions'], dtype=np.float64)
                            en = np.asarray(sg[ENERGY], dtype=np.float64)
                            assert z.ndim == 1 and z.size > 0, 'invalid atomic numbers'
                            assert xyz.shape == (1, len(z), 3), f'{state}: shape {xyz.shape}'
                            assert en.shape == (1,), f'{state}: energy shape {en.shape}'
                            assert np.isfinite(xyz).all() and np.isfinite(en).all(), 'nonfinite coordinates/energy'
                            assert set(z) <= {1,6,7,8}, 'unsupported element'
                            assert int(np.sum(z != 1)) <= 7, 'more than seven heavy atoms'
                            assert np.array_equal(z, np.asarray(reaction['atomic_numbers'])), 'state atom ordering differs from path'
                            exact, invariant = geometry_signatures(z, xyz[0])
                            source_hash_id = str(np.asarray(sg['hash']).item()) if 'hash' in sg else None
                            states[state] = {'z': z, 'xyz': xyz[0], 'energy': float(en[0]),
                                             'exact': exact, 'invariant': invariant, 'source_hash': source_hash_id}
                            tag = {**meta, 'state': state}
                            indexes['exact_geometry'][exact].append(tag)
                            indexes['distance_candidate'][invariant].append(tag)
                            if source_hash_id is not None: indexes['raw_source_hash'][source_hash_id].append(tag)
                            source_shapes.setdefault(state, {'positions_dtype': str(sg['positions'].dtype),
                                                            'energy_dtype': str(sg[ENERGY].dtype),
                                                            'keys': sorted(sg.keys()),
                                                            'attrs': {str(k):str(v) for k,v in sg.attrs.items()}})
                        z = states['reactant']['z']
                        assert all(np.array_equal(z,s['z']) for s in states.values()), 'endpoint atom ordering mismatch'
                        barrier = states['transition_state']['energy'] - states['reactant']['energy']
                        assert math.isfinite(barrier)
                        composition = tuple(sorted(Counter(z.tolist()).items()))
                        composition_sets[split].add(composition)
                        addr = str(h5py.h5o.get_info(reaction.id).addr)
                        indexes['reaction_object'][addr].append(meta)
                        rp = (states['reactant']['exact'], states['product']['exact'])
                        indexes['directed_pair'][sign(rp)].append(meta)
                        indexes['unordered_pair'][sign(sorted(rp))].append(meta)
                        split_rows.append({**meta, 'formula':formula, 'states':states, 'barrier':barrier,
                                           'path_frame_count':int(reaction['positions'].shape[0]),
                                           'source_reaction_attrs': {str(k):str(v) for k,v in reaction.attrs.items()}})
                    except (KeyError, AssertionError, ValueError) as exc:
                        exclusions.append({**meta, 'reason': str(exc)})
            # Opaque stable ordering is reproducible, while raw source names are withheld.
            split_rows.sort(key=lambda r:sign(['sie-transition1x-v4-endpoints', r['source_path']]))
            for idx,row in enumerate(split_rows,1):
                label = {'train':'train','val':'validation','test':'test'}[split]
                row['reaction_id'] = f'rxn_{label}_{idx:05d}'
            rows.extend(split_rows)
            print('Read', split, len(split_rows), 'reactions;', len(formula_sets[split]), 'formulas', flush=True)

    cross = {key: cross_groups(index) for key,index in indexes.items()}
    overlaps = []
    for a,b in itertools.combinations(SPLITS,2):
        overlaps.append({'splits':[a,b], 'source_formula_intersection':sorted(formula_sets[a]&formula_sets[b]),
                         'actual_element_count_intersection':[list(c) for c in sorted(composition_sets[a]&composition_sets[b])]})
    duplicate_pairs = {key:[{'signature':k,'members':m} for k,m in indexes[key].items() if len(m)>1]
                       for key in ['directed_pair','unordered_pair']}
    barriers = {s: np.array([r['barrier'] for r in rows if r['split']==s]) for s in SPLITS}
    audit = {
        'source':source_hash, 'source_data_group_reactions':data_path_count, 'source_root_attrs':attrs,
        'source_final_state_schema':source_shapes,
        'counts':{s:sum(r['split']==s for r in rows) for s in SPLITS},
        'private_reference_statistics':{s:{'min_eV':float(v.min()),'max_eV':float(v.max()),
                                          'negative_count':int(np.sum(v<0)),
                                          'median_eV':float(np.median(v))} for s,v in barriers.items()},
        'source_path_frame_counts':{s:sum(r['path_frame_count'] for r in rows if r['split']==s) for s in SPLITS},
        'formula_counts':{s:len(formula_sets[s]) for s in SPLITS},
        'split_formulas':{s:sorted(formula_sets[s]) for s in SPLITS},
        'actual_composition_counts':{s:len(composition_sets[s]) for s in SPLITS},
        'split_composition_overlap':overlaps,
        'cross_split_matches':cross,
        'within_split_exact_endpoint_pair_duplicates':duplicate_pairs,
        'exclusions':exclusions,
        'hash_warning':'Source hash is SHA1(str(positions)+formula) modulo 10**8; collisions alone do not prove molecular identity.',
        'invariant_signature_scope':'Element-pair sorted distances rounded to 1e-8 angstrom, invariant to rotation, translation and same-element permutation. Candidate identifier, not a unique molecular graph; exact composition disjointness separately establishes cross-split separation.',
        'path_frame_isolation':'All source frames in a reaction use its parent atomic_numbers. Composition-disjoint splits exclude identical whole-system frames even after atom permutation; no trajectory frames are exported.',
        'convergence_scope':'Use the author-released dataset of converged CINEB paths. No independent per-record convergence, Hessian or IRC certification is fabricated.',
        'environment':{'python':platform.python_version(),'h5py':h5py.__version__,'numpy':np.__version__},
    }
    dump(BUILD/'data_audit.json',audit)
    dump(BUILD/'exclusions.json',exclusions)
    # Any unexpected exclusions or actual cross-split identity requires human review.
    blocking = bool(exclusions) or any(x['actual_element_count_intersection'] for x in overlaps)
    blocking |= any(cross[key] for key in cross if key != 'raw_source_hash')
    if blocking:
        raise RuntimeError('Unexpected source defect or overlap; data_audit.json written, package not exported. Do not change split silently.')
    assert len(rows) == data_path_count == 10073, 'published reaction-count mismatch'
    for directory in [INPUT,HIDDEN]:directory.mkdir(parents=True,exist_ok=True)
    mapping=[]; output_manifest=[]; hidden_states=[]
    for split in SPLITS:
        selected=[r for r in rows if r['split']==split]
        exported=[]
        for row in selected:
            r,p,t=(row['states'][s] for s in STATES)
            public={'reaction_id':row['reaction_id'],'atomic_numbers':r['z'].tolist(),
                    'reactant_positions_angstrom':r['xyz'].tolist(),
                    'product_positions_angstrom':p['xyz'].tolist()}
            if split=='train':
                public.update({'forward_barrier_eV':row['barrier'],
                               'transition_state_positions_angstrom':t['xyz'].tolist(),
                               'reactant_energy_eV':r['energy'],
                               'product_energy_eV':p['energy'],
                               'transition_state_energy_eV':t['energy']})
            exported.append(public)
            mapping.append({'reaction_id':row['reaction_id'],'split':split,'source_path':row['source_path'],
                            'formula':row['formula'],'source_state_hashes':{s:row['states'][s]['source_hash'] for s in STATES},
                            'path_frame_count':row['path_frame_count'],
                            'source_reaction_attrs':row['source_reaction_attrs']})
            if split!='train':
                hidden_states.append({'reaction_id':row['reaction_id'],'split':split,
                                      'atomic_numbers':r['z'].tolist(),
                                      'transition_state_positions_angstrom':t['xyz'].tolist(),
                                      'reactant_energy_eV':r['energy'],'product_energy_eV':p['energy'],
                                      'transition_state_energy_eV':t['energy'],'forward_barrier_eV':row['barrier']})
        filename={'train':'train.jsonl.gz','val':'validation.jsonl.gz','test':'test.jsonl.gz'}[split]
        output_manifest.append(write_jsonl(INPUT/filename,exported))
        if split!='train':
            prefix={'val':'validation','test':'test'}[split]
            output_manifest.append(write_labels(HIDDEN/f'{prefix}_labels.csv',selected))
            output_manifest.append(write_labels(INPUT/('validation_predictions.csv' if split=='val' else 'predictions.csv'),selected,blank=True))
    output_manifest.append(write_jsonl(HIDDEN/'reference_states.jsonl.gz',hidden_states))
    dump(BUILD/'source_id_mapping.json',mapping)
    dump(BUILD/'data_manifest.json',output_manifest)
    print('EXPORT_COMPLETE', json.dumps(audit['counts']), flush=True)


if __name__=='__main__':main()
