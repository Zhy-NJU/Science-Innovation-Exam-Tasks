#!/usr/bin/env python3
"""Independent black-box reaction task tests with the real public SDK.

Task files stay read-only. Every deliberate corruption uses a real scratch copy.
Local subprocess/Actions checks are not Docker or OS-isolation certification.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import uuid
import warnings
import zipfile

STATE_KEY = 'reaction_barrier.validation.v1'
THRESHOLD = 0.0433641
COLUMNS = ['reaction_id', 'forward_barrier_eV']
METRICS = ['mae_eV', 'rmse_eV', 'p95_absolute_error_eV', 'within_1_kcal_per_mol_fraction']
FINAL_METRICS = ['forward_barrier_mae', 'forward_barrier_rmse', 'absolute_error_p95', 'within_1_kcal_per_mol_fraction']
ARGS = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_text(ids, values):
    f = io.StringIO(newline='')
    w = csv.writer(f, lineterminator='\n')
    w.writerow(COLUMNS)
    w.writerows(zip(ids, (format(float(value), '.17g') for value in values)))
    return f.getvalue()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + '\n')


def read_labels(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    return [r['reaction_id'] for r in rows], [float(r['forward_barrier_eV']) for r in rows]


def source_manifest(package):
    return [{'path':p.relative_to(package).as_posix(), 'bytes':p.stat().st_size, 'sha256':sha(p)}
            for p in sorted(package.rglob('*')) if p.is_file()]


def zip_bytes(names=('predict.py',), mode=stat.S_IFREG | 0o644):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name in names:
                item = zipfile.ZipInfo(name)
                item.create_system = 3
                item.external_attr = mode << 16
                item.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(item, b'# Independent numerical fixture; never executed.\n')
    return stream.getvalue()


def make_fixture(original, destination):
    """Replace only hidden labels and their matching protocol metadata in a copy."""
    shutil.copytree(original / 'verifier', destination / 'verifier')
    for name in ('instruction.json', 'meta.json'):
        shutil.copyfile(original / name, destination / name)
    protocol_path = destination / 'verifier/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    substitutions = {}
    counts = {}
    identifiers = {}
    for split in ('validation', 'test'):
        path = destination / ('verifier/test/data/' + split + '_labels.csv')
        ids, values = read_labels(path)
        if len(ids) < 5:
            raise AssertionError('At least five real IDs needed for a boundary fixture')
        identifiers[split] = ids[:5]
        old_digest, old_count = sha(path), len(ids)
        path.write_text(csv_text(ids[:5], [0, 0, 0, 0, 0]), encoding='utf-8')
        substitutions[old_digest] = sha(path)
        counts[old_count] = 5
    changes = []
    def replace(value, key='', path=''):
        if isinstance(value, dict):
            return {k:replace(v, k, path + '/' + k) for k,v in value.items()}
        if isinstance(value, list):
            return [replace(v, key, path + '/' + str(i)) for i,v in enumerate(value)]
        if isinstance(value, str) and value in substitutions:
            changes.append(path)
            return substitutions[value]
        if type(value) is int and value in counts and (key == 'count' or 'count' in key or key in ('n_samples','n')):
            changes.append(path)
            return counts[value]
        return value
    updated = replace(protocol)
    if len(changes) < 4:
        raise AssertionError('Protocol adaptation needs explicit count/digest fields; changed ' + repr(changes))
    write_json(protocol_path, updated)
    return identifiers, changes


def make_synthetic_fixture(original, destination):
    """Explicit approved test protocol only; never written to the source package."""
    shutil.copytree(original/'verifier', destination/'verifier')
    data=destination/'verifier/test/data';data.mkdir(parents=True,exist_ok=True)
    config={'schema_version':'1.0','task_id':'reaction_barrier','id_column':'reaction_id',
            'prediction_column':'forward_barrier_eV','metric':'forward_barrier_mae','nonnegative':False,
            'max_validation_calls':20,'chemical_accuracy_threshold_eV':THRESHOLD,
            'submission_max_bytes':2097152,'validation_request_max_bytes':4194304,'invalid_score':10.0}
    identifiers={}
    for split in ('validation','test'):
        identifiers[split]=[split+'_reaction_%02d'%i for i in range(5)]
        path=data/(split+'_labels.csv');path.write_text(csv_text(identifiers[split],[0]*5))
        config[split]={'label_file':'test/data/'+split+'_labels.csv','label_sha256':sha(path),'count':5}
    write_json(destination/'verifier/protocol.json',config)
    write_json(destination/'meta.json',{'id':'reaction_barrier'})
    return identifiers, ['Synthetic fixture only; no released reference labels or measured baseline tested.']


class ReactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix='reaction-fixture-', dir=ARGS.scratch_root)
        cls.fixture = Path(cls.shared.name) / 'task'
        cls.ids, cls.fixture_protocol_changes = (make_synthetic_fixture if ARGS.synthetic_only else make_fixture)(ARGS.package, cls.fixture)
        cls.task_id = json.loads((cls.fixture / 'meta.json').read_text())['id']

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='reaction-case-', dir=ARGS.scratch_root)
        self.root = Path(self.temp.name)
        self.package = self.fixture
        self.context = VerifyContext.create(self.root / 'context', maximum_bytes=1_000_000)
        self.submission = self.root / 'submission'
        self.submission.mkdir()
        self.make_submission()
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def make_submission(self, values=None):
        (self.submission / 'predictions.csv').write_text(csv_text(self.ids['test'], [0]*5 if values is None else values))
        (self.submission / 'method.md').write_text('Independent score fixture with explicit predictions; no fitted model.\n')
        write_json(self.submission / 'sources.json', {'data':['Synthetic independent test fixture'], 'models':[], 'software':['Python standard library']})
        (self.submission / 'research_code.zip').write_bytes(zip_bytes())

    def payload(self, values=None):
        return {'predictions_csv':csv_text(self.ids['validation'], [0]*5 if values is None else values)}

    def environment(self, context=True):
        env = dict(os.environ)
        env.update(PYTHONPATH=str(ARGS.framework / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', LANG='C.UTF-8')
        env.pop('SIE_VERIFY_CONTEXT', None)
        if context:
            env['SIE_VERIFY_CONTEXT'] = str(self.context.root)
        return env

    def invoke(self, phase, payload=None, raw=None, context=True, request_mode=None):
        self.counter += 1
        output = self.root / ('result-%s-%s.json' % (phase, self.counter))
        command = [ARGS.python, '-B', str(self.package / ('verifier/' + phase + '/run/main.py')),
                   '--input', str(ARGS.package / 'input')]
        if phase == 'validation':
            request = self.root / ('request-%s.json' % self.counter)
            data = json.dumps(self.payload() if payload is None else payload).encode() if raw is None else raw
            if request_mode == 'fifo':
                os.mkfifo(request)
            elif request_mode == 'symlink':
                target = self.root / ('request-target-%s.json' % self.counter)
                target.write_bytes(data)
                request.symlink_to(target)
            else:
                request.write_bytes(data)
            command += ['--request', str(request), '--response', str(output)]
        else:
            command += ['--submission', str(self.submission), '--result', str(output)]
        run = subprocess.run(command, env=self.environment(context), capture_output=True, text=True, timeout=15)
        data = output.read_bytes() if output.is_file() else None
        result = json.loads(data) if data is not None else None
        return run, result, data

    def valid_validation(self, payload=None):
        before = len(self.context.get(STATE_KEY, {}).get('rounds', []))
        run, result, _ = self.invoke('validation', payload=payload)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(set(result), {'status','round','remaining','n_samples','metrics'})
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['round'], before + 1)
        self.assertEqual(result['remaining'], 20 - before - 1)
        self.assertEqual(result['n_samples'], 5)
        self.assertEqual(set(result['metrics']), set(METRICS))
        self.assertTrue(all(type(v) in (int,float) and math.isfinite(v) for v in result['metrics'].values()))
        self.assertEqual(len(self.context[STATE_KEY]['rounds']), before + 1)
        return result

    def rejected_validation(self, **kwargs):
        before = self.context.snapshot()
        run, result, _ = self.invoke('validation', **kwargs)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(set(result), {'status','error'})
        self.assertEqual(result['status'], 'error')
        self.assertIsInstance(result['error'], str)
        self.assertTrue(result['error'])
        self.assertEqual(self.context.snapshot(), before)
        return result

    def final(self, valid=True):
        run, value, raw = self.invoke('test')
        self.assertEqual(run.returncode, 0, run.stderr)
        validate_result(value, self.task_id)
        self.assertEqual(value['valid'], valid, value)
        self.assertEqual(value['primary_metric']['name'], 'forward_barrier_mae')
        self.assertEqual(value['primary_metric']['direction'], 'minimize')
        if not valid:
            self.assertEqual(value['primary_metric']['value'],json.loads((self.package/'verifier/protocol.json').read_text())['invalid_score'])
        if valid:
            metrics = {r['name']:r['value'] for r in value['metrics']}
            self.assertEqual(set(metrics), set(FINAL_METRICS + ['n_samples','validation_calls']))
            self.assertEqual(metrics['n_samples'], 5)
            self.assertEqual(value['primary_metric']['value'], metrics['forward_barrier_mae'])
        return value, raw

    def test_known_scientific_values(self):
        values = [0, 0.01, -0.04, 0.1, -1]
        expected = [0.23, math.sqrt(1.0117/5), 0.82, 0.6]
        response = self.valid_validation(self.payload(values))
        for key, target in zip(METRICS, expected):
            self.assertAlmostEqual(response['metrics'][key], target, places=13)
        self.make_submission(values)
        result, _ = self.final()
        metrics = {r['name']:r['value'] for r in result['metrics']}
        for key, target in zip(FINAL_METRICS, expected):
            self.assertAlmostEqual(metrics[key], target, places=13)
        self.assertEqual(metrics['validation_calls'], 1)

    def test_exact_predictions_and_inclusive_threshold(self):
        result, _ = self.final()
        self.assertEqual(result['primary_metric']['value'], 0)
        values = [0, THRESHOLD, -THRESHOLD, math.nextafter(THRESHOLD, math.inf), -math.nextafter(THRESHOLD, math.inf)]
        response = self.valid_validation(self.payload(values))
        self.assertEqual(response['metrics']['within_1_kcal_per_mol_fraction'], 0.6)

    def test_all_finite_extreme_predictions_are_scored(self):
        largest = sys.float_info.max
        for value in [1e200, largest, -largest]:
            with self.subTest(value=value):
                response = self.valid_validation(self.payload([value]*5))
                for key in METRICS[:3]:
                    self.assertTrue(math.isclose(response['metrics'][key], abs(value), rel_tol=1e-14))
                self.make_submission([value]*5)
                result, _ = self.final()
                self.assertTrue(math.isclose(result['primary_metric']['value'], abs(value), rel_tol=1e-14))

    def test_negative_reference_barriers_are_not_clipped(self):
        local=self.root/'signed';shutil.copytree(self.fixture,local);self.package=local
        protocol=json.loads((local/'verifier/protocol.json').read_text())
        for split in ('validation','test'):
            path=local/'verifier/test/data'/(split+'_labels.csv')
            path.write_text(csv_text(self.ids[split],[-2,-1,0,1,2]))
            protocol[split]['label_sha256']=sha(path)
        write_json(local/'verifier/protocol.json',protocol)
        response=self.valid_validation(self.payload([-2,-1,0,1,2]))
        self.assertEqual(response['metrics']['mae_eV'],0)
        self.make_submission([-2,-1,0,1,2]);result,_=self.final()
        self.assertEqual(result['primary_metric']['value'],0)

    def test_20_cross_process_calls_and_exhaustion(self):
        for index in range(20):
            if index in (0, 19):
                self.rejected_validation(payload={})
            self.valid_validation()
            self.context = VerifyContext(self.context.root)
        self.rejected_validation()
        self.rejected_validation(payload={})
        self.assertEqual(len(self.context[STATE_KEY]['rounds']), 20)

    def test_repeated_valid_predictions_each_consume_a_round(self):
        self.valid_validation()
        self.valid_validation()
        rounds = self.context[STATE_KEY]['rounds']
        self.assertEqual(rounds[0]['request_sha256'], rounds[1]['request_sha256'])
        self.assertEqual(len(rounds), 2)

    def test_validation_strict_object_and_no_paths_or_aux_keys(self):
        cases = [{}, [], None, {'predictions_csv':None}, {'predictions_csv':0},
                 {'predictions_csv':'\ud800'},
                 {'predictions_csv':'/submission/predictions.csv'}, {'path':'predictions.csv'},
                 {**self.payload(), 'budget':0}, {**self.payload(), 'context':'/tmp/evil'},
                 {'payload':self.payload()}, {**self.payload(), 'selection':self.ids['validation']}]
        for value in cases:
            with self.subTest(value=value):
                self.rejected_validation(raw=json.dumps(value).encode())

    def test_validation_bad_json_encoding_duplicates_and_size(self):
        for raw in [b'{', b'\xff', b'{"predictions_csv":NaN}', b'{"predictions_csv":Infinity}',
                    b'{"predictions_csv":"a","predictions_csv":"b"}', b' '* (4*1024*1024+1)]:
            with self.subTest(size=len(raw), prefix=raw[:40]):
                self.rejected_validation(raw=raw)

    def test_validation_fifo_and_symlink_rejected_without_blocking(self):
        self.rejected_validation(request_mode='fifo')
        self.rejected_validation(request_mode='symlink')

    def test_invalid_numeric_predictions_do_not_consume_budget(self):
        for token in ['nan','NaN','inf','-Infinity','1e309','','not-a-number','1'*129]:
            text = csv_text(self.ids['validation'], [0]*5).replace(self.ids['validation'][0]+',0', self.ids['validation'][0]+','+token)
            with self.subTest(token=token[:16]):
                self.rejected_validation(payload={'predictions_csv':text})

    def test_full_development_ids_and_exact_header(self):
        original = self.payload()['predictions_csv']
        lines = original.splitlines(keepends=True)
        cases = [original.replace('forward_barrier_eV','wrong'), original + lines[1], ''.join(lines[:-1]),
                 original.replace(self.ids['validation'][0], 'unknown'), original.replace(self.ids['validation'][0], self.ids['test'][0]),
                 original.replace(self.ids['validation'][0], self.ids['validation'][1]),
                 original.replace('reaction_id,forward_barrier_eV','reaction_id,forward_barrier_eV,extra'),
                 original.replace('reaction_id,forward_barrier_eV','reaction_id,reaction_id'),
                 original.replace(self.ids['validation'][0], '../../hidden/labels.csv')]
        for index, text in enumerate(cases):
            with self.subTest(case=index):
                self.rejected_validation(payload={'predictions_csv':text})

    def test_validation_row_order_and_two_experiment_isolation(self):
        payload=self.payload([0,1,2,3,4])
        first=self.valid_validation(payload)
        lines=payload['predictions_csv'].splitlines(keepends=True)
        shuffled={'predictions_csv':lines[0]+''.join(reversed(lines[1:]))}
        second=self.valid_validation(shuffled)
        self.assertEqual(first['metrics'],second['metrics'])
        old_context=self.context
        self.context=VerifyContext.create(self.root/'second-context',maximum_bytes=1_000_000)
        third=self.valid_validation(payload)
        self.assertEqual(third['round'],1)
        self.assertEqual(len(old_context[STATE_KEY]['rounds']),2)

    def test_final_rejects_same_csv_failure_classes(self):
        original = (self.submission/'predictions.csv').read_text()
        lines = original.splitlines(keepends=True)
        cases = [original+lines[1], ''.join(lines[:-1]), original.replace(self.ids['test'][0], 'unknown'),
                 original.replace('forward_barrier_eV','wrong'), original.replace(self.ids['test'][0]+',0',self.ids['test'][0]+',NaN'),
                 original.replace(self.ids['test'][0]+',0',self.ids['test'][0]+',Inf'), original.replace(self.ids['test'][0], '../../labels.csv')]
        for index, text in enumerate(cases):
            with self.subTest(case=index):
                (self.submission/'predictions.csv').write_text(text)
                self.final(False)

    def test_final_row_order_and_repeated_frozen_result(self):
        self.valid_validation()
        self.make_submission([0,1,2,3,4])
        first, raw = self.final()
        second, raw2 = self.final()
        self.assertEqual(raw, raw2)
        path = self.submission/'predictions.csv'
        lines = path.read_text().splitlines(keepends=True)
        path.write_text(lines[0]+''.join(reversed(lines[1:])))
        third, _ = self.final()
        self.assertEqual(first, third)

    def test_validation_never_reads_final_labels(self):
        local = self.root/'isolated';shutil.copytree(self.fixture, local)
        (local/'verifier/test/data/test_labels.csv').unlink()
        self.package = local
        self.valid_validation()

    def test_missing_context_is_infrastructure_failure(self):
        for phase in ('validation','test'):
            run, result, _ = self.invoke(phase, context=False)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertTrue(result is None or result.get('valid') is not True)

    def test_state_not_taken_from_submission_files(self):
        self.valid_validation()
        (self.submission/'usage.json').write_text('{"rounds":[]}')
        self.final(False)
        self.assertEqual(len(self.context[STATE_KEY]['rounds']), 1)

    def test_corrupted_state_fails_closed_without_reset(self):
        self.valid_validation()
        good = self.context[STATE_KEY]
        cases = [[], None, {}, {'rounds':[]}, {**good,'rounds':{}}, {**good,'rounds':[{}]},
                 {**good,'rounds':good['rounds']*21}]
        for key, value in good.items():
            if key != 'rounds' and isinstance(value, str):
                cases.append({**good,key:'wrong'})
        for metric, value in [('mae_eV',True),('rmse_eV',-1),('within_1_kcal_per_mol_fraction',1.01)]:
            broken=copy.deepcopy(good);broken['rounds'][0]['metrics'][metric]=value;cases.append(broken)
        broken=copy.deepcopy(good);broken['rounds'][0]['request_sha256']='z'*64;cases.append(broken)
        for index, state in enumerate(cases):
            with self.subTest(case=index):
                self.context[STATE_KEY] = state
                before = self.context.snapshot()
                for phase in ('validation','test'):
                    run, _, _ = self.invoke(phase)
                    self.assertEqual(run.returncode, 2, run.stderr)
                    self.assertEqual(self.context.snapshot(), before)
        self.context[STATE_KEY] = good

    def test_real_sdk_snapshot_restore_not_claimed_runner_rollback(self):
        self.valid_validation()
        before = self.context.snapshot()
        self.valid_validation()
        self.context.replace_all(before)
        response = self.valid_validation()
        self.assertEqual(response['round'], 2)

    def test_missing_and_extra_final_artifacts(self):
        for name in ['predictions.csv','method.md','sources.json','research_code.zip']:
            with self.subTest(name=name):
                path=self.submission/name;data=path.read_bytes();path.unlink()
                self.final(False);path.write_bytes(data)
        (self.submission/'extra.txt').write_text('extra')
        self.final(False)

    def test_method_utf8_nonempty_and_limit(self):
        for data in [b'', b' \n\t', b'\xff', b'a'*(64*1024+1)]:
            with self.subTest(size=len(data)):
                (self.submission/'method.md').write_bytes(data)
                self.final(False)

    def test_sources_requires_three_lists(self):
        cases=[[],{}, {'data':[],'models':[]}, {'data':{},'models':[],'software':[]},
               {'data':[],'models':None,'software':[]}, {'data':[],'models':[],'software':'Python'}]
        for obj in cases:
            with self.subTest(obj=obj):
                write_json(self.submission/'sources.json',obj);self.final(False)
        (self.submission/'sources.json').write_text('{"data":[],"models":[],"software":[],"data":[]}')
        self.final(False)

    def test_final_special_files_are_rejected(self):
        for name in ['predictions.csv','method.md','sources.json','research_code.zip']:
            path=self.submission/name;data=path.read_bytes();path.unlink()
            target=self.root/('target-'+name);target.write_bytes(data)
            path.symlink_to(target);self.final(False);path.unlink()
            os.mkfifo(path);self.final(False);path.unlink();path.write_bytes(data)

    def test_zip_path_traversal_absolute_windows_and_duplicates(self):
        for names in [('../evil.py',),('/absolute.py',),('a/../../evil.py',),('a\\evil.py',),('C:/evil.py',),('same.py','same.py')]:
            with self.subTest(names=names):
                (self.submission/'research_code.zip').write_bytes(zip_bytes(names))
                self.final(False)

    def test_zip_symlink_and_special_types(self):
        for mode in [stat.S_IFLNK|0o777, stat.S_IFIFO|0o600, stat.S_IFCHR|0o600, stat.S_IFDIR|0o755]:
            with self.subTest(mode=mode):
                (self.submission/'research_code.zip').write_bytes(zip_bytes(mode=mode))
                self.final(False)

    def test_zip_invalid_empty_count_and_declared_expansion(self):
        for data in [b'not zip', zip_bytes(()), zip_bytes(tuple('x%04d.py'%i for i in range(1025)))]:
            (self.submission/'research_code.zip').write_bytes(data);self.final(False)
        data=bytearray(zip_bytes());central=data.index(b'PK\x01\x02')
        struct.pack_into('<I',data,central+24,512*1024*1024+1)
        (self.submission/'research_code.zip').write_bytes(data);self.final(False)

    def test_zip_corrupt_deflate_is_invalid_not_infrastructure_failure(self):
        data=bytearray(zip_bytes())
        self.assertEqual(struct.unpack_from('<H',data,8)[0],zipfile.ZIP_DEFLATED)
        name_length,extra_length=struct.unpack_from('<HH',data,26)
        data[30+name_length+extra_length]=0x07  # Reserved DEFLATE block type 3.
        (self.submission/'research_code.zip').write_bytes(data)
        self.final(False)

    def test_zip_and_csv_size_caps(self):
        for name, size in [('research_code.zip',64*1024*1024+1),('predictions.csv',2*1024*1024+1)]:
            self.make_submission()
            with (self.submission/name).open('wb') as f:f.truncate(size)
            self.final(False)

    def test_reference_corruption_is_infrastructure_failure(self):
        local=self.root/'corrupt';shutil.copytree(self.fixture,local);self.package=local
        for phase, name in [('validation','validation_labels.csv'),('test','test_labels.csv')]:
            path=local/'verifier/test/data'/name
            original=path.read_bytes();path.write_bytes(original+b'\n')
            run, _, _=self.invoke(phase)
            self.assertEqual(run.returncode,2,run.stderr)
            path.write_bytes(original)

    def test_official_actions_retry_identity_and_terminal_behavior(self):
        case=self
        class LocalSubprocessAdapter:
            calls=0
            def invoke(adapter,kind,directory,request=None):
                adapter.calls+=1
                run,result,_=case.invoke(kind,payload=request)
                if run.returncode:raise RuntimeError(run.stderr)
                return result
        adapter=LocalSubprocessAdapter()
        actions=Actions(adapter,self.root/'actions',{'request_bytes':5*1024*1024})
        envelope={'schema_version':'1.0','request_id':str(uuid.uuid4()),'action':'validation','payload':self.payload()}
        first=actions.handle(envelope)
        self.assertEqual(actions.handle(copy.deepcopy(envelope)),first)
        self.assertEqual(adapter.calls,1)
        self.assertEqual(len(self.context[STATE_KEY]['rounds']),1)
        changed=copy.deepcopy(envelope);changed['payload']=self.payload([1]*5)
        with self.assertRaises(ContractError):actions.handle(changed)
        new=copy.deepcopy(envelope);new['request_id']=str(uuid.uuid4());actions.handle(new)
        self.assertEqual(len(self.context[STATE_KEY]['rounds']),2)
        terminal={'schema_version':'1.0','request_id':str(uuid.uuid4()),'action':'test','payload':None}
        reply=actions.handle(terminal)
        self.assertTrue(reply['terminal'] and reply['accepted'])
        self.assertNotIn('response',reply)
        new['request_id']=str(uuid.uuid4())
        with self.assertRaises(ContractError):actions.handle(new)

    def test_dev_and_final_response_do_not_include_reference_rows(self):
        response=self.valid_validation(self.payload([1]*5))
        text=json.dumps(response)
        for identifier in self.ids['test']+self.ids['validation']:
            self.assertNotIn(identifier,text)
        self.assertNotIn('request_sha256',text)
        result,_=self.final()
        for identifier in self.ids['test']+self.ids['validation']:
            self.assertNotIn(identifier,json.dumps(result))


class ActualInputTests(unittest.TestCase):
    def test_entire_export_allowlists_geometry_and_split_isolation(self):
        endpoint={'reaction_id','atomic_numbers','reactant_positions_angstrom','product_positions_angstrom'}
        train_extra={'forward_barrier_eV','transition_state_positions_angstrom','reactant_energy_eV','product_energy_eV','transition_state_energy_eV'}
        ids={};compositions={};counts={}
        config=json.loads((ARGS.package/'verifier/protocol.json').read_text())
        for split in ('train','validation','test'):
            ids[split]=set();compositions[split]=set();counts[split]=0
            with gzip.open(ARGS.package/'input'/(split+'.jsonl.gz'),'rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    self.assertEqual(set(row),endpoint|train_extra if split=='train' else endpoint)
                    self.assertNotIn(row['reaction_id'],ids[split]);ids[split].add(row['reaction_id']);counts[split]+=1
                    z=row['atomic_numbers'];self.assertGreater(len(z),0)
                    self.assertTrue(all(type(v)is int and v in (1,6,7,8) for v in z))
                    self.assertLessEqual(sum(v!=1 for v in z),7)
                    compositions[split].add(tuple((v,z.count(v)) for v in (1,6,7,8)))
                    keys=['reactant_positions_angstrom','product_positions_angstrom']+(['transition_state_positions_angstrom'] if split=='train' else [])
                    for key in keys:
                        xyz=row[key];self.assertEqual(len(xyz),len(z))
                        self.assertTrue(all(len(v)==3 and all(type(x)in (float,int) and math.isfinite(x) for x in v) for v in xyz))
                    if split=='train':
                        self.assertEqual(row['forward_barrier_eV'],row['transition_state_energy_eV']-row['reactant_energy_eV'])
            if split!='train':
                refids,refvalues=read_labels(ARGS.package/'verifier/test/data'/(split+'_labels.csv'))
                self.assertEqual(set(refids),ids[split]);self.assertEqual(len(refids),counts[split])
                self.assertEqual(config[split]['count'],counts[split])
                self.assertTrue(all(math.isfinite(v) for v in refvalues))
                template=ARGS.package/'input'/('validation_predictions.csv' if split=='validation' else 'predictions.csv')
                with template.open(newline='') as f:
                    reader=csv.DictReader(f);self.assertEqual(reader.fieldnames,COLUMNS);rows=list(reader)
                self.assertEqual({r['reaction_id'] for r in rows},ids[split]);self.assertEqual(len(rows),counts[split])
                self.assertTrue(all(r['forward_barrier_eV']=='' for r in rows))
        self.assertEqual(sum(counts.values()),10073)
        for left,right in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(ids[left]&ids[right]);self.assertFalse(compositions[left]&compositions[right])

    def test_real_reference_barriers_match_hidden_endpoint_energies(self):
        references={split:dict(zip(*read_labels(ARGS.package/'verifier/test/data'/(split+'_labels.csv')))) for split in ('validation','test')}
        seen={split:set() for split in references}
        with gzip.open(ARGS.package/'verifier/test/data/reference_states.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);split='validation' if row['split']=='val' else row['split']
                self.assertIn(split,references);identifier=row['reaction_id'];self.assertNotIn(identifier,seen[split]);seen[split].add(identifier)
                barrier=row['transition_state_energy_eV']-row['reactant_energy_eV']
                self.assertEqual(barrier,row['forward_barrier_eV']);self.assertEqual(barrier,references[split][identifier])
        for split in references:self.assertEqual(seen[split],set(references[split]))


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.records=[]
    def addSuccess(self,test):super().addSuccess(test);self.records.append({'test':test.id(),'status':'passed'})
    def addFailure(self,test,err):super().addFailure(test,err);self.records.append({'test':test.id(),'status':'failed','detail':self._exc_info_to_string(err,test)})
    def addError(self,test,err):super().addError(test,err);self.records.append({'test':test.id(),'status':'error','detail':self._exc_info_to_string(err,test)})
    def addSubTest(self,test,subtest,err):
        super().addSubTest(test,subtest,err)
        if err:self.records.append({'test':subtest.id(),'status':'failed','detail':self._exc_info_to_string(err,test)})


def main():
    global ARGS, VerifyContext, validate_result, Actions, ContractError
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--framework',type=Path,required=True)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--only',help='Optional unittest name substring for targeted debugging')
    p.add_argument('--synthetic-only',action='store_true',help='Use only known five-row fixtures while real data/protocol are not installed')
    ARGS=p.parse_args();ARGS.package=ARGS.package.resolve();ARGS.framework=ARGS.framework.resolve()
    ARGS.report.parent.mkdir(parents=True,exist_ok=True);ARGS.scratch_root=ARGS.report.parent
    sys.path.insert(0,str(ARGS.framework/'src'))
    from science_innovation_exam.verify_context import VerifyContext
    from science_innovation_exam.verifier import validate_result
    from science_innovation_exam.actions import Actions
    from science_innovation_exam.common.errors import ContractError
    def observed_manifest():
        manifest=source_manifest(ARGS.package)
        return [r for r in manifest if r['path'].startswith('verifier/') and r['path'].endswith('.py')] if ARGS.synthetic_only else manifest
    before=observed_manifest()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReactionTests)
    if not ARGS.synthetic_only:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ActualInputTests))
    if ARGS.only:suite=unittest.TestSuite(t for t in suite if ARGS.only in t.id())
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
    after=observed_manifest()
    report={'scope':'Independent scientific/artifact black-box checks using unchanged official SDK in real subprocesses. Official Actions tested through an explicitly local subprocess adapter, not Docker. SDK snapshot restoration tested directly; automatic complete-run rollback not certified.',
            'framework_commit':subprocess.check_output(['git','-C',str(ARGS.framework),'rev-parse','HEAD'],text=True).strip(),
            'task_package':str(ARGS.package),'synthetic_fixture_only':ARGS.synthetic_only,
            'source_hash_scope':'verifier Python files only during construction' if ARGS.synthetic_only else 'all task files',
            'source_files_before':before,'source_files_after':after,
            'source_unchanged':before==after,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'passed':result.wasSuccessful() and before==after,'results':result.records}
    ARGS.report.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
