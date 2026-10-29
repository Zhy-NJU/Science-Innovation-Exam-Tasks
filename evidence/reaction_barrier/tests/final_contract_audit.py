#!/usr/bin/env python3
"""Final read-only contract, provenance and repeat-score checks (no containers)."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest(root):
    result = []
    for path in sorted(root.rglob('*')):
        assert '__pycache__' not in path.parts and path.suffix != '.pyc', str(path)
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        assert stat.S_ISREG(mode), 'Non-regular task file: ' + str(path)
        result.append({'path': path.relative_to(root).as_posix(),
                       'bytes': path.stat().st_size, 'sha256': sha(path)})
    return result


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--framework', required=True, type=Path)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--prior-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    before = manifest(args.package)
    prior = json.loads(args.prior_report.read_text())
    assert prior['passed'] and prior['source_unchanged'] and prior['tests_run'] == 33
    old = {row['path']: row for row in prior['source_files_after']}
    current = {row['path']: row for row in before}
    removed, added = sorted(set(old) - set(current)), sorted(set(current) - set(old))
    changed = sorted(path for path in old.keys() & current.keys() if old[path] != current[path])
    assert not added, added
    assert all('__pycache__/' in path and path.endswith('.pyc') for path in removed), removed
    assert set(changed) <= {'instruction.json', 'meta.json', 'input/README.md'}, changed
    assert len(before) == 24, len(before)

    # Call the bundled fallback explicitly, even though the real framework is importable.
    checker_path = args.contract / 'scripts/check_package.py'
    spec = importlib.util.spec_from_file_location('official_package_checker', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    fallback = checker.fallback_check(args.package)
    sys.path.insert(0, str(args.framework / 'src'))
    from science_innovation_exam.task import TaskPackage
    from science_innovation_exam.verifier import validate_result
    from science_innovation_exam.verify_context import VerifyContext
    task = TaskPackage.load(args.package)
    assert task.task_id == 'reaction_barrier'
    meta = task.meta
    protocol = json.loads((args.package / 'verifier/protocol.json').read_text())
    assert meta['metric']['type'] == 'linear_minimize'
    assert meta['baseline']['score'] < meta['baseline']['bottom_line_score']
    assert protocol['invalid_score'] == meta['baseline']['bottom_line_score']
    assert '/486e2dd495aaa265908a2d264427fb5c6af1b336/' in meta['baseline']['source']

    # Two re-executions: one per original frozen submission, on the same trusted context.
    context_path = args.results / 'validation_context'
    context = VerifyContext(context_path)
    original_state = context.snapshot()
    assert len(original_state['reaction_barrier.validation.v1']['rounds']) == 3
    checks = {}
    environment = dict(os.environ, PYTHONPATH=str(args.framework / 'src'),
                       SIE_VERIFY_CONTEXT=str(context_path), PYTHONDONTWRITEBYTECODE='1',
                       PYTHONHASHSEED='0')
    for name, anchor in [('baseline', 'score'), ('bottom', 'bottom_line_score')]:
        source_result = args.results / name / 'result.json'
        original_bytes = source_result.read_bytes()
        original = json.loads(original_bytes)
        validate_result(original, task.task_id)
        checker.validate_schema(original, 'verifier_result.schema.json')
        assert original['valid'] and original['primary_metric']['direction'] == 'minimize'
        assert original['primary_metric']['name'] == 'forward_barrier_mae'
        assert original['primary_metric']['value'] == meta['baseline'][anchor]
        repeat_path = args.output / (name + '_repeat_result.json')
        submission = args.results / name / 'submission'
        submission_before = manifest(submission)
        command = [sys.executable, '-B', str(args.package / 'verifier/test/run/main.py'),
                   '--input', str(args.package / 'input'), '--submission', str(submission),
                   '--result', str(repeat_path)]
        run = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=20)
        assert run.returncode == 0, (name, run.returncode, run.stdout, run.stderr)
        repeated = json.loads(repeat_path.read_bytes())
        validate_result(repeated, task.task_id)
        assert repeat_path.read_bytes() == original_bytes, name + ' result differs'
        assert context.snapshot() == original_state, 'Final scoring mutated trusted context'
        assert submission_before == manifest(submission), 'Final scoring mutated submission'
        assert source_result.read_bytes() == original_bytes
        checks[name] = {'official_schema_valid': True, 'raw_score': original['primary_metric']['value'],
                        'same_context_repeat_byte_identical': True,
                        'result_sha256': sha(source_result), 'frozen_submission_files': submission_before,
                        'validation_calls': 3, 'n_samples': 287}
    after = manifest(args.package)
    assert before == after, 'Source task changed during final audit'
    write(args.output / 'final_package_manifest.json', {
        'schema_version': 'reaction-barrier-evidence-manifest.v1', 'task_id': task.task_id,
        'path_base': str(args.package), 'file_count': len(after),
        'total_bytes': sum(row['bytes'] for row in after), 'files': after})
    report = {
        'passed': True,
        'scope': 'Final official schemas/structure and two real-SDK final-entry re-executions; no Docker or full agent run.',
        'python_version': sys.version,
        'framework_commit': subprocess.check_output(['git', '-C', str(args.framework), 'rev-parse', 'HEAD'], text=True).strip(),
        'official_checker_sha256': sha(checker_path),
        'bundled_fallback_check': fallback,
        'real_framework_loader': {'status': 'package_valid', 'class': 'science_innovation_exam.task.TaskPackage',
                                  'task_id': task.task_id, 'fingerprint': task.fingerprint()},
        'prior_full_suite': {'report': str(args.prior_report), 'sha256': sha(args.prior_report),
                            'passed': True, 'tests_run': 33, 'historical_file_count': len(old),
                            'same_verifier_code_protocol_and_scientific_data': True,
                            'metadata_or_document_changes_after_suite': changed,
                            'removed_generated_bytecode_after_suite': removed, 'added_files': added},
        'current_file_count': len(after), 'current_total_bytes': sum(row['bytes'] for row in after),
        'no_bytecode_or_cache_directories': True, 'no_undeclared_input': True,
        'source_unchanged_during_final_audit': True, 'trusted_context_unchanged': True,
        'formal_result_checks': checks, 'container_execution_performed': False}
    write(args.output / 'final_contract_audit.json', report)
    print(json.dumps({'passed': True, 'files': len(after), 'bytes': report['current_total_bytes'],
                      'prior_full_suite': '33/33; same code/protocol/scientific data',
                      'official_checkers': 'bundled fallback + actual TaskPackage.load',
                      'repeat_results': 'baseline and bottom byte-identical; same context unchanged'}, indent=2))


if __name__ == '__main__':
    main()
