#!/usr/bin/env python3
"""Local subprocess tests of the unchanged public VerifyContext SDK.

This is genuine SDK execution, not a substitute SDK or Docker/runner test.
Run with --framework pointing to the public framework checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RealSDKTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='real-sdk-', dir=self.scratch_root)
        self.root = Path(self.tmp.name)
        self.context = self.Context.create(self.root / 'context', maximum_bytes=4096)

    def tearDown(self):
        self.tmp.cleanup()

    def child(self, code, context=True):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(self.framework / 'src')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env.pop('SIE_VERIFY_CONTEXT', None)
        if context:
            env['SIE_VERIFY_CONTEXT'] = str(self.context.root)
        return subprocess.run([sys.executable, '-B', '-c', code], env=env,
                              capture_output=True, text=True, timeout=20)

    def test_current_and_cross_process_persistence(self):
        code = 'from science_innovation_exam.verify_context import VerifyContext; c=VerifyContext.current(); n=c.get("accepted_count",0)+1; c["accepted_count"]=n; print(n)'
        for expected in (1, 2):
            p = self.child(code)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(int(p.stdout), expected)
            self.assertEqual(self.Context(self.context.root)['accepted_count'], expected)

    def test_missing_context_fails_closed(self):
        p = self.child('from science_innovation_exam.verify_context import VerifyContext; VerifyContext.current()', context=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('SIE_VERIFY_CONTEXT is not set', p.stderr)

    def test_nonfinite_and_over_limit_values_leave_state_unchanged(self):
        self.context['existing'] = {'count': 3}
        before = self.context.snapshot()
        for value in (float('nan'), float('inf'), -float('inf'), 'x' * 5000):
            with self.assertRaises(self.ContractError):
                self.context['rejected'] = value
            self.assertEqual(self.context.snapshot(), before)

    def test_log_is_diagnostic_not_persistent_state(self):
        p = self.child('from science_innovation_exam.verify_context import VerifyContext; c=VerifyContext.current(); c.log({"phase":"audit","note":"real SDK diagnostic"})')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout), {'phase': 'audit', 'note': 'real SDK diagnostic'})
        self.assertEqual(self.context.snapshot(), {})

    def test_real_sdk_snapshot_restore_and_experiment_isolation(self):
        self.context['count'] = 2
        before = self.context.snapshot()
        self.context['count'] = 19
        self.context.replace_all(before)
        self.assertEqual(self.Context(self.context.root).snapshot(), {'count': 2})
        other = self.Context.create(self.root / 'other', maximum_bytes=4096)
        self.assertEqual(other.snapshot(), {})

    def test_symlink_root_rejected(self):
        link = self.root / 'linked-context'
        link.symlink_to(self.context.root, target_is_directory=True)
        with self.assertRaises(self.SecurityBoundaryError):
            self.Context(link)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    framework = args.framework.resolve()
    sys.path.insert(0, str(framework / 'src'))
    import science_innovation_exam.verify_context as sdk
    from science_innovation_exam.common.errors import ContractError, SecurityBoundaryError
    assert Path(sdk.__file__).resolve() == framework / 'src/science_innovation_exam/verify_context.py'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    RealSDKTests.framework = framework
    RealSDKTests.scratch_root = args.report.parent
    RealSDKTests.Context = sdk.VerifyContext
    RealSDKTests.ContractError = ContractError
    RealSDKTests.SecurityBoundaryError = SecurityBoundaryError
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RealSDKTests))
    report = {
        'scope': 'Unchanged official VerifyContext SDK in local Python subprocesses. No mock SDK; no Docker, complete runner rollback or scientific package integration is claimed.',
        'framework_repository': 'https://github.com/PrismaX-Team/Science-Innovation-Exam-Framework',
        'framework_commit': subprocess.check_output(['git', '-C', str(framework), 'rev-parse', 'HEAD'], text=True).strip(),
        'sdk_file': 'src/science_innovation_exam/verify_context.py',
        'sdk_sha256': hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest(),
        'python_version': sys.version.split()[0],
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'passed': result.wasSuccessful(),
    }
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
