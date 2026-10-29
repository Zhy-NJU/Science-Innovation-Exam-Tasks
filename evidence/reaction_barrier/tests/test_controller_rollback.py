#!/usr/bin/env python3
"""Official controller rollback unit tests with the genuine SQLite SDK.

Only container execution and Linux ownership changes are test doubles.
This is deliberately not described as real Docker/task end-to-end execution.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class ControllerRollbackTests(unittest.TestCase):
    def exercise(self, outcome):
        with tempfile.TemporaryDirectory(prefix='controller-unit-',dir=self.scratch) as tmp:
            root=Path(tmp)
            context=self.SDK.create(root/'context',maximum_bytes=4096)
            context['history']={'rounds':[{'accepted':1}]}
            initial=context.snapshot()
            invocation=root/'invocation'
            observed={'service_wrote_state':False}
            class TaskDouble:
                task_id='infrastructure_fixture'
                def verifier_mounts(self):return []
            class ServiceDouble:
                def run_verifier(service,spec):
                    # Represents a temporary verifier that wrote state before its outcome.
                    context['history']={'rounds':[{'accepted':1},{'accepted':2}]}
                    context['partial_write']='must be rolled back on failure'
                    observed['service_wrote_state']=True
                    if outcome=='exit_nonzero':return 3
                    if outcome=='timeout':raise TimeoutError('simulated container timeout')
                    if outcome=='malformed_json':
                        (invocation/'output/response.json').write_text('{')
                    elif outcome=='nonobject_json':
                        (invocation/'output/response.json').write_text('[]')
                    elif outcome=='success':
                        (invocation/'output/response.json').write_text('{"status":"ok"}')
                    else:raise AssertionError(outcome)
                    return 0
            config=copy.deepcopy(self.DEFAULTS)
            verifier=self.module.Verifier(ServiceDouble(),TaskDouble(),'unused-image',context.root,config,root)
            # Bypass only host chown to UID 65532; no replacement state/rollback code.
            with patch.object(self.module,'writable',lambda path:None):
                if outcome=='success':
                    response=verifier.invoke('validation',invocation,request={'fixture':True})
                    self.assertEqual(response,{'status':'ok'})
                    self.assertEqual(len(context['history']['rounds']),2)
                else:
                    expected={'exit_nonzero':self.ScoringError,'timeout':TimeoutError,
                              'malformed_json':json.JSONDecodeError,'nonobject_json':self.ScoringError}[outcome]
                    with self.assertRaises(expected):
                        verifier.invoke('validation',invocation,request={'fixture':True})
                    self.assertEqual(self.SDK(context.root).snapshot(),initial)
                    self.assertTrue((invocation/'error.json').is_file())
                self.assertTrue(observed['service_wrote_state'])

    def test_nonzero_exit_restores_original_sqlite_state(self):self.exercise('exit_nonzero')
    def test_timeout_restores_original_sqlite_state(self):self.exercise('timeout')
    def test_malformed_json_restores_original_sqlite_state(self):self.exercise('malformed_json')
    def test_nonobject_response_restores_original_sqlite_state(self):self.exercise('nonobject_json')
    def test_successful_invocation_preserves_committed_state(self):self.exercise('success')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();framework=args.framework.resolve()
    sys.path.insert(0,str(framework/'src'))
    import science_innovation_exam.verifier as verifier_module
    from science_innovation_exam.verify_context import VerifyContext
    from science_innovation_exam.config import DEFAULTS
    from science_innovation_exam.common.errors import ScoringError
    args.report.parent.mkdir(parents=True,exist_ok=True)
    ControllerRollbackTests.module=verifier_module
    ControllerRollbackTests.SDK=VerifyContext
    ControllerRollbackTests.DEFAULTS=DEFAULTS
    ControllerRollbackTests.ScoringError=ScoringError
    ControllerRollbackTests.scratch=args.report.parent
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControllerRollbackTests))
    report={'scope':'Official Verifier.invoke controller unit tests with actual SQLite VerifyContext; container service and Linux chown are explicit test doubles. Not real container execution, actual timeout enforcement, or full task integration.',
            'framework_commit':subprocess.check_output(['git','-C',str(framework),'rev-parse','HEAD'],text=True).strip(),
            'verifier_sha256':hashlib.sha256(Path(verifier_module.__file__).read_bytes()).hexdigest(),
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
