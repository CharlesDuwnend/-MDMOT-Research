#!/usr/bin/env python3
"""GPU policy regression checks; prohibited devices never initialize CUDA."""
import contextlib
import io
import os
import sys
import types
import unittest
from unittest.mock import patch

import verified_gpu as gpu

ROWS = [dict(index=3, uuid='GPU-permitted', name='A100', memory_mib=40960),
        dict(index=2, uuid='GPU-display', name='Display', memory_mib=4096),
        dict(index=1, uuid='GPU-small', name='Small', memory_mib=4096),
        dict(index=2, uuid='GPU-big-but-forbidden', name='A100', memory_mib=40960)]


class Policy(unittest.TestCase):
    def test_uuid_selection_is_physical(self):
        self.assertEqual(gpu.select_allowed(ROWS, 'GPU-permitted')['index'], 3)

    def test_numeric_ordinal_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'UUID'):
            gpu.select_allowed(ROWS, '3')

    def test_index_two_rejected_even_when_large(self):
        with self.assertRaises(RuntimeError):
            gpu.select_allowed(ROWS, 'GPU-big-but-forbidden')

    def test_4gb_rejected_at_any_index(self):
        for uuid in ('GPU-small', 'GPU-display'):
            with self.assertRaises(RuntimeError):
                gpu.select_allowed(ROWS, uuid)

    def test_unknown_or_duplicate_uuid_rejected(self):
        for rows, uuid in ((ROWS, 'GPU-unknown'), (ROWS + [ROWS[0]], 'GPU-permitted')):
            with self.assertRaises(RuntimeError):
                gpu.select_allowed(rows, uuid)

    def test_bad_cli_selection_never_queries_cuda(self):
        with patch.object(gpu, 'inventory', return_value=ROWS), \
             patch.object(gpu, 'verify_device') as verify, \
             patch.object(sys, 'argv', ['verified_gpu.py','--uuid','GPU-display']):
            with self.assertRaises(RuntimeError):
                gpu.main()
            verify.assert_not_called()

    def test_visibility_mismatch_rejected_before_torch(self):
        with patch.object(gpu, 'inventory', return_value=ROWS), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'3'}), \
             patch.object(gpu, 'cuda_driver_uuid') as driver:
            with self.assertRaisesRegex(RuntimeError, 'CUDA_VISIBLE_DEVICES'):
                gpu.verify_device('GPU-permitted')
            driver.assert_not_called()

    def test_runtime_uuid_name_and_memory_must_match(self):
        for uuid, name, memory in (('GPU-display','A100',40960),
                                   ('GPU-permitted','Display',40960),
                                   ('GPU-permitted','A100',4096)):
            prop = types.SimpleNamespace(name=name, total_memory=memory*2**20)
            fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
                is_available=lambda:True, device_count=lambda:1,
                get_device_properties=lambda n:prop))
            with patch.object(gpu,'inventory',return_value=ROWS), \
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-permitted'}), \
                 patch.dict(sys.modules,{'torch':fake}), \
                 patch.object(gpu,'cuda_driver_uuid',return_value=uuid):
                with self.assertRaises(RuntimeError):
                    gpu.verify_device('GPU-permitted')

    def test_script_runs_only_after_verification_with_selected_uuid(self):
        events=[]
        def verify(uuid):
            events.append(('verify',os.environ['CUDA_VISIBLE_DEVICES']))
            return {'status':'test'}
        def run(path,run_name):
            events.append(('script',path,list(sys.argv)))
        with patch.object(gpu,'inventory',return_value=ROWS), \
             patch.object(gpu,'verify_device',side_effect=verify), \
             patch.object(gpu.runpy,'run_path',side_effect=run), \
             patch.object(sys,'argv',['verified_gpu.py','--uuid','GPU-permitted','--','example.py','--steps','1']), \
             patch.object(sys,'path',list(sys.path)), patch.dict(os.environ,{}), \
             contextlib.redirect_stdout(io.StringIO()):
            gpu.main()
        self.assertEqual(events,[('verify','GPU-permitted'),('script','example.py',['example.py','--steps','1'])])


if __name__=='__main__':
    unittest.main()
