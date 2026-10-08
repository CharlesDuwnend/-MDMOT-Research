#!/usr/bin/env python3
"""Verify a UUID-selected CUDA device before an MDMT training entry point.

Set CUDA_VISIBLE_DEVICES to one complete GPU UUID before importing torch. Numeric
CUDA ordinals need not equal the physical indices printed by nvidia-smi.
"""
import argparse
import csv
import ctypes
import json
import os
import runpy
import subprocess
import sys
import uuid


def inventory():
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,pci.bus_id',
        '--format=csv,noheader,nounits'], text=True)
    return [dict(index=int(r[0]), uuid=r[1].strip(), name=r[2].strip(),
                 memory_mib=int(r[3]), pci_bus_id=r[4].strip())
            for r in csv.reader(raw.splitlines())]


def select_allowed(rows, requested):
    if not requested.startswith('GPU-'):
        raise RuntimeError('A complete GPU UUID is required; numeric ordinals are forbidden.')
    matches = [r for r in rows if r['uuid'] == requested]
    if len(matches) != 1:
        raise RuntimeError('Requested GPU UUID is not uniquely present in nvidia-smi.')
    row = matches[0]
    if row['index'] == 2 or row['memory_mib'] <= 4096:
        raise RuntimeError('Physical GPU2 and GPUs with at most 4GB are forbidden.')
    return row


def cuda_driver_uuid():
    # The CUDA driver observes the same visibility mapping as PyTorch. Querying
    # its UUID avoids trusting an environment label or a reported model name.
    lib = ctypes.CDLL('libcuda.so.1')
    device = ctypes.c_int()
    value = (ctypes.c_ubyte * 16)()
    def checked(rc, op):
        if rc != 0:
            raise RuntimeError('%s failed with CUDA error %d' % (op, rc))
    checked(lib.cuInit(0), 'cuInit')
    checked(lib.cuDeviceGet(ctypes.byref(device), 0), 'cuDeviceGet')
    checked(lib.cuDeviceGetUuid(ctypes.byref(value), device), 'cuDeviceGetUuid')
    return 'GPU-' + str(uuid.UUID(bytes=bytes(value)))


def verify_device(requested):
    selected = select_allowed(inventory(), requested)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != requested:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must equal the complete selected UUID.')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device must be visible.')
    actual_uuid = cuda_driver_uuid()
    prop = torch.cuda.get_device_properties(0)
    actual_mib = prop.total_memory / 2**20
    if actual_uuid != requested or prop.name != selected['name']:
        raise RuntimeError('CUDA UUID/name does not match the selected physical GPU.')
    if actual_mib <= 4096 or abs(actual_mib - selected['memory_mib']) > 1024:
        raise RuntimeError('CUDA memory does not match the permitted physical GPU.')
    return dict(status='VERIFIED_PHYSICAL_GPU', selected=selected,
                cuda_uuid=actual_uuid, cuda_name=prop.name,
                cuda_total_memory_bytes=prop.total_memory,
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uuid', required=True)
    parser.add_argument('script', nargs=argparse.REMAINDER,
                        help='Optional: -- training_script.py [arguments]. Runs after verification in this process.')
    args = parser.parse_args()
    # Reject prohibited devices before any CUDA initialization.
    select_allowed(inventory(), args.uuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.uuid
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    print(json.dumps(verify_device(args.uuid), indent=2), flush=True)
    script = args.script[1:] if args.script[:1] == ['--'] else args.script
    if script:
        sys.argv = script
        sys.path.insert(0, os.path.dirname(os.path.abspath(script[0])))
        runpy.run_path(script[0], run_name='__main__')


if __name__ == '__main__':
    main()
