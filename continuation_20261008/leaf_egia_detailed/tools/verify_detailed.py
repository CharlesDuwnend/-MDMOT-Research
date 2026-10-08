"""Reconstruct the actual detector/ReID frame ledger, independently of flags."""
import hashlib
import json


def verify_input_ledger(report, sequence, frames):
    assert report['sequence'] == sequence and report['frames_seen'] == frames
    assert report['input_hash_schema'] == 'leaf-online-input-v1'
    assert report['input_frames_hashed'] == frames
    records = report['input_records']
    assert len(records) == frames
    rolling = hashlib.sha256(b'leaf-online-input-v1\0')
    for frame, record in enumerate(records, 1):
        assert record['frame'] == frame and record['sequence'] == sequence
        detector, embedding = record['detector_shape'], record['embedding_shape']
        assert len(detector) == len(embedding) == 2 and detector[1] == 7
        assert detector[0] == embedding[0] and embedding[1] > 0
        for key in ['detector_sha256', 'embedding_sha256']:
            assert len(record[key]) == 64 and len(bytes.fromhex(record[key])) == 32
        payload = {k: v for k, v in record.items()
                   if k not in ('frame_input_sha256', 'rolling_input_sha256')}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                               separators=(',', ':')).encode()).hexdigest()
        assert digest == record['frame_input_sha256']
        rolling.update(bytes.fromhex(digest))
        assert rolling.hexdigest() == record['rolling_input_sha256']
    assert rolling.hexdigest() == report['input_stream_sha256']
    return records
