"""Signed classification-only baseline. Never changes event state or reviews."""
import sys
sys.dont_write_bytecode = True
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package',required=True)
    p.add_argument('--stage',required=True)
    a = p.parse_args()
    sys.path.insert(0,a.package)
    from live_operator.vlm_review import VLMReviewClient,VLMReviewConfig,VLMReviewError,evidence_request_id,VLM_EVIDENCE_REVISION
    root=Path('/media/boshi/Data/00_active_projects/JianKong')
    stage=Path(a.stage)
    manifest=json.loads((stage/'selected.json').read_text())
    config=VLMReviewConfig(endpoint='https://192.168.104.54:8879/v1/review',
        shared_secret_file=root/'02_configs/runtime/vlm_review/shared_secret',
        tls_ca_file=root/'02_configs/runtime/vlm_review/tls-ca-192.168.104.54.crt',
        expected_model_version='mage-vl-awq-v20260809-1f7f5266fa4e-phone-use-prompt-v2',
        expected_prompt_revision=manifest['prompt_revision'],timeout_seconds=180)
    client=VLMReviewClient(config)
    client.check_health()
    records=[]
    for row in manifest['records']:
        with urllib.request.urlopen('http://127.0.0.1:8767/api/status',timeout=8) as h:
            status=json.load(h)
        assert status['state']=='running' and status['aggregate_fps']>=76 and len(status['cameras'])==8 and not status.get('alerts') and all(c['status']=='online' for c in status['cameras'])
        folder=Path(row['source'])
        clip,overlay=folder/'event.mp4',folder/'overlay.json'
        eid=row['manual']['event_id']
        request_id=evidence_request_id(eid,clip,overlay,model_version=config.expected_model_version,
            prompt_revision=config.expected_prompt_revision,evidence_revision=VLM_EVIDENCE_REVISION)
        start=time.monotonic()
        for attempt in range(90):
            try:
                result=client.review(eid,request_id,clip,overlay)
                break
            except VLMReviewError as exc:
                if 'HTTP 503' not in str(exc):
                    raise
                time.sleep(2)
        else:
            raise RuntimeError('Mage busy for 180 seconds; no baseline result')
        record=dict(sample=row['sample'],group=row['group'],result=asdict(result),
                    elapsed_seconds=round(time.monotonic()-start,2),fps_before=status['aggregate_fps'])
        records.append(record)
        (stage/'mage-baseline.json').write_text(json.dumps(dict(records=records,production_filtering=False),ensure_ascii=False,indent=2))
        print('MAGE_RESULT',row['sample'],result.label,record['elapsed_seconds'],flush=True)

if __name__=='__main__':
    main()
