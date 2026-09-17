"""Approved 8-event phone-use gate comparison, never production filtering.

Requires external systemd RuntimeMaxSec and ExecStopPost restarting Mage.
Uses only the standalone lifecycle helper, never trusted production imports.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import qwen_capture_prompt_probe_20260917 as helper

F, K, U = 'FILTER_FALSE_POSITIVE', 'KEEP_NON_CALL_PHONE_USE', 'UNCERTAIN'

class ForegroundAbort(BaseException):
    """Cannot be swallowed by startup's transient HTTP Exception handling."""

def guard():
    result = helper.guard()
    assert not result['alerts'], result
    return result

def parse_label(raw, finish_reason='stop'):
    if finish_reason != 'stop' or re.fullmatch(r'LABEL=(FILTER_FALSE_POSITIVE|KEEP_NON_CALL_PHONE_USE|UNCERTAIN)', raw.strip()) is None:
        return U
    return raw.strip().split('=')[1]

def aggregate(labels):
    if labels and labels[0] != F:
        return labels[0]
    if len(labels) > 1 and labels[1] != F:
        return labels[1]
    return labels[2] if len(labels) == 3 else U

def build_request(video, prompt):
    request = helper.build_request(video, prompt)
    request['max_tokens'] = 24
    return request

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True)
    args = p.parse_args()
    stage = Path(args.stage)
    assert os.geteuid() == 0 and stage.is_absolute()
    verified = json.loads((helper.MODEL/'download_verified.json').read_text())
    assert len(verified) == 2 and all((helper.MODEL/v['file']).stat().st_size == v['bytes'] for v in verified)
    assert hashlib.sha256(helper.BUILD.read_bytes()).hexdigest() == '45dd78b118d27ece11c70168f7602a287fa3c1faf5e4bb0673458f743701f3c4'
    manifest = json.loads((stage/'selected.json').read_text())
    jobs = manifest['records']
    assert len(jobs) == 8 and sum(j['group']=='nonphone' for j in jobs) == 4
    for j in jobs:
        for mode in ('native','focus','early'):
            assert hashlib.sha256((stage/j['sample']/(mode+'.avi')).read_bytes()).hexdigest() == j['videos'][mode]
    before = guard()
    assert subprocess.run(['systemctl','is-active','--quiet',helper.SERVICE]).returncode == 0
    out = stage/'qwen-results'
    out.mkdir(mode=0o700, exist_ok=False)
    subprocess.run(['chown','zty:zty',str(out)],check=True)
    def write(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    def terminated(signum,frame):
        raise KeyboardInterrupt('bounded experiment terminated')
    signal.signal(signal.SIGTERM,terminated)
    signal.signal(signal.SIGINT,terminated)
    watch_stop = threading.Event()
    def foreground_failed(signum,frame):
        raise ForegroundAbort('foreground lifecycle supervisor failed')
    signal.signal(signal.SIGUSR1,foreground_failed)
    def supervise():
        while not watch_stop.wait(2):
            try:
                guard()
            except BaseException as exc:
                write('foreground-failure.json',dict(type=type(exc).__name__,error=repr(exc)))
                print('FOREGROUND_FAILURE',repr(exc),flush=True)
                if not watch_stop.is_set():
                    os.kill(os.getpid(),signal.SIGUSR1)
                return
    threading.Thread(target=supervise,daemon=True).start()
    records, server = [], None
    command = helper.server_command()
    write('manifest.json',dict(input=manifest,command=command,before=before,max_tokens=24,
                              production_filtering=False,time_limit_seconds=1200))
    log = (out/'llama-server.log').open('w')
    start = time.monotonic()
    try:
        subprocess.run(['systemctl','stop',helper.SERVICE],check=True,timeout=45)
        server = subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
        while True:
            assert server.poll() is None and time.monotonic()-start < 180, 'Qwen startup deadline'
            try:
                if helper.http('http://127.0.0.1:18879/health').get('status') == 'ok':
                    break
            except Exception:
                pass
            guard()
            time.sleep(2)
        write('props.json',helper.http('http://127.0.0.1:18879/props'))
        print('QWEN_READY',flush=True)
        for j in jobs:
            row = dict(sample=j['sample'],group=j['group'],passes=[])
            for mode in ('native','focus','early'):
                assert time.monotonic()-start < 1200-330, 'bounded time budget'
                frames = 8 if mode == 'early' else 20
                prompt = manifest['prompts'][mode]
                offset, t = log.tell(), time.monotonic()
                response = helper.completion_with_guard('http://127.0.0.1:18879/v1/chat/completions',
                    build_request((stage/j['sample']/(mode+'.avi')).read_bytes(),prompt),guard)
                choice = response['choices'][0]
                raw = choice['message'].get('content','')
                label = parse_label(raw,choice.get('finish_reason'))
                with (out/'llama-server.log').open('rb') as audit:
                    audit.seek(offset)
                    text = audit.read().decode(errors='replace')
                decoded = len(re.findall(r'lazy callback returned bitmap with dimensions',text))
                pass_row = dict(mode=mode,raw=raw,response=response,label=label,
                    elapsed_seconds=round(time.monotonic()-t,2),decoded_frames=decoded,
                    expected_frames=frames,nonconsecutive_position_warnings=text.count('non-consecutive token position'),
                    strict_valid=choice.get('finish_reason')=='stop' and raw.strip()=='LABEL='+label,
                    after=guard())
                row['passes'].append(pass_row)
                write('inflight.json',row)
                assert decoded == frames, 'native frame audit mismatch'
                print('PASS',j['sample'],mode,label,pass_row['elapsed_seconds'],repr(raw),flush=True)
                if label != F:
                    break
            row['label'] = aggregate([r['label'] for r in row['passes']])
            row['retained'] = row['label'] != F
            records.append(row)
            write('results.json',dict(records=records,planned=8,completed=len(records),production_filtering=False))
        write('summary.json',dict(planned=8,completed=len(records),foreground_end=guard()))
    except BaseException as exc:
        write('error.json',dict(type=type(exc).__name__,error=str(exc),completed=len(records)))
        raise
    finally:
        watch_stop.set()
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        try:
            if server and server.poll() is None:
                try:
                    os.killpg(server.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(server.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    server.wait(timeout=5)
            log.close()
        finally:
            subprocess.run(['systemctl','start',helper.SERVICE],check=True,timeout=60)
            subprocess.run(['chown','-R','zty:zty',str(out)],check=True)
            print('MAGE_RESTART_REQUESTED',flush=True)

if __name__ == '__main__':
    main()
