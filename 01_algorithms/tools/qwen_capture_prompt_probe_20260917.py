"""Standalone approved offline prompt A/B. No production package imports or filtering.

Run as root only inside a bounded systemd unit with ExecStopPost starting Mage.
All model inference is unprivileged and bound to loopback. Media stays outside Git.
"""
import sys

# A root import must never create root-owned caches in trusted production trees.
sys.dont_write_bytecode = True

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
import urllib.request


ROOT = Path('/home/zty/YL/JianKong')
STAGE = ROOT / '00_staging/capture_tight_facts_20260916'
MODEL = Path('/opt/jiankong-offline/qwen3.8-27b-q4_k_m-0669b986-20260917')
BUILD = ROOT / '08_envs/llama-cpp-83078fec/build/bin/llama-server'
SERVICE = 'jiankong-mage-vl-54.service'
DESCRIPTIONS = {
    '01': '中央坐在蓝黑色椅子上、背对摄像头、穿条纹上衣的长发人物',
    '02': '上半部坐在灰黑椅子上的短发黑衣人物，不是下方粉色上衣人物',
    '03': '前景站立、黑色短袖胸前有浅色小图案的人物，不是后方绿色上衣人物',
    '06': '前景靠近显示器、穿浅米色长袖、身体前倾的人物，不是隔板后方的人物',
}
COMMON = '''目标：{target}。
绿色TARGET框是事件人物，黄色PHONE框是其已确认的手机，蓝色编号轮廓是附近屏幕。标注仅用于定位。
请观看完整短视频，只描述绿色框内人物，不把其他人的动作算到他身上。
不需要分析手机正面或背面，不猜测不可见镜头朝向，不判断拍摄意图、快门或手机真假。
'''
QUESTIONS = {
    'opportunity': '''只回答：本段中，手机有没有机会指向附近屏幕进行拍摄？
依据手机所处位置、举起或转动的动作、与附近屏幕的位置关系判断，并指出对应视频时间。
有可见机会回答“有机会”；只看到普通操作但没看到明确机会回答“未观察到明确机会”；画面或空间关系不清回答“无法判断”。
“未观察到明确机会”不等于“确定不能拍屏”。不要因为手机显示屏朝向本人就认定镜头背离电脑屏幕。
只输出四行，使用等号：
TARGET=<人物位置和外观>
MOTION=<持机位置和整段动作变化，附时间；不清楚写未知>
OPPORTUNITY=<有机会|未观察到明确机会|无法判断>
BASIS=<可见判断依据及不确定处>
''',
    'behavior': '''先观察整段动作，再把动作事实与是否确定不能拍屏分开记录。
1. 是否全程低头使用手机，还是有抬头、转身等变化？附视频时间；看不清就写未知。
2. 手机是否全程未举起，是否抬高手臂、提高手机、转动或停留？附视频时间；不要把头低着误当成手没抬起。
3. 综合持机位置、动作和附近屏幕位置，是否存在指向屏幕拍摄的机会？
4. 这段画面是否真正证实全程不能拍屏？没有这种可核对证据回答“未证实”，无法判断回答“无法判断”。
全程低头、没有抬高手臂、低位持机只是动作事实，不能单独证明不能拍屏；不要把未看到拍摄动作当作排除证据。
手机显示屏朝向本人也不能单独证明镜头背离电脑屏幕。看不清的空间关系不要编造。
只输出六行，使用等号：
TARGET=<人物位置和外观>
HEAD_DOWN=<整段低头使用是否成立、有无变化及时间；不清楚写未知>
RAISED=<手机与手臂是否举起、有无变化及时间；不清楚写未知>
OPPORTUNITY=<有机会|未观察到明确机会|无法判断>
CANNOT_CAPTURE=<已证实|未证实|无法判断>
BASIS=<机会和不能拍屏分别依据什么；不确定处>
''',
}


def build_request(video, prompt):
    return {
        'model': 'qwen3.8-27b-q4_k_m-0669b986',
        'messages': [{'role': 'user', 'content': [
            {'type': 'input_video', 'input_video': {'data': base64.b64encode(video).decode()}},
            {'type': 'text', 'text': prompt},
        ]}],
        'temperature': 0, 'max_tokens': 192,
        'chat_template_kwargs': {'enable_thinking': False}, 'stream': False,
    }


def parse_response(variant, raw):
    fallback = {'valid': False, 'retained': True}
    fields = (('target', 'motion', 'opportunity', 'basis') if variant == 'opportunity'
              else ('target', 'head_down', 'raised', 'opportunity', 'cannot_capture', 'basis'))
    lines = raw.strip().splitlines()
    if len(lines) != len(fields):
        return fallback
    values = {}
    for field, line in zip(fields, lines):
        prefix = field.upper() + '='
        if not line.startswith(prefix) or not line[len(prefix):].strip():
            return fallback
        values[field] = line[len(prefix):].strip()
    if values['opportunity'] not in {'有机会', '未观察到明确机会', '无法判断'}:
        return fallback
    if variant == 'behavior' and values['cannot_capture'] not in {'已证实', '未证实', '无法判断'}:
        return fallback
    return dict(values, valid=True, retained=True)


def server_command():
    return ['sudo', '-u', 'zty', str(BUILD), '-m', str(MODEL/'Qwen3.8-27B-Q4_K_M.gguf'),
            '--mmproj', str(MODEL/'mmproj-Qwen3.8-27B-BF16.gguf'), '--host', '127.0.0.1',
            '--port', '18879', '--alias', 'qwen3.8-27b-q4_k_m-0669b986', '--fit', 'off',
            '--gpu-layers', '22', '--ctx-size', '16384', '--parallel', '1', '--threads', '8',
            '--threads-batch', '8', '--batch-size', '256', '--ubatch-size', '128',
            '--image-max-tokens', '256', '--video-fps', '0', '--video-timestamp-interval', '168',
            '--no-warmup', '--log-verbosity', '5']


def http(url, data=None, timeout=10):
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def completion_with_guard(url, data, check_foreground, poll_seconds=2):
    """Supervise one HTTP generation; guard errors unwind into server cleanup.

    A daemon worker must not delay the root runner's finally/ExecStopPost when
    foreground health fails. Terminating Qwen closes its outstanding request.
    This creates no additional model concurrency: the caller awaits every job.
    """
    outcome = queue.Queue(maxsize=1)

    def request_worker():
        try:
            outcome.put((True, http(url, data, timeout=320)))
        except BaseException as error:
            outcome.put((False, error))

    threading.Thread(target=request_worker, daemon=True).start()
    while True:
        try:
            success, result = outcome.get(timeout=poll_seconds)
        except queue.Empty:
            check_foreground()
            continue
        if not success:
            raise result
        return result


def guard():
    status = http('http://192.168.104.53:8767/api/status')
    result = {'state': status.get('state'), 'online': sum(c.get('status') == 'online' for c in status.get('cameras', [])),
              'fps': status.get('aggregate_fps'), 'alerts': status.get('alerts')}
    assert result['state'] == 'running' and result['online'] == 8 and result['fps'] >= 76, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-actions-[a-z0-9-]+', args.output_name), 'invalid scoped output name'
    assert os.geteuid() == 0, 'service lifecycle requires root'
    assert BUILD.is_file() and (MODEL/'download_verified.json').is_file()
    proof = {p['sample']: p for p in json.loads((STAGE/'tight_video_proof.json').read_text())}
    verified = json.loads((MODEL/'download_verified.json').read_text())
    assert len(verified) == 2 and all((MODEL/v['file']).stat().st_size == v['bytes'] for v in verified)
    jobs = []
    # Same clip A/B adjacent; any prefix-cache benefit is recorded, not assumed.
    for sample in ['02', '03', '01', '06']:
        video = STAGE/('sample_'+sample+'_tight.avi')
        assert hashlib.sha256(video.read_bytes()).hexdigest() == proof[sample]['model_video_sha256']
        assert proof[sample]['frame_count'] == 30
        for variant in ['opportunity', 'behavior']:
            prompt = COMMON.format(target=DESCRIPTIONS[sample]) + QUESTIONS[variant]
            jobs.append({'sample': sample, 'variant': variant, 'prompt': prompt,
                         'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                         'video_sha256': proof[sample]['model_video_sha256'], 'screen_ids': proof[sample]['screen_ids']})
    before = guard()
    assert subprocess.run(['systemctl', 'is-active', '--quiet', SERVICE]).returncode == 0
    out = STAGE/args.output_name
    out.mkdir(mode=0o700, exist_ok=False)
    subprocess.run(['chown', 'zty:zty', str(out)], check=True)

    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

    def terminated(signum, frame):
        raise KeyboardInterrupt('bounded experiment terminated')

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    command = server_command()
    write('manifest.json', {'jobs': jobs, 'command': command, 'before': before,
                           'download_verified': verified, 'online_filtering': False, 'formal_accuracy_computed': False,
                           'maximum_experiment_seconds': 1440, 'systemd_hard_limit_seconds': 1800,
                           'no_production_package_imports': True, 'bytecode_write_disabled': sys.dont_write_bytecode,
                           'request_guard_period_seconds': 2})
    records, server = [], None
    start = time.monotonic()
    log = (out/'llama-server.log').open('w')
    try:
        subprocess.run(['systemctl', 'stop', SERVICE], check=True, timeout=45)
        server = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        load_start = time.monotonic()
        while True:
            assert server.poll() is None, 'Qwen startup exited'
            try:
                if http('http://127.0.0.1:18879/health').get('status') == 'ok':
                    break
            except Exception:
                pass
            assert time.monotonic()-load_start < 180, 'Qwen startup deadline'
            guard()
            time.sleep(2)
        print('QWEN_READY', round(time.monotonic()-load_start, 2), flush=True)
        write('props.json', http('http://127.0.0.1:18879/props'))
        stop = None
        for job in jobs:
            if time.monotonic()-start > 1440-330:
                stop = 'bounded time budget; unstarted jobs remain untested'
                break
            record = dict(job, before=guard())
            video = (STAGE/('sample_'+job['sample']+'_tight.avi')).read_bytes()
            assert hashlib.sha256(video).hexdigest() == job['video_sha256']
            offset, request_start = log.tell(), time.monotonic()
            response = completion_with_guard('http://127.0.0.1:18879/v1/chat/completions',
                                             build_request(video, job['prompt']), guard)
            raw = response['choices'][0]['message'].get('content', '')
            record.update(response=response, elapsed_seconds=round(time.monotonic()-request_start, 2),
                          diagnostics=parse_response(job['variant'], raw),
                          finish_reason=response['choices'][0].get('finish_reason'),
                          truncated=response['choices'][0].get('finish_reason') == 'length')
            records.append(record)
            write('results.json', {'records': records, 'online_filtering': False, 'formal_accuracy_computed': False})
            with (out/'llama-server.log').open('rb') as audit:
                audit.seek(offset)
                audit_text = audit.read().decode('utf-8', errors='replace')
            record['native_video_audit'] = {
                'decoded_frames': len(re.findall(r'lazy callback returned bitmap with dimensions', audit_text)),
                'temporal_frame_merges': len(re.findall(r'merging 2 frames at part index', audit_text)),
                'nonconsecutive_position_warnings': len(re.findall(r'non-consecutive token position', audit_text)),
            }
            write('results.json', {'records': records, 'online_filtering': False, 'formal_accuracy_computed': False})
            assert record['native_video_audit']['decoded_frames'] == 30, 'native video frame audit mismatch'
            record['after'] = guard()
            write('results.json', {'records': records, 'online_filtering': False, 'formal_accuracy_computed': False})
            print('QWEN_RESULT', job['sample'], job['variant'], json.dumps(record['diagnostics'], ensure_ascii=False), flush=True)
        write('summary.json', {'planned': len(jobs), 'completed': len(records), 'stop_reason': stop,
                               'foreground_end': guard(), 'online_filtering': False, 'formal_accuracy_computed': False})
    except BaseException as error:
        write('experiment-error.json', {'type': type(error).__name__, 'error': str(error), 'completed': len(records)})
        raise
    finally:
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)
        log.close()
        subprocess.run(['systemctl', 'start', SERVICE], check=True, timeout=60)
        subprocess.run(['chown', '-R', 'zty:zty', str(out)], check=True)
        print('MAGE_RESTART_REQUESTED', flush=True)


if __name__ == '__main__':
    main()
