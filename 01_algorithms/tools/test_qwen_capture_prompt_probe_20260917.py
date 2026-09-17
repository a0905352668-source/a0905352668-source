"""Contracts for a standalone offline probe, not model accuracy tests."""
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest


SCRIPT = Path(__file__).with_name('qwen_capture_prompt_probe_20260917.py')


def load_probe():
    assert SCRIPT.is_file(), 'standalone probe implementation is missing'
    spec = importlib.util.spec_from_file_location('offline_probe', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uploads_one_actual_video_and_never_an_image_list():
    probe = load_probe()
    request = probe.build_request(b'actual video bytes', 'fixed prompt')
    assert request['messages'][0]['content'] == [
        {'type': 'input_video', 'input_video': {'data': 'YWN0dWFsIHZpZGVvIGJ5dGVz'}},
        {'type': 'text', 'text': 'fixed prompt'},
    ]
    assert request['max_tokens'] == 192
    assert request['chat_template_kwargs'] == {'enable_thinking': False}


def test_action_facts_and_impossibility_are_preserved_independently():
    probe = load_probe()
    raw = ('TARGET=前景黑衣人物\nHEAD_DOWN=全程低头，0-4.88秒\n'
           'RAISED=全程未举起，0-4.88秒\nOPPORTUNITY=无法判断\n'
           'CANNOT_CAPTURE=未证实\nBASIS=镜头和空间关系不清')
    parsed = probe.parse_response('behavior', raw)
    assert parsed['valid'] is True
    assert parsed['head_down'] == '全程低头，0-4.88秒'
    assert parsed['raised'] == '全程未举起，0-4.88秒'
    assert parsed['cannot_capture'] == '未证实'
    assert parsed['retained'] is True


def test_even_an_impossibility_claim_has_no_exclusion_authority():
    probe = load_probe()
    parsed = probe.parse_response('behavior',
        'TARGET=事件人\nHEAD_DOWN=全程\nRAISED=未举起\n'
        'OPPORTUNITY=未观察到明确机会\nCANNOT_CAPTURE=已证实\nBASIS=低头')
    assert parsed['valid'] is True
    assert parsed['cannot_capture'] == '已证实'
    assert parsed['retained'] is True


def test_malformed_and_empty_output_falls_back_to_retention():
    probe = load_probe()
    for raw in ['OPPORTUNITY: EXCLUDE', 'TARGET=\nMOTION=无\nOPPORTUNITY=有机会\nBASIS=举机',
                'TARGET=人\nMOTION=举起\nOPPORTUNITY=EXCLUDE\nBASIS=屏幕朝本人']:
        parsed = probe.parse_response('opportunity', raw)
        assert parsed['valid'] is False
        assert parsed['retained'] is True


def test_only_loopback_qwen_is_launched_as_unprivileged_user():
    probe = load_probe()
    command = probe.server_command()
    assert command[:3] == ['sudo', '-u', 'zty']
    assert command[command.index('--host') + 1] == '127.0.0.1'
    assert command[command.index('--port') + 1] == '18879'
    assert command[command.index('--video-fps') + 1] == '0'
    assert command[command.index('--gpu-layers') + 1] == '22'


def test_guard_failure_interrupts_wait_for_a_real_inflight_http_request():
    probe = load_probe()
    assert hasattr(probe, 'completion_with_guard'), 'during-request foreground supervision is missing'
    release = threading.Event()
    request_started = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            request_started.set()
            release.wait(2)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def failed_guard():
        assert request_started.wait(1), 'HTTP request never reached the test server'
        raise RuntimeError('foreground degraded')

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match='foreground degraded'):
            probe.completion_with_guard('http://127.0.0.1:'+str(server.server_port), {}, failed_guard, poll_seconds=0.01)
        assert not release.is_set(), 'guard was deferred until after completion'
        assert time.monotonic()-started < 1, 'guard failure waited for HTTP timeout'
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_healthy_guard_allows_the_original_http_response_through():
    probe = load_probe()
    release = threading.Event()
    checks = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            release.wait(2)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true,"original":"preserved"}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def healthy_guard():
        checks.append(True)
        release.set()

    try:
        response = probe.completion_with_guard('http://127.0.0.1:'+str(server.server_port), {},
                                               healthy_guard, poll_seconds=0.01)
        assert checks, 'foreground was never checked during the pending request'
        assert response == {'ok': True, 'original': 'preserved'}
    finally:
        release.set()
        server.shutdown()
        server.server_close()
