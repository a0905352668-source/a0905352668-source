"""Offline target-binding diagnostics must never become production exclusions."""
import pytest
from live_operator import mage_vl_service as service


def test_fixed_target_variant_is_accepted_by_capture_request():
    assert service.capture_prompt_variant({'capture_prompt_variant': 'target_pro_none'}) == 'target_pro_none'


def parser():
    fn = getattr(service, 'parse_target_diagnostic', None)
    assert callable(fn), 'target diagnostic parser not implemented'
    return fn


@pytest.mark.parametrize('label', ['CAPTURE_POSSIBLE', 'UNCERTAIN', 'EXCLUDE'])
def test_diagnostic_keeps_three_labels_separate(label):
    result = parser()(f'TARGET=上半部黑衣人物\nPHONE=第7帧举起，竖持\nBLOCK=未知\nLABEL={label}')
    assert result['label'] == label
    assert result['parsed'] is True
    assert result['target'] == '上半部黑衣人物'


@pytest.mark.parametrize('raw', [
    'TARGET=\nPHONE=竖持\nBLOCK=未知\nLABEL=EXCLUDE',
    'TARGET=人\nPHONE=\nBLOCK=未知\nLABEL=EXCLUDE',
    'TARGET=人\nPHONE=竖持\nBLOCK=\nLABEL=EXCLUDE',
    'TARGET=人\nPHONE=竖持\nBLOCK=未知\nLABEL: EXCLUDE',
    'TARGET=人\nPHONE=竖持\nBLOCK=未知\nLABEL=IMPOSSIBLE_FLAT_OR_DOWN',
    'TARGET=人\nPHONE=竖持\nBLOCK=未知',
])
def test_empty_malformed_or_retired_output_is_not_valid_exclusion(raw):
    result = parser()(raw)
    assert result['label'] == 'UNCERTAIN'
    assert result['parsed'] is False


def test_stop_reason_distinguishes_length_limit_from_eos():
    fn = getattr(service, 'diagnostic_stop_reason', None)
    assert callable(fn), 'generation stop audit not implemented'
    assert fn([11,12], 2, [99]) == 'max_new_tokens'
    assert fn([11,99], 2, [99]) == 'eos'
    assert fn([11], 2, [99]) == 'other'


@pytest.mark.parametrize('label,parsed,truncated,want', [
    ('EXCLUDE', True, False, 'UNCERTAIN'),
    ('CAPTURE_POSSIBLE', True, True, 'UNCERTAIN'),
    ('CAPTURE_POSSIBLE', False, False, 'UNCERTAIN'),
    ('CAPTURE_POSSIBLE', True, False, 'CAPTURE_POSSIBLE'),
])
def test_business_mapping_never_adopts_exclusion_or_truncation(label,parsed,truncated,want):
    assert service.target_business_label(dict(label=label,parsed=parsed,truncated=truncated)) == want


@pytest.mark.parametrize('diagnostic_label,ids,want,reason', [
    ('EXCLUDE', [42,99], 'UNCERTAIN', 'eos'),
    ('CAPTURE_POSSIBLE', [42,99], 'CAPTURE_POSSIBLE', 'eos'),
    ('CAPTURE_POSSIBLE', [42]*192, 'UNCERTAIN', 'max_new_tokens'),
])
@pytest.mark.parametrize('variant,markings,crop_profile',[('target_pro_02','clean','wide25'),('target_box_02','person-phone-screens-v1','wide25'),('target_box_pose_02','person-phone-screens-v1','tight10')])
def test_capture_consumer_records_diagnostic_without_exclusion_leak(tmp_path,monkeypatch,diagnostic_label,ids,want,reason,variant,markings,crop_profile):
    # Only heavy external inference/preprocessing are substituted; the real
    # review_capture result, parser, token audit and business mapping execute.
    import contextlib
    import json
    import sys
    import threading
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from live_operator.capture_evidence import CaptureSequence
    torch = SimpleNamespace(inference_mode=contextlib.nullcontext,cuda=SimpleNamespace(empty_cache=lambda:None))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(StoppingCriteria=object,StoppingCriteriaList=list))
    frames = (Image.new('RGB',(448,448)),)*30
    sequence = CaptureSequence('p',frames,tuple(range(30)),tuple(i/6 for i in range(30)),('s',),(0,0,448,448))
    monkeypatch.setattr(service,'decode_capture_frames',lambda *args,**kwargs:(sequence,))
    def preprocess(processor,text,seq,*,audit):
        audit['video_sha256'] = 'actual-file-hash'
        return {'input_ids':np.array([[7,8]]),'image_grid_thw':np.array([[1,28,28]]*30)}
    monkeypatch.setattr(service,'process_capture_video',preprocess)
    class Model:
        device = 'cpu'
        generation_config = SimpleNamespace(eos_token_id=99)
        def generate(self,**kwargs):
            assert kwargs['max_new_tokens']==192 and kwargs['do_sample'] is False
            return np.array([[7,8]+ids])
    reviewer = service.MageVLReviewer.__new__(service.MageVLReviewer)
    reviewer.model = Model()
    reviewer.model_version = 'test-model'
    reviewer._lock = threading.Lock()
    raw = f'TARGET=上半部黑衣人物\nPHONE=第7帧举起\nBLOCK=未知\nLABEL={diagnostic_label}'
    reviewer.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:raw,eos_token_id=99))
    reviewer.capture_chat_texts = {variant:'fixed-chat'}
    overlay = tmp_path/'overlay.json'
    visibility = tmp_path/'visibility.json'
    overlay.write_text('{}')
    visibility.write_text(json.dumps({'capture_prompt_variant':variant,'capture_input_profile':'video5s30','capture_markings_profile':markings,'capture_crop_profile':crop_profile}))
    result = reviewer.review_capture(tmp_path/'clip.mp4',overlay,visibility,threading.Event())
    assert result['label']==want
    assert result['candidate_diagnostics'][0]['label']==diagnostic_label
    assert result['candidate_diagnostics'][0]['stop_reason']==reason
    assert result['candidate_diagnostics'][0]['generated_tokens']==len(ids)
    assert result['candidate_evidence'][0]['model_video_sha256']=='actual-file-hash'
    assert result['candidate_evidence'][0]['markings_profile']==markings
    assert result['candidate_evidence'][0]['crop_profile']==crop_profile
    assert result['candidate_evidence'][0]['visual_grid_thw']==[[1,28,28]]*30
    assert result['candidate_output_parsed']==[True]
