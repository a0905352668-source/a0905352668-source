"""Catch misplaced, fabricated, or accidentally default-on offline markings."""
import numpy as np
import pytest
from PIL import Image
from live_operator import capture_evidence as evidence
from live_operator import mage_vl_service as service
from live_operator.tests.test_capture_evidence import _synthetic_video, _overlay, _visibility


def renderer():
    from live_operator import capture_markings
    return capture_markings.draw_capture_markings


def test_markings_map_rectangles_and_polygon_through_letterbox_without_changing_source():
    original = Image.new('RGB',(448,448),(20,30,40))
    result = renderer()(original,crop_box=(100,50,500,250),person_box=(200,100,300,200),phone_boxes=((270,140,280,150),),screen_polygons={'s1':((350,80),(400,80),(400,130),(350,130))})
    # 400x200 -> 448x224, y origin 112. Hand-derived positions.
    assert result.getpixel((112,168))==(0,255,0)
    assert result.getpixel((190,213))==(255,255,0)
    assert result.getpixel((280,146))==(0,128,255)
    assert result.getpixel((0,0))==(20,30,40)
    assert original.getpixel((112,168))==(20,30,40)


def test_markings_do_not_fabricate_a_missing_phone_box():
    result = renderer()(Image.new('RGB',(448,448)),crop_box=(0,0,448,448),person_box=(50,50,100,100),phone_boxes=(),screen_polygons={})
    assert not np.any(np.all(np.asarray(result)==[255,255,0],axis=2))


def test_decode_markings_keep_sampling_crop_and_only_accepted_target_phone(tmp_path):
    video = _synthetic_video(tmp_path/'marked.avi')
    overlay = _overlay()
    for entry in overlay['bbox_timeline']:
        entry['phone_boxes'].append({'box':[10,210,25,230],'accepted':False})
    clean = evidence.decode_capture_frames(video,overlay,_visibility())[0]
    visibility = dict(_visibility(),capture_markings_profile='person-phone-screens-v1')
    marked = evidence.decode_capture_frames(video,overlay,visibility)[0]
    assert marked.source_frame_indices==clean.source_frame_indices
    assert marked.times==clean.times and marked.crop_box==clean.crop_box
    assert marked.screen_ids==clean.screen_ids and marked.track_id==clean.track_id
    image = np.asarray(marked.frames[0])
    assert np.any(np.all(image==[0,255,0],axis=2))
    assert np.any(np.all(image==[255,255,0],axis=2))
    assert np.any(np.all(image==[0,128,255],axis=2))
    # Rejected lower-left phone would map near (21,379); leave it unchanged.
    assert marked.frames[0].getpixel((21,379))==clean.frames[0].getpixel((21,379))
    explicit_clean = evidence.decode_capture_frames(video,overlay,dict(_visibility(),capture_markings_profile='clean'))[0]
    assert all(a.tobytes()==b.tobytes() for a,b in zip(clean.frames,explicit_clean.frames))


def test_unknown_markings_are_rejected_instead_of_silently_used(tmp_path):
    video = _synthetic_video(tmp_path/'unknown.avi')
    with pytest.raises(ValueError):
        evidence.decode_capture_frames(video,_overlay(),dict(_visibility(),capture_markings_profile='unknown'))


def test_malformed_phone_list_does_not_crash_or_create_phone_marks(tmp_path):
    video = _synthetic_video(tmp_path/'malformed.avi')
    overlay = _overlay()
    for entry in overlay['bbox_timeline']:
        entry['phone_boxes']=1
    sequence = evidence.decode_capture_frames(video,overlay,dict(_visibility(),capture_markings_profile='person-phone-screens-v1'))[0]
    assert len(sequence.frames)==30
    assert not np.any(np.all(np.asarray(sequence.frames[0])==[255,255,0],axis=2))


def test_box_prompt_consumes_same_target_cue_and_can_be_used_by_both_arms():
    from live_operator.capture_target_diagnostic import TARGET_DIAGNOSTIC_PROMPTS
    assert service.capture_prompt_variant({'capture_prompt_variant':'target_box_02'})=='target_box_02'
    text = TARGET_DIAGNOSTIC_PROMPTS['target_box_02']
    assert text.startswith(TARGET_DIAGNOSTIC_PROMPTS['target_pro_02'])
    assert '标注只用于定位' in text


def test_markings_cannot_enter_old_exclusion_capture_path(tmp_path,monkeypatch):
    import json
    import sys
    import threading
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(StoppingCriteria=object,StoppingCriteriaList=list))
    overlay=tmp_path/'overlay.json'
    visibility=tmp_path/'visibility.json'
    overlay.write_text('{}')
    visibility.write_text(json.dumps({'capture_prompt_variant':'exclusion','capture_input_profile':'video5s30','capture_markings_profile':'person-phone-screens-v1'}))
    reviewer=service.MageVLReviewer.__new__(service.MageVLReviewer)
    reviewer.model_version='test-model'
    result=reviewer.review_capture(tmp_path/'clip.mp4',overlay,visibility,threading.Event())
    assert result['label']=='UNCERTAIN' and result['evidence_complete'] is False
    assert result['candidate_count']==0 and result['candidate_outputs']==[]
