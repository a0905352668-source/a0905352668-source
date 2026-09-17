"""Tighter evidence is opt-in and facts cannot exclude events."""
import pytest
from live_operator import capture_evidence as evidence
from live_operator import mage_vl_service as service
from live_operator.tests.test_capture_evidence import _synthetic_video,_overlay,_visibility


def test_tight_crop_shrinks_margins_without_changing_frames_or_screen_selection(tmp_path):
    video=_synthetic_video(tmp_path/'tight.avi')
    old=evidence.decode_capture_frames(video,_overlay(),_visibility())[0]
    tight=evidence.decode_capture_frames(video,_overlay(),dict(_visibility(),capture_crop_profile='tight10'))[0]
    assert old.crop_box==(0,0,260,240)
    assert tight.crop_box==(0,0,230,219)
    assert tight.times==old.times and tight.source_frame_indices==old.source_frame_indices
    assert tight.screen_ids==old.screen_ids and len(tight.frames)==30
    explicit=evidence.decode_capture_frames(video,_overlay(),dict(_visibility(),capture_crop_profile='wide25'))[0]
    assert all(a.tobytes()==b.tobytes() for a,b in zip(old.frames,explicit.frames))


def test_unknown_crop_profile_is_rejected(tmp_path):
    video=_synthetic_video(tmp_path/'unknown.avi')
    with pytest.raises(ValueError):
        evidence.decode_capture_frames(video,_overlay(),dict(_visibility(),capture_crop_profile='unknown'))


@pytest.mark.parametrize('kind',['pose','relation'])
def test_fact_prompts_are_registered_and_request_uncertainty_not_exclusion(kind):
    from live_operator.capture_target_diagnostic import TARGET_DIAGNOSTIC_PROMPTS
    variant='target_box_'+kind+'_02'
    assert service.capture_prompt_variant({'capture_prompt_variant':variant})==variant
    text=TARGET_DIAGNOSTIC_PROMPTS[variant]
    assert 'LABEL=UNCERTAIN' in text and 'EXCLUDE' not in text
    assert '绿色' in text and '黄色' in text and '蓝色' in text
