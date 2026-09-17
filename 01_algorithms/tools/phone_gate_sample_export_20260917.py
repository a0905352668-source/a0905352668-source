"""Read-only production evidence export; no model calls or event mutations."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import argparse
import cv2
from PIL import Image, ImageDraw

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    sys.path.insert(0, args.package)
    from live_operator.mage_vl_service import (probe_video_metadata, select_candidate_sequences,
        decode_candidate_frames, NATIVE_PROMPT, FOCUS_PROMPT, EARLY_RESCUE_PROMPT, PROMPT_REVISION)
    out = Path(args.output)
    out.mkdir(mode=0o700, exist_ok=False)
    rows = []
    for review in Path(args.pool).rglob('review.json'):
        d = json.loads(review.read_text())
        group = ('negative' if d.get('reason') == 'model_misdetect' and d.get('result') == 'false_positive'
                 else 'positive' if d.get('result') == 'confirmed' and d.get('category') == 'phone_use' else None)
        if group:
            rows.append((group, d.get('reviewed_at', ''), review.parent, d))
    records, counts = [], {'negative': 0, 'positive': 0}
    for group, _, folder, review in sorted(rows, key=lambda r: r[1], reverse=True):
        if counts[group] >= 12:
            continue
        clip, overlay = folder/'event.mp4', folder/'overlay.json'
        if not clip.is_file() or not overlay.is_file():
            continue
        payload = json.loads(overlay.read_text())
        meta = probe_video_metadata(clip)
        seq = select_candidate_sequences(payload, frame_count=20, target_span_seconds=5,
                                        pre_alarm_frames=4, video_fps=meta.fps, video_frame_count=meta.frame_count)
        declared = payload.get('overlay', payload)
        if len(seq) != 1 or declared.get('alarm_track_count') != 1:
            continue
        native = decode_candidate_frames(clip, seq, panel_mode='native')
        focus = decode_candidate_frames(clip, seq, panel_mode='context_focus')
        if not native or not focus or len(native[0][1]) != 20 or len(set(native[0][2])) != 20 or native[0][2] != focus[0][2]:
            continue
        sample = group[:1] + str(counts[group]+1).zfill(2)
        dest = out/sample
        dest.mkdir()
        sheet = Image.new('RGB', (448*3, 448*2+26), 'white')
        ImageDraw.Draw(sheet).text((8,8), sample+' '+review['camera']+' '+review.get('occurred_at',''), fill='black')
        for row, (mode, frames, indices) in enumerate((('native',native[0][1],native[0][2]),('focus',focus[0][1],focus[0][2]))):
            for col, n in enumerate((0,9,19)):
                sheet.paste(frames[n], (col*448,26+row*448))
            for name, subset in ((mode, frames),) + ((('early',frames[:8]),) if mode == 'native' else ()):
                path = dest/(name+'.avi')
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 4, (448,448))
                assert writer.isOpened()
                for frame in subset:
                    import numpy as np
                    writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
                writer.release()
                cap = cv2.VideoCapture(str(path))
                assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(subset)
                cap.release()
        sheet.save(dest/'preview.jpg', quality=95)
        record = dict(sample=sample, manual=review, source=str(folder), indices=list(native[0][2]),
                      videos={name:hashlib.sha256((dest/(name+'.avi')).read_bytes()).hexdigest() for name in ('native','focus','early')})
        records.append(record)
        counts[group] += 1
        print('EXPORTED', sample, review['event_id'], flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(records=records,prompt_revision=PROMPT_REVISION,
        prompts=dict(native=NATIVE_PROMPT,focus=FOCUS_PROMPT,early=EARLY_RESCUE_PROMPT)), ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
