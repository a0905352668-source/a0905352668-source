"""Optional offline localization marks; never evidence of camera visibility."""
from typing import Any, Mapping, Sequence

CAPTURE_MARKINGS_PROFILES = ('clean', 'person-phone-screens-v1')


def capture_markings_profile(visibility: Mapping[str, Any]) -> str:
    profile = visibility.get('capture_markings_profile', 'clean')
    if not isinstance(profile, str) or profile not in CAPTURE_MARKINGS_PROFILES:
        raise ValueError('unknown offline capture markings profile')
    return profile


def draw_capture_markings(frame: Any, *, crop_box: tuple[int,int,int,int],
                          person_box: tuple[int,int,int,int],
                          phone_boxes: Sequence[tuple[int,int,int,int]],
                          screen_polygons: Mapping[str,Sequence[tuple[float,float]]]) -> Any:
    from PIL import ImageDraw
    result = frame.copy()
    left, top, right, bottom = crop_box
    scale = min(frame.width/(right-left), frame.height/(bottom-top))
    width, height = round((right-left)*scale), round((bottom-top)*scale)
    origin_x, origin_y = (frame.width-width)//2, (frame.height-height)//2
    draw = ImageDraw.Draw(result)

    def point(x: float, y: float) -> tuple[int,int]:
        return (max(origin_x,min(origin_x+width-1,round((x-left)*width/(right-left))+origin_x)),
                max(origin_y,min(origin_y+height-1,round((y-top)*height/(bottom-top))+origin_y)))

    def rectangle(box: Sequence[int], label: str, color: tuple[int,int,int]) -> None:
        a, b = point(box[0],box[1]), point(box[2],box[3])
        if b[0]<=a[0] or b[1]<=a[1]:
            return
        draw.rectangle((*a,*b),outline=color,width=2)
        draw.text((a[0],max(origin_y,a[1]-12)),label,fill=color)

    for screen_id, polygon in screen_polygons.items():
        points = [point(x,y) for x,y in polygon]
        draw.line([*points,points[0]],fill=(0,128,255),width=2)
        draw.text((min(x for x,y in points),max(origin_y,min(y for x,y in points)-12)),screen_id,fill=(0,128,255))
    rectangle(person_box,'TARGET',(0,255,0))
    for box in phone_boxes:
        rectangle(box,'PHONE',(255,255,0))
    return result
