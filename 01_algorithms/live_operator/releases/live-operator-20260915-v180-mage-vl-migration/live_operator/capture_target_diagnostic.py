"""Fixed offline-only Pro target-binding experiment; no exclusion authority."""
from typing import Any, Sequence

TARGET_DESCRIPTIONS = {
    'none': '未提供额外定位',
    '01': '中央坐在蓝黑色椅子上、背对摄像头、穿条纹上衣的长发人物',
    '02': '上半部坐在灰黑椅子上的短发黑衣人物，不是下方粉色上衣人物',
    '03': '前景站立、黑色短袖胸前有浅色小图案的人物，不是后方绿色上衣人物',
    '06': '前景靠近显示器、穿浅米色长袖、身体前倾的人物，不是隔板后方的人物',
}
TARGET_TEMPLATE = '''目标：{target}。
已确认目标使用真手机，不再判断手机真假。

只判断本段内手机是否可能拍到任一相关屏幕的任一部分，
不判断快门、意图或是否正在录像。

先定位目标，再描述手机可见姿态和手机到屏幕的遮挡。
监控视线被遮挡，不等于手机视线被遮挡；不要猜测不可见镜头方向。

CAPTURE_POSSIBLE：可见位置和姿态支持至少一次拍屏机会。
UNCERTAIN：人物、镜头朝向、空间关系或时间/屏幕覆盖不清。
EXCLUDE：可核对证据足以证明本段全程，所有相关屏幕对所有可能镜头均不可拍。

低位、横竖持、通话状、显示屏明暗、未见拍摄动作，
均不能单独支持EXCLUDE。存在未知、冲突或任何机会，不得排除。

只输出四行：
TARGET=<位置和外观，或未知>
PHONE=<帧号及可见姿态/变化，或未知>
BLOCK=<遮挡物、被挡屏幕及手机视线依据，或未知>
LABEL=<CAPTURE_POSSIBLE|UNCERTAIN|EXCLUDE>
'''
TARGET_DIAGNOSTIC_PROMPTS = {
    'target_pro_' + key: TARGET_TEMPLATE.format(target=value)
    for key, value in TARGET_DESCRIPTIONS.items()
}
TARGET_MAX_NEW_TOKENS = 192


def parse_target_diagnostic(raw: str) -> dict[str, Any]:
    fallback = {'label': 'UNCERTAIN', 'parsed': False, 'target': '', 'phone': '', 'block': ''}
    lines = raw.strip().splitlines()
    prefixes = ('TARGET=', 'PHONE=', 'BLOCK=', 'LABEL=')
    if len(lines) != 4 or any(not line.startswith(prefix) for line, prefix in zip(lines, prefixes)):
        return fallback
    values = [line[len(prefix):].strip() for line, prefix in zip(lines, prefixes)]
    if not all(values) or values[-1] not in {'CAPTURE_POSSIBLE', 'UNCERTAIN', 'EXCLUDE'}:
        return fallback
    return dict(zip(('target', 'phone', 'block', 'label'), values), parsed=True)


def diagnostic_stop_reason(ids: Sequence[int], cap: int, eos_ids: Sequence[int]) -> str:
    if ids and ids[-1] in eos_ids:
        return 'eos'
    if len(ids) >= cap:
        return 'max_new_tokens'
    return 'other'


def target_business_label(diagnostic: dict[str, Any]) -> str:
    # EXCLUDE is only a recorded hypothesis, never a business exclusion.
    if diagnostic.get('parsed') and not diagnostic.get('truncated') and diagnostic.get('label') == 'CAPTURE_POSSIBLE':
        return 'CAPTURE_POSSIBLE'
    return 'UNCERTAIN'
