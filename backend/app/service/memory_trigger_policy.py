"""长期记忆的轻量入队信号；只识别用户直接表达，不调用模型。"""

import re
from typing import Any, Optional, Sequence


_PREFIX = r"(?:(?:请你?|麻烦你?|帮我|务必|一定要|另外|还有)[，,\s]*)*"
_REMEMBER = re.compile(
    rf"^{_PREFIX}(?:记住|记下(?:来)?|记一下|牢记)(?!是什么意思|是什么|这个词).+"
    r"|^(?:please\s+)?(?:remember|keep in mind|make a note of)\b.+",
    re.IGNORECASE,
)
_PREFERENCE = re.compile(
    r"^我(?:通常|一直|一贯|长期|平时|更|比较|最|个人)?(?:喜欢|偏好|倾向于|习惯|不喜欢).+"
    r"|^我的(?:偏好|习惯)(?:是|为|[：:]).+"
    r"|^(?:以后|今后|往后|从现在(?:开始|起)|后续每次|接下来每次).{0,80}"
    r"(?:回答|回复|报告|输出|沟通|称呼|默认|一律|都用|使用|采用).*"
    r"|^i\s+(?:(?:always|usually)\s+)?(?:prefer|like|dislike)\b.+"
    r"|^(?:from now on|in future|for future (?:answers|responses))\b.+",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    rf"^{_PREFIX}(?:更正|纠正|修正)(?:一下)?[：:,，\s]*.+"
    r"|^(?:不对[：:,，\s]*)?(?:不是.{1,60}[，,]\s*(?:而)?是|我(?:刚才|之前|上次)说错了).+"
    rf"|^{_PREFIX}(?:(?:把|将).{{0,40}}|(?:我(?:的)?|我们(?:的)?|本次|这次|研究|输出){{0,2}})"
    r"(?:范围|时间|年份|预算|目标|要求|格式|语言|名字|姓名|地址|邮箱|负责人|截止日期|偏好)"
    r".{0,30}(?:改为|改成|调整为|更新为|换成).+"
    r"|^(?:correction\s*:|i meant\b|actually[,]?\s+my\b).+",
    re.IGNORECASE,
)
_TRANSIENT = re.compile(
    r"这次|本次|今天|今晚|暂时|仅本轮|只在本轮|\b(?:today|tonight|this time|for this (?:answer|report))\b",
    re.IGNORECASE,
)
_NEGATIVE_REQUEST = re.compile(
    rf"^{_PREFIX}(?:不要|不用|不必|别)(?:再)?(?:记|保存|存入)"
    r"|^(?:please\s+)?(?:do not|don't|never)\s+(?:remember|save|store)\b",
    re.IGNORECASE,
)


def _statements(content: str):
    # 引用/代码中的“记住”不是用户对系统的直接要求；未闭合代码块也排除。
    content = re.sub(r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)", "", content)
    content = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith(">"))
    content = re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"|`[^`\n]*`', "", content)
    for statement in re.split(r"(?<=[。！？!?；;\n])|(?<=\.)\s+", content):
        statement = statement.strip()
        if statement and not _NEGATIVE_REQUEST.search(statement):
            yield statement


def priority_memory_trigger(messages: Sequence[Any]) -> Optional[str]:
    """返回可审计的及时入队原因；不是事实正确性或长期保存价值的验证器。"""
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "user":
            continue
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        for statement in _statements(str(content or "")):
            if _REMEMBER.search(statement):
                return "explicit_remember"
            if statement.endswith(("?", "？")):
                continue
            if _CORRECTION.search(statement):
                return "important_correction"
            if _PREFERENCE.search(statement) and not _TRANSIENT.search(statement):
                return "stable_preference"
    return None
