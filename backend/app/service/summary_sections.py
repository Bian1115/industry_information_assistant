"""新摘要用三栏目；读取旧栏目时保留原位置，合并视图才迁移待办。"""

SECTIONS = ("SESSION INTENT", "SUMMARY", "ARTIFACTS")
LEGACY_SECTIONS = (*SECTIONS, "NEXT STEPS")


def section_names(parts):
    return LEGACY_SECTIONS if "NEXT STEPS" in parts else SECTIONS


def valid_sections(names, *, allow_legacy=True):
    return tuple(names) == SECTIONS or (allow_legacy and tuple(names) == LEGACY_SECTIONS)


def current_section(name):
    return "SUMMARY" if name == "NEXT STEPS" else name


def render_prose(parts):
    """不改旧摘要或审计位置；只在下一次生成输入中合并已有待办正文。"""
    current = {name: parts[name] for name in SECTIONS}
    pending = parts.get("NEXT STEPS", "None")
    if pending != "None":
        current["SUMMARY"] = (current["SUMMARY"] + "\n\n" + pending
                              if current["SUMMARY"] != "None" else pending)
    return "\n\n".join("## " + name + "\n" + current[name] for name in SECTIONS)
