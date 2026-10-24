"""原生 Markdown 的输出预算边界；不解析 HTML 或重新渲染正文。"""
import re


_TABLE_LINE = re.compile(r"^\|.*\|$")
_NOTE = re.compile(r"^(?:注|说明|Note)\s*[:：]", re.I)


def limit_body_preserving_tables(text, max_chars):
    """预算落在原生表格中时延至表尾和紧随的注释，不截断单元格或多级表头。"""
    if len(text) <= max_chars:
        return text
    # 源表以净化HTML保留rowspan/colspan；预算不能把这种新输出截在单元格内。
    for table in re.finditer(r"<table\b[^>]*>.*?</table\s*>", text, re.I | re.S):
        if table.start() <= max_chars < table.end():
            fence = ""
            for line in text[:table.start()].splitlines():
                marker = re.match(r"^(`{3,}|~{3,})", line.strip())
                if marker:
                    fence = "" if fence == marker[1][0] else marker[1][0] if not fence else fence
            if not fence:
                end = table.end()
                while note := re.match(r"\s*\n\s*(?:注|说明|Note)\s*[:：][^\n]*", text[end:], re.I):
                    end += note.end()
                return text[:end].rstrip()
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    i = 0
    fence = ""
    while i < len(lines):
        marker = re.match(r"^(`{3,}|~{3,})", lines[i].strip())
        if marker:
            if not fence:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = ""
            i += 1
            continue
        if fence:
            i += 1
            continue
        if not _TABLE_LINE.fullmatch(lines[i].strip()):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and _TABLE_LINE.fullmatch(lines[j].strip()):
            j += 1
        if j - i < 2:
            i = j
            continue
        table_end = offsets[j]
        cursor = j
        while cursor < len(lines):
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor >= len(lines) or not _NOTE.match(lines[cursor].strip()):
                break
            table_end = offsets[cursor + 1]
            cursor += 1
        if offsets[i] <= max_chars < table_end:
            return text[:table_end].rstrip()
        i = j
    return text[:max_chars].rstrip()
