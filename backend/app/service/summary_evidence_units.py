"""正文、Markdown表格和字段记录的证据单元；所有内容保留原文位置。"""
from __future__ import annotations

import re

from service.claimify import sentence_spans

VERSION = 'native-markdown-evidence-units-v2'
_BREAK = re.compile(r'\r?\n|\\{1,4}n')
_LABEL = re.compile(r'^(?:记录\s*\w+|Record\s+[\w-]+)\s*[:：]', re.I)
_CAPTION = re.compile(r'^(?:表\s*\d+|Table\s+\d+)\b', re.I)
_NOTE = re.compile(r'^(?:注\s*[:：]|说明\s*[:：]|Note\s*[:：]|\[\^[^\]]+\])', re.I)
_IDENTITY = re.compile(r'^(?:main id|trial id|record id|registration (?:number|no\.?|id)|试验编号|注册编号|记录编号)$', re.I)
_SECTION = re.compile(r'^(?:Key inclusion & exclusion criteria|Primary outcomes?|Secondary outcomes?|Study design|纳入与排除标准|主要终点|次要终点)$', re.I)


def render_spans(full, spans):
    """只用原文片段和换行组装；不虚构表头，也不把组装结果伪装成连续引用。"""
    return '\n'.join(full[s['start']:s['end']] for s in spans)


def validate_spans(full, piece):
    spans = piece['source_spans']
    if not spans or any(type(s.get('start')) is not int or type(s.get('end')) is not int
                        or not 0 <= s['start'] < s['end'] <= len(full) for s in spans):
        raise ValueError('结构化证据原文位置无效')
    if render_spans(full, spans) != piece['text']:
        raise ValueError('结构化证据与原文片段不一致')
    if spans != piece['fixed_prefix'] + piece['body_rows'] + piece['fixed_suffix']:
        raise ValueError('结构化证据表头、数据行或注释位置不一致')


def pack_structured(full, kind, prefix, rows, suffix, fits, **metadata):
    """完整行/字段打包，每片重复表头/标识及注释；超长单行明确拒绝。"""
    fixed = prefix + suffix
    if not fits(render_spans(full, fixed)):
        raise ValueError('表头、记录标识或注释超出证据预算，未截断')
    groups = []; batch = []
    if metadata.get('table_requires_whole'):
        if not fits(render_spans(full, prefix + rows + suffix)):
            raise ValueError('原生表格存在多级表头或空白单元格，整表超出预算，未猜测拆分或补填关联')
        groups = [rows]
    pending_rows = [] if metadata.get('table_requires_whole') else rows
    for row in pending_rows:
        if fits(render_spans(full, prefix + batch + [row] + suffix)):
            batch.append(row)
        else:
            if batch:
                groups.append(batch)
                batch = []
            if not fits(render_spans(full, prefix + [row] + suffix)):
                if kind == 'record' and row.get('role') in {'field', 'field_fragment'}:
                    groups.append(_long_field(full, prefix, row, suffix, fits, metadata))
                    continue
                raise ValueError('单个表格行或记录字段连同表头超出证据预算，未拆断数据行')
            batch = [row]
    if batch or (not rows and not groups):
        groups.append(batch)
    units = []
    for group in groups:
        if isinstance(group, dict):
            units.extend(group['units'])
            continue
        spans = prefix + group + suffix
        if not spans:
            continue
        units.append({'unit_type': kind, 'quote_kind': 'assembled_source_spans',
                      'start': min(s['start'] for s in spans), 'end': max(s['end'] for s in spans),
                      'text': render_spans(full, spans), 'source_spans': spans,
                      'fixed_prefix': prefix, 'body_rows': group, 'fixed_suffix': suffix, **metadata})
    return units


def _long_field(full, prefix, field, suffix, fits, metadata):
    """过长字段只切字段值，每片重复原字段名和记录标识，不丢尾部。"""
    start, end = field['start'], field['end']; label = []
    if field['role'] == 'field':
        raw = full[start:end]
        match = re.match(r'(?:\|\s*[^|:：]+[:：](?:\s*\|)?|[^:：]+[:：])', raw)
        if not match:
            raise ValueError('过长记录字段无法定位字段名，未猜测切分')
        label = [{'start': start, 'end': start + match.end(), 'role': 'field_label', 'key': field.get('key')}]
        start += match.end()
    fixed_prefix = prefix + label
    if not fits(render_spans(full, fixed_prefix + suffix)):
        raise ValueError('记录字段名和标识超出预算，未截断')
    fragments = []; cursor = start
    while cursor < end:
        def candidate(right):
            return {'start': cursor, 'end': right, 'role': 'field_fragment', 'key': field.get('key')}
        if fits(render_spans(full, fixed_prefix + [candidate(end)] + suffix)):
            right = end
        else:
            low, high = 1, end - cursor
            if not fits(render_spans(full, fixed_prefix + [candidate(cursor + 1)] + suffix)):
                raise ValueError('字段名、标识及单字符值也超出预算，未截断')
            while low < high:
                middle = (low + high + 1) // 2
                if fits(render_spans(full, fixed_prefix + [candidate(cursor + middle)] + suffix)):
                    low = middle
                else:
                    high = middle - 1
            right = cursor + low
            # 尽量沿完整句或词边界；确实过长的无空格字段才按字符片段保留。
            boundaries = [b for _, b in sentence_spans(full[cursor:right])]
            boundaries.extend(m.end() for m in re.finditer(r'\s+', full[cursor:right]))
            usable = [b for b in boundaries if low // 2 <= b < low]
            if usable:
                right = cursor + max(usable)
        fragment = candidate(right)
        spans = fixed_prefix + [fragment] + suffix
        if not fits(render_spans(full, spans)):
            raise ValueError('记录字段片段超预算，未提交NLI')
        fragments.append({'unit_type': 'record', 'quote_kind': 'assembled_source_spans',
                          'start': min(s['start'] for s in spans), 'end': max(s['end'] for s in spans),
                          'text': render_spans(full, spans), 'source_spans': spans,
                          'fixed_prefix': fixed_prefix, 'body_rows': [fragment], 'fixed_suffix': suffix,
                          **metadata, 'field_fragmented': True})
        cursor = right
    return {'units': fragments}


def _lines(full, start, end):
    """真实换行及工具JSON中的转义换行只作识别视图，原文不解码覆盖。"""
    ranges = []; cursor = start
    for m in _BREAK.finditer(full, start, end):
        ranges.append((cursor, m.start())); cursor = m.end()
    ranges.append((cursor, end))
    lines = []
    for left, right in ranges:
        raw_start = left
        while left < right:
            ws = re.match(r'(?:\s|\\{1,4}t)+', full[left:right])
            if not ws:
                break
            left += ws.end()
        while right > left and full[right - 1].isspace():
            right -= 1
        lines.append({'raw_start': raw_start, 'start': left, 'end': right, 'text': full[left:right]})
    return lines


def _span(line, role):
    return {'start': line['start'], 'end': line['end'], 'role': role}


def _cells(line):
    text = line['text']; separators = []; in_code = False
    for i, char in enumerate(text):
        if char == '`':
            in_code = not in_code
        if char == '|' and not in_code and (i == 0 or text[i - 1] != '\\'):
            separators.append(i)
    if not separators:
        return []
    cells = []
    edges = [-1] + separators + [len(text)]
    for left, right in zip(edges, edges[1:]):
        text_left, text_right = left + 1, right
        while text_left < text_right and text[text_left].isspace(): text_left += 1
        while text_right > text_left and text[text_right - 1].isspace(): text_right -= 1
        value = text[text_left:text_right]
        if (left == -1 or right == len(text)) and not value:
            continue
        # 工具JSON的闭合符号是容器，不是表格末列。
        if right == len(text) and re.fullmatch(r'["}\], ]+', value):
            continue
        cells.append({'text': value, 'start': line['start'] + max(left, 0),
                      'end': line['start'] + min(right + 1, len(text))})
    return cells


def _separator(line):
    cells = _cells(line)
    return bool(cells) and all(re.fullmatch(r':?-{3,}:?', c['text']) for c in cells)


def _fields(line):
    cells = _cells(line)
    if cells:
        def key(cell):
            value = cell['text']
            if _SECTION.fullmatch(value): return value.casefold()
            match = re.match(r'([^|:：]{1,80})[:：]', value)
            if match and match.group(1).casefold() not in {'http', 'https'}:
                return match.group(1).strip().casefold()
            return None
        if len(cells) > 1 and key(cells[0]) is None and key(cells[1]) is not None:
            cells = cells[1:]  # 网页说明文字后接字段记录。
        fields = []; i = 0
        while i < len(cells):
            field_key = key(cells[i])
            if field_key is None:
                return []
            j = i + 1
            while j < len(cells) and key(cells[j]) is None:
                j += 1
            fields.append({'start': cells[i]['start'], 'end': cells[j - 1]['end'],
                           'role': 'field', 'key': field_key})
            i = j
        return fields
    match = re.match(r'([^:：\n]{1,80})[:：]\s*\S', line['text'])
    if match and not re.match(r'https?$', match.group(1), re.I):
        return [{**_span(line, 'field'), 'key': match.group(1).strip().casefold()}]
    return []


def _native_table(lines, start):
    """识别原生管道表格；保留堆叠表头、表题和无表头行，不重写原文。"""
    table_start = start
    outside_caption = []
    if _CAPTION.match(lines[start]['text']):
        outside_caption = [_span(lines[start], 'caption')]
        start += 1
    stop = start
    while stop < len(lines) and _cells(lines[stop]):
        stop += 1
    if stop - start < 2:
        return None
    first_separator = next((i for i in range(start, stop) if _separator(lines[i])), None)
    if first_separator is None:
        # 管道字段记录继续走记录逻辑，不把注册字段误当无表头数据表。
        if any(_fields(line) for line in lines[start:stop]):
            return None
        header_end = start
        prefix = outside_caption
        columns = len(_cells(lines[start]))
        header_status = 'absent'
    else:
        if first_separator == start:
            raise ValueError('表格分隔行前缺少原文表头')
        columns = len(_cells(lines[first_separator]))
        header_end = first_separator + 1
        # 原生输出可能把 caption 也表示为一行表头；连续的表头/分隔行
        # 属于同一表，不能把第二组表头读作数据后再丢掉后续行。
        while header_end + 1 < stop and _separator(lines[header_end + 1]):
            header_end += 2
        prefix = list(outside_caption)
        header_lines = [i for i in range(start, header_end) if not _separator(lines[i])]
        for i in range(start, header_end):
            role = 'separator' if _separator(lines[i]) else 'header'
            values = [cell['text'] for cell in _cells(lines[i])]
            if (role == 'header' and len(header_lines) > 1 and i == header_lines[0]
                    and sum(bool(value) for value in values) == 1):
                role = 'caption'
            prefix.append(_span(lines[i], role))
        header_status = 'explicit'
    body_end = header_end
    while body_end < stop:
        if body_end + 1 < stop and _separator(lines[body_end + 1]):
            break  # 后一张紧邻的表格有自己的表头。
        if _separator(lines[body_end]):
            raise ValueError('表格数据区出现无对应表头的分隔行')
        body_end += 1
    for line in lines[start:body_end]:
        if len(_cells(line)) != columns:
            raise ValueError('表格行列数不一致，未猜测列映射')
    rows = [_span(line, 'row') for line in lines[header_end:body_end]]
    headers = [span for span in prefix if span['role'] == 'header']
    # Markdown 的空格不能区分真实空值与合并单元格，也不保证后续文字行
    # 不是次级表头。出现这些信号时保留整表，不推断 header/data 边界。
    header_positions = {span['start'] for span in headers}
    sparse_header = any(not cell['text'] for line in lines[start:header_end]
                        if line['start'] in header_positions for cell in _cells(line))
    leading_text_row = bool(rows and not re.search(r'\d', lines[header_end]['text']))
    if headers and (sparse_header or leading_text_row):
        header_status = 'ambiguous'
    caption_positions = {span['start'] for span in prefix if span['role'] == 'caption'}
    has_empty_cell = any(not cell['text'] for line in lines[start:body_end]
                         if not _separator(line) and line['start'] not in caption_positions
                         for cell in _cells(line))
    notes = []
    consumed = body_end
    cursor = body_end
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor]['text']:
            cursor += 1
        if cursor >= len(lines) or not _NOTE.match(lines[cursor]['text']):
            break
        notes.append(_span(lines[cursor], 'note'))
        consumed = cursor + 1
        cursor += 1
    return {
        'start': table_start, 'stop': consumed, 'prefix': prefix, 'rows': rows, 'notes': notes,
        'header_status': header_status, 'column_count': columns,
        'table_requires_whole': has_empty_cell or header_status == 'ambiguous',
    }


def split_evidence_units(full, start, end, fits):
    """短表整表、长表按行重复表头；记录按重复起始字段/显式记录标识分开。"""
    lines = _lines(full, start, end); result = []; i = 0; plain_start = start

    def flush_plain(stop):
        for left, right in sentence_spans(full[plain_start:stop]):
            result.append({'unit_type': 'sentence', 'start': plain_start + left, 'end': plain_start + right,
                           'text': full[plain_start + left:plain_start + right]})

    fence = ''
    while i < len(lines):
        marker = re.match(r'^(`{3,}|~{3,})', lines[i]['text'])
        if marker:
            if not fence:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = ''
            i += 1
            continue
        if fence:
            i += 1
            continue
        table = _native_table(lines, i)
        if table is not None:
            flush_plain(lines[i]['raw_start'])
            result.extend(pack_structured(
                full, 'table', table['prefix'], table['rows'], table['notes'], fits,
                header_status=table['header_status'], column_count=table['column_count'],
                table_requires_whole=table['table_requires_whole'],
                table_start=lines[i]['start'],
            ))
            j = table['stop']
            plain_start = lines[j]['raw_start'] if j < len(lines) else end
            i = j; continue

        label = _LABEL.match(lines[i]['text'])
        first_fields = _fields(lines[i + 1]) if label and i + 1 < len(lines) else _fields(lines[i])
        # 单独一句“要求：...”仍按正文处理；至少两个连续字段或管道字段才视为记录。
        fields_start = i + 1 if label else i
        is_record = bool(first_fields) and (bool(_cells(lines[fields_start])) or label or
                    (fields_start + 1 < len(lines) and bool(_fields(lines[fields_start + 1]))))
        if is_record:
            groups = []; fields = []; record_label = [_span(lines[i], 'record_label')] if label else []
            j = fields_start
            while j < len(lines):
                if j != fields_start and _LABEL.match(lines[j]['text']):
                    break
                # 不吞并紧接在字段记录之后的标准表格。
                if j + 1 < len(lines) and _separator(lines[j + 1]):
                    break
                current = _fields(lines[j])
                if not current:
                    break
                for field in current:
                    new_identity = _IDENTITY.match(field['key']) and any(_IDENTITY.match(f['key']) for f in fields)
                    registry_start = (field['key'] in {'register', '注册机构', '登记机构'}
                                      and fields and not record_label
                                      and not any(_IDENTITY.match(f['key']) for f in fields))
                    if fields and (field['key'] == fields[0]['key'] or new_identity or registry_start):
                        groups.append((record_label, fields)); fields = []; record_label = []
                    fields.append(field)
                j += 1
            if fields:
                groups.append((record_label, fields))
            flush_plain(lines[i]['raw_start'])
            for record_label, fields in groups:
                identities = [f for f in fields if _IDENTITY.match(f['key'])]
                prefix = record_label + identities
                rows = [f for f in fields if f not in identities]
                result.extend(pack_structured(full, 'record', prefix, rows, [], fits,
                                              identity_missing=not bool(prefix)))
            plain_start = lines[j]['raw_start'] if j < len(lines) else end
            i = j; continue
        i += 1
    flush_plain(end)
    return result


def structured_windows(piece, full, fits):
    """NLI二次装包也沿完整行/字段切分，并重复原表头或记录标识。"""
    validate_spans(full, piece)
    return pack_structured(full, piece['unit_type'], piece['fixed_prefix'], piece['body_rows'],
                           piece['fixed_suffix'], fits,
                           **{k: piece[k] for k in ('identity_missing', 'field_fragmented',
                                                   'header_status', 'column_count',
                                                   'table_requires_whole', 'table_start') if k in piece})
