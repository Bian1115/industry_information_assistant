"""依据页面自身语义结构补足多帖、列表和数据表；不读取评测标签。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from html import escape
from math import ceil
import re
import unicodedata

from bs4 import BeautifulSoup
from markdownify import markdownify
from service.web_document import CHALLENGE_HEADING, ERROR_PAGE


_NOISE = re.compile(r"(?:^|[\s_-])(?:nav|navbar|navigation|footer|breadcrumb|sidebar|advert|advertisement|"
                    r"sponsored|related|recommendations|cookie|share-tools|social-links|postprofile|"
                    r"post-controls|signature|modal|files-panel)(?:$|[\s_-])", re.I)
_POSTS = (".cooked, .message-body, .postbody, .post_body, #forumposts .post, "
          ".postarea .post, .post_wrapper .post, .s-prose, [itemprop='articleBody']")
_WORDS = re.compile(r"[\u3400-\u9fff]|[^\W_]+", re.UNICODE)


def _plain(node):
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _normalized(text):
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold())


def _tokens(text):
    return Counter(_WORDS.findall(unicodedata.normalize("NFKC", text).casefold()))


def _f1(a, b):
    x, y = _tokens(a), _tokens(b)
    shared = sum((x & y).values())
    return 2 * shared / max(1, sum(x.values()) + sum(y.values()))


def _excluded(node, allow_form=False):
    for parent in [node, *node.parents]:
        if not getattr(parent, "name", None):
            continue
        if parent.name in {"nav", "header", "footer", "script", "style"} or (parent.name == "form" and not allow_form):
            return True
        attrs = " ".join([str(parent.get("id", "")), " ".join(parent.get("class") or []), str(parent.get("role", ""))])
        if _NOISE.search(attrs):
            return True
        if parent.has_attr("hidden") or parent.get("aria-hidden") == "true" or re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", parent.get("style", ""), re.I):
            return True
    return False


def _clean_scope(node):
    root = deepcopy(node)
    for child in list(root.find_all(["script", "style", "nav", "header", "footer", "aside", "form", "button", "input", "select", "svg"])):
        if child.parent is not None:
            child.decompose()
    for child in list(root.find_all(True)):
        if child.parent is not None and _excluded(child):
            child.decompose()
    for child in list(root.select("dialog, [role=dialog], [role=menu], [role=listbox], .dropdown-menu, .select-menu, details-menu")):
        if child.parent is not None:
            child.decompose()
    return root


def _scope_markdown(node):
    root = _clean_scope(node)
    return markdownify(str(root), heading_style="ATX", strip=["a", "img"], wrap=False).strip()


def select_structured_body(soup, baseline):
    """只在可确认的多帖或主区卡片页面扩大范围，文章沿用已有主体选择。"""
    posts = [p for p in soup.select(_POSTS) if not _excluded(p, allow_form=True) and len(_plain(p)) >= 60]
    # 嵌套选择器优先保留帖子正文，不把外层用户档案重新加回来。
    chosen_ids = {id(p) for p in posts}
    posts = [p for p in posts if not any(id(a) in chosen_ids for a in p.find_all(True))]
    if len(posts) >= 2:
        text = "\n\n".join(_scope_markdown(p) for p in posts)
        if len(text) > len(baseline) * 1.2:
            title = soup.find("h1")
            if title:
                text = "# " + _plain(title) + "\n\n" + text
            return text, "discussion_posts"
    if posts:
        # 单帖页面的用户档案、主题导航并不是要补回的列表。
        return baseline, "article"
    articles = [n for n in soup.find_all("article") if not _excluded(n) and len(_plain(n)) >= 100]
    if len(articles) == 1 and len(baseline) >= len(_plain(articles[0])) * .5:
        return baseline, "article"
    mains = [n for n in soup.select("main, [role=main], #main-content, #main_content, #main_content_section")
             if not _excluded(n) and len(_plain(n)) >= 200]
    if not mains:
        return baseline, "article"
    main = max(mains, key=lambda n: len(_plain(n)))
    clean = _clean_scope(main)
    main_chars = len(_plain(clean))
    if len(baseline) >= 1000 and clean.find("table") is not None and _pipe_blocks(baseline):
        return baseline, "article"
    cards = len(clean.find_all(["li", "article"]))
    product = bool(soup.select("[itemtype*='Product'], [itemprop=price], [itemprop=sku]"))
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        if re.search(r'"@type"\s*:\s*(?:"Product"|\[[^\]]*"Product")', script.get_text()):
            product = True
            break
    if product and len(baseline) >= 2000:
        return baseline, "article"
    headings = len(clean.find_all(["h2", "h3", "h4"]))
    if ((cards >= 5 and headings >= 3) or product) and len(baseline) < main_chars * .55:
        articles = [a for a in clean.find_all("article") if len(_plain(a)) >= 80]
        if len(articles) >= 3 and sum(len(_plain(a)) for a in articles) >= main_chars * .3:
            candidate = "\n\n".join(_scope_markdown(a) for a in articles)
        else:
            candidate = _scope_markdown(clean)
        # 大规模目录不用于替换已有长文章；最多扩大到可审查的主区。
        if len(candidate) >= max(200, len(baseline) * 1.35) and len(candidate) <= 150000:
            return candidate, "product_main" if product else "listing_main"
    return baseline, "article"


def _source_tables(soup):
    tables = []
    seen = set()
    for table in soup.find_all("table"):
        if table.find("table") is not None or _excluded(table):
            continue
        rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
        matrix = [r.find_all(["td", "th"], recursive=False) for r in rows]
        cells = [c for row in matrix for c in row]
        nonempty = [_plain(c) for c in cells if _plain(c)]
        if len(rows) < 2 or max(map(len, matrix), default=0) < 2 or len(nonempty) < 4 or len(cells) > 5000:
            continue
        text = " ".join(nonempty)
        signature = tuple(nonempty)
        if signature in seen:
            continue
        if len(text) > 100000 or len(text) / len(nonempty) > 1500:
            continue
        link_chars = sum(len(_plain(a)) for a in table.find_all("a"))
        if link_chars > len(text) * .8 and len(text) / len(nonempty) < 35:
            continue
        html = ["<table>"]
        caption = table.find("caption", recursive=False)
        if caption:
            html.append("<caption>" + escape(_plain(caption)) + "</caption>")
        for row in matrix:
            html.append("<tr>")
            for cell in row:
                attrs = []
                for key in ["rowspan", "colspan"]:
                    try:
                        value = max(1, min(int(cell.get(key, 1)), 1000))
                    except (ValueError, TypeError):
                        value = 1
                    if value > 1:
                        attrs.append(f' {key}="{value}"')
                html.append(f"<{cell.name}{''.join(attrs)}>" + escape(_plain(cell)) + f"</{cell.name}>")
            html.append("</tr>")
        html.append("</table>")
        tables.append({"text": text, "cells": nonempty, "html": "".join(html)})
        seen.add(signature)
        if len(tables) == 64:
            break
    return tables


def _pipe_blocks(content):
    blocks, start, offset, fence = [], None, 0, ""
    for line in content.splitlines(keepends=True) + ["\n"]:
        marker = re.match(r"^(`{3,}|~{3,})", line.strip())
        if marker:
            fence = "" if fence == marker[1][0] else marker[1][0] if not fence else fence
        is_row = not fence and bool(re.match(r"^\|.*\|\s*$", line.strip()))
        if is_row and start is None:
            start = offset
        if not is_row and start is not None:
            blocks.append((start, offset, content[start:offset])); start = None
        offset += len(line)
    return blocks


def restore_source_tables(soup, content):
    """替换能定位的原生表格；补回未被选择的源表，不从参考答案构造内容。"""
    sources = _source_tables(soup)
    diagnostics = {"source_tables": len(sources), "replaced_tables": 0, "recovered_tables": 0, "flat_tables_restored": 0}
    used = set()
    replacements = []
    for start, end, block in _pipe_blocks(content):
        matches = sorted(((_f1(block, t["text"]), i) for i, t in enumerate(sources) if i not in used), reverse=True)
        if matches and matches[0][0] >= .55:
            i = matches[0][1]; used.add(i)
            replacements.append((start, end, sources[i]["html"] + "\n\n")); diagnostics["replaced_tables"] += 1
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + content[end:]
    for i, table in enumerate(sources):
        if i in used:
            continue
        # 已经是结构化HTML的表格无需重复补回。
        if any(_f1(_plain(t), table["text"]) >= .9 for t in BeautifulSoup(content, "lxml").find_all("table")):
            continue
        source_tokens, body_tokens = _tokens(table["text"]), _tokens(content)
        recall = sum((source_tokens & body_tokens).values()) / max(1, sum(source_tokens.values()))
        if recall < .65:
            content += "\n\n" + table["html"]
            diagnostics["recovered_tables"] += 1
            continue
        text_chars, positions = [], []
        for position, char in enumerate(content):
            value = _normalized(char)
            text_chars.extend(value); positions.extend([position] * len(value))
        compact = "".join(text_chars)
        anchors = [_normalized(c) for c in table["cells"] if len(_normalized(c)) >= 8]
        found = [(compact.find(a), len(a)) for a in anchors]
        found = [(p, n) for p, n in found if p >= 0]
        if len(found) >= max(2, ceil(len(anchors) * .65)):
            left, right = min(p for p, _ in found), max(p + n for p, n in found)
            if right - left <= len(_normalized(table["text"])) * 1.4:
                a, b = positions[left], positions[min(right - 1, len(positions) - 1)] + 1
                fence = ""
                for line in content[:a].splitlines():
                    marker = re.match(r"^(`{3,}|~{3,})", line.strip())
                    if marker:
                        fence = "" if fence == marker[1][0] else marker[1][0] if not fence else fence
                if not fence and not re.search(r"</?table\b", content[a:b], re.I):
                    content = content[:a] + "\n\n" + table["html"] + "\n\n" + content[b:]
                    diagnostics["flat_tables_restored"] += 1
                    continue
        # 文字基本已在正文但无法安全界定边界时，不叠加整表制造重复。
    return content.strip(), diagnostics


def enrich_html_body(html, baseline):
    soup = BeautifulSoup(html, "lxml")
    headings = [soup.title, *soup.find_all(["h1", "h2"], limit=5)]
    if any(CHALLENGE_HEADING.fullmatch(_plain(h)) or ERROR_PAGE.fullmatch(_plain(h)) for h in headings if h):
        return baseline, {"profile": "blocked_page", "source_tables": 0, "replaced_tables": 0,
                          "recovered_tables": 0, "flat_tables_restored": 0}
    selected, profile = select_structured_body(soup, baseline)
    selected, tables = restore_source_tables(soup, selected)
    return selected, {"profile": profile, **tables}
