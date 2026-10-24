"""验证正文补足不能靠拼接导航取巧，且源表不丢跨度或被重复添加。"""
import sys
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from service.web_content_selection import enrich_html_body, restore_source_tables
from service.web_document_structure import limit_body_preserving_tables
from service.web_document import index_document


class StructuredContentTests(unittest.TestCase):
    def test_discussion_keeps_multiple_posts_without_profile_or_navigation(self):
        a = "First participant describes a reproducible installation failure and shares the complete error details."
        b = "Second participant explains how the configuration can be corrected and gives a verified solution."
        html = f'<nav>Login Forum menu</nav><h1>Installation issue</h1><div class="poster">Joined Messages</div><div id="forumposts"><div class="post">{a}</div><div class="post">{b}</div></div>'
        text, meta = enrich_html_body(html, a)
        self.assertIn(a, text); self.assertIn(b, text)
        self.assertNotIn("Joined Messages", text); self.assertNotIn("Forum menu", text)
        self.assertEqual(meta["profile"], "discussion_posts")

    def test_regular_article_does_not_expand_to_navigation(self):
        text = "The article describes audited operating results and explains the changes in revenue."
        html = '<nav>' + ''.join(f'<li>Menu {i}</li>' for i in range(30)) + f'</nav><main><article><p>{text}</p></article></main>'
        actual, _ = enrich_html_body(html, text)
        self.assertEqual(actual, text)

    def test_article_with_recommended_cards_keeps_its_article_boundary(self):
        article = "The official bulletin contains the complete announcement and its publication details. " * 3
        cards = ''.join(f'<li><h3>Unrelated story {i}</h3><p>Recommendations unrelated to the announcement.</p></li>' for i in range(15))
        html = f'<main><article><p>{article}</p></article><ul>{cards}</ul></main>'
        actual, _ = enrich_html_body(html, article.strip())
        self.assertEqual(actual, article.strip())

    def test_one_post_does_not_expand_to_forum_controls(self):
        post = "A collector asks for help identifying the manufacturer and production date of a watch."
        controls = ''.join(f'<li><h3>Forum control {i}</h3></li>' for i in range(20))
        html = f'<main><div id="forumposts"><div class="post">{post}</div></div><ul>{controls}</ul></main>'
        actual, _ = enrich_html_body(html, post)
        self.assertEqual(actual, post)

    def test_moderation_form_does_not_hide_explicit_post_bodies(self):
        a = 'A participant provides a detailed problem report containing the steps needed to reproduce the issue.'
        b = 'Another participant provides the verified solution and the reasoning behind each configuration change.'
        html = f'<form><input type="checkbox"><div id="forumposts"><div class="post">{a}</div><div class="post">{b}</div></div><button>Delete selected posts</button></form>'
        actual, _ = enrich_html_body(html, a)
        self.assertIn(a, actual); self.assertIn(b, actual)
        self.assertNotIn('Delete selected posts', actual)

    def test_recovered_source_table_keeps_spans_and_excludes_footer(self):
        table = '<table><tr><th rowspan="2">Year</th><th>Revenue</th></tr><tr><td>2025</td><td>120</td></tr><tr><td>2026</td><td>150</td></tr></table>'
        html = '<main><p>Audited results.</p>' + table + '</main><footer><table><tr><td>Ads</td><td>Offers</td></tr><tr><td>Login</td><td>Subscribe</td></tr></table></footer>'
        text, meta = enrich_html_body(html, "Audited results.")
        self.assertIn('rowspan="2"', text)
        self.assertIn("150", text); self.assertNotIn("Subscribe", text)
        self.assertEqual(meta["recovered_tables"], 1)
        again, _ = enrich_html_body(html, text)
        self.assertEqual(again.count("<table>"), 1)

    def test_existing_native_table_is_replaced_without_duplicate_numbers(self):
        source = '<table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2025</td><td>12345</td></tr><tr><td>2026</td><td>15678</td></tr></table>'
        text, meta = restore_source_tables(BeautifulSoup(source, "lxml"), '| Year | Revenue |\n| 2025 | 12345 |\n| 2026 | 15678 |')
        self.assertEqual(text.count("12345"), 1)
        self.assertEqual(meta["replaced_tables"], 1)

    def test_hidden_modal_and_download_inventory_are_not_appended(self):
        table = '<table><tr><td>Item</td><td>Value</td></tr><tr><td>AAAA</td><td>1111</td></tr><tr><td>BBBB</td><td>2222</td></tr></table>'
        html = f'<main><p>Research overview.</p><div class="modal fade">{table}</div><div class="files-panel">{table}</div></main>'
        actual, meta = enrich_html_body(html, 'Research overview.')
        self.assertEqual(actual, 'Research overview.')
        self.assertEqual(meta['source_tables'], 0)

    def test_challenge_page_does_not_gain_apparent_body_from_metadata_tables(self):
        html = '<title>Verify you are human</title><table><tr><td>IP</td><td>127.0.0.1</td></tr><tr><td>Agent</td><td>Browser details</td></tr></table>'
        text, meta = enrich_html_body(html, 'Please verify you are human')
        self.assertEqual(text, 'Please verify you are human')
        self.assertEqual(meta['source_tables'], 0)

    def test_html_table_and_qualifying_note_survive_budget_boundary(self):
        table = '<table><tr><td>Long cell</td><td>Values</td></tr><tr><td>2026</td><td>150</td></tr></table>'
        text = 'Intro\n\n' + table + '\n注：金额以百万元计。\n\nFollowing prose'
        actual = limit_body_preserving_tables(text, 30)
        self.assertIn("</table>", actual); self.assertIn("百万元", actual)
        self.assertNotIn("Following prose", actual)

    def test_example_html_inside_code_fence_does_not_expand_budget(self):
        text = '```html\n<table><tr><td>' + 'x' * 100 + '</td></tr></table>\n```'
        self.assertEqual(limit_body_preserving_tables(text, 35), text[:35])

    def test_source_table_stays_whole_in_evidence_sentence_index(self):
        table = '<table><tr><td>2026</td><td>Revenue is USD 1.25 million. Growth is 20%.</td></tr></table>'
        document = index_document(table, 'https://example.com/report')
        self.assertEqual(len(document['sentence_index']), 1)
        span = document['sentence_index'][0]
        self.assertEqual(table[span['start']:span['end']], table)


if __name__ == '__main__':
    unittest.main()
