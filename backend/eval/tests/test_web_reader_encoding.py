"""网页字符解码协议及已发生的中文乱码回归。"""

import codecs
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from service.web_reader_service import WebReaderService


class WebReaderEncodingTests(unittest.TestCase):
    def test_legacy_meta_when_http_omits_charset(self):
        html = '<html><meta charset="gb2312"><p>文物机构9744个，从业人员18.4万人。</p></html>'
        self.assertEqual(WebReaderService._decode_html(html.encode("gb2312"), "text/html"), html)

    def test_http_equiv_and_gbk_extension_characters(self):
        html = '<meta content="text/html; charset=gb2312" http-equiv="Content-Type"><p>扩展字符：镕</p>'
        self.assertEqual(WebReaderService._decode_html(html.encode("gbk"), "text/html"), html)

    def test_invalid_utf8_http_declaration_falls_back_to_meta(self):
        html = '<meta charset="GBK"><p>污水处理费征收标准。</p>'
        self.assertEqual(WebReaderService._decode_html(html.encode("gbk"), "text/html;charset=utf-8"), html)

    def test_quoted_and_case_insensitive_http_charset(self):
        html = '<p>农业银行绿色信贷余额。</p>'
        self.assertEqual(WebReaderService._decode_html(html.encode("gbk"), 'text/html; Charset = "GB2312"'), html)

    def test_valid_http_charset_precedes_conflicting_meta(self):
        html = '<meta charset="utf-8"><p>Crème brûlée</p>'
        self.assertEqual(WebReaderService._decode_html(html.encode("cp1252"), "text/html; charset=windows-1252"), html)

    def test_valid_utf8_survives_stale_meta(self):
        html = '<meta charset="gb2312"><p>正常UTF-8正文，不能再制造乱码。</p>'
        self.assertEqual(WebReaderService._decode_html(html.encode(), "text/html"), html)

    def test_bom_precedes_conflicting_http_and_meta(self):
        html = '<meta charset="gb2312"><p>带BOM的正文。</p>'
        for encoding in ["utf-8-sig", "utf-16", "utf-32"]:
            with self.subTest(encoding=encoding):
                self.assertEqual(WebReaderService._decode_html(html.encode(encoding), "text/html; charset=gbk"), html)

    def test_big_endian_bom(self):
        html = '<p>大端正文。</p>'
        for marker, encoding in [(codecs.BOM_UTF16_BE, "utf-16-be"), (codecs.BOM_UTF32_BE, "utf-32-be")]:
            with self.subTest(encoding=encoding):
                self.assertEqual(WebReaderService._decode_html(marker + html.encode(encoding), "text/html"), html)

    def test_unknown_http_encoding_does_not_hide_valid_meta(self):
        html = '<meta charset="gb2312"><p>政府通知附件。</p>'
        self.assertEqual(WebReaderService._decode_html(html.encode("gbk"), "text/html;charset=unknown-encoding"), html)

    def test_commented_meta_is_ignored(self):
        html = '<!-- <meta charset="windows-1252"> --><meta charset="gbk"><p>真实编码声明。</p>'
        self.assertEqual(WebReaderService._decode_html(html.encode("gbk"), "text/html"), html)

    def test_utf8_and_empty_response(self):
        for html in ["", "<p>正常中文 English 日本語 2026。</p>"]:
            self.assertEqual(WebReaderService._decode_html(html.encode(), "text/html; charset=utf-8"), html)


if __name__ == "__main__":
    unittest.main()
