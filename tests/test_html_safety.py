"""Escaping tests for everything rendered through unsafe_allow_html.

The views build markup as f-strings and hand it to Streamlit with
``unsafe_allow_html=True``. Most of what goes in is third-party text - headlines
and company names from Yahoo, ticker symbols typed into the app - so an
unescaped angle bracket there is executable markup, not a cosmetic bug.

These tests exist because that was a real hole: news headlines, summaries,
speaker names and the story link all reached the page unescaped, and the link
went straight into an ``href`` where a ``javascript:`` URL would have run on
click.
"""

from __future__ import annotations

import pytest

from views.common import _escape, escape, finding, safe_href, stat

XSS = "<img src=x onerror=alert(1)>"


class TestEscape:
    def test_angle_brackets_are_neutralised(self):
        assert "<img" not in _escape(XSS)
        assert "&lt;img" in _escape(XSS)

    def test_ampersand_escaped_first(self):
        # Escaping & after < would double-encode into &amp;lt;.
        assert _escape("<") == "&lt;"
        assert _escape("&lt;") == "&amp;lt;"

    def test_quotes_escaped_so_attributes_cannot_be_closed(self):
        # Attributes are single-quoted in these f-strings, so an apostrophe is
        # the character that breaks out - as in "Moody's" or "Lowe's".
        assert "'" not in _escape("Moody's")
        assert '"' not in _escape('say "hi"')

    def test_attribute_breakout_is_blocked(self):
        payload = "' onmouseover='alert(1)"
        rendered = f"<div title='{_escape(payload)}'>x</div>"
        assert "onmouseover='alert(1)'" not in rendered

    def test_public_alias_is_the_same_function(self):
        assert escape is _escape

    @pytest.mark.parametrize("value", [None, 0, 1.5, True])
    def test_non_strings_do_not_raise(self, value):
        assert isinstance(_escape(value), str)


class TestSafeHref:
    @pytest.mark.parametrize("url", [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
        "vbscript:msgbox(1)",
        "file:///etc/passwd",
        "//evil.example.com/x",
        "  javascript:alert(1)",
    ])
    def test_dangerous_schemes_are_dropped(self, url):
        assert safe_href(url) is None

    @pytest.mark.parametrize("url", [
        "https://finance.yahoo.com/news/story-123.html",
        "http://example.com/a?b=c&d=e",
    ])
    def test_ordinary_links_survive(self, url):
        assert safe_href(url) is not None

    def test_quote_in_url_cannot_close_the_attribute(self):
        assert safe_href("https://e.com/' onclick='alert(1)") is None

    def test_angle_bracket_in_url_is_rejected(self):
        assert safe_href("https://e.com/<script>") is None

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_link_is_none_not_a_crash(self, value):
        assert safe_href(value) is None

    def test_ampersand_is_encoded_in_the_returned_href(self):
        assert "&amp;" in safe_href("http://example.com/a?b=c&d=e")


class TestNewsHeadlineRendering:
    """The specific construction that was vulnerable, end to end."""

    @staticmethod
    def _item(**over):
        item = {
            "title": "Ordinary headline", "symbols": ["NVDA"], "summary": "",
            "move_pct": 1.0, "sessions": 1, "same_session": True,
            "age_hours": 2.0, "publisher": "Reuters", "link": None,
        }
        item.update(over)
        return item

    def test_hostile_title_is_inert(self):
        from views.news import _headline
        out = _headline(self._item(title=f"Nvidia {XSS}"))
        assert "<img" not in out
        assert "onerror" not in out or "&lt;img" in out

    def test_hostile_summary_is_inert(self):
        from views.news import _headline
        out = _headline(self._item(summary=f"body {XSS}"))
        assert "<img src=x" not in out

    def test_javascript_link_never_reaches_an_href(self):
        from views.news import _headline
        out = _headline(self._item(link="javascript:alert(document.cookie)"))
        assert "javascript:" not in out
        # The headline still renders, just without a link.
        assert "Ordinary headline" in out

    def test_good_link_still_renders_with_noopener(self):
        from views.news import _headline
        out = _headline(self._item(link="https://example.com/story"))
        assert "https://example.com/story" in out
        # target=_blank without noopener hands the opened tab a window.opener
        # reference back into this page.
        assert "noopener" in out

    def test_hostile_speaker_name_is_inert(self):
        from views.news import _headline
        out = _headline(self._item(matched=XSS))
        assert "<img" not in out

    def test_hostile_ticker_is_inert(self):
        from views.news import _headline
        out = _headline(self._item(symbols=[XSS]))
        assert "<img" not in out


class TestSharedComponents:
    """stat() and finding() were already escaping; keep it that way."""

    def test_stat_escapes_its_label_and_value(self):
        out = stat(XSS, XSS, sub=XSS)
        assert "<img" not in out

    def test_finding_escapes_title_and_body(self):
        out = finding("note", XSS, XSS)
        assert "<img" not in out


class TestTickerLogo:
    """The mark is built from a ticker, which reaches a URL and the page."""

    def test_monogram_uses_the_root_symbol(self):
        from views.common import logo
        # "SHOP.TO" is Shopify; initials taken after the split would read "TO".
        assert ">SH<" in logo("SHOP.TO")
        assert ">BR<" in logo("BRK-B")

    def test_the_full_symbol_reaches_the_url(self):
        from views.common import logo
        assert "SHOP.TO.png" in logo("SHOP.TO")

    def test_lowercase_is_normalised(self):
        from views.common import logo
        assert "NVDA.png" in logo("nvda")

    @pytest.mark.parametrize("hostile", [
        "AAPL'><script>alert(1)</script>",
        "AAPL\" onerror=\"alert(1)",
        "../../etc/passwd",
        "AAPL/../../x",
    ])
    def test_hostile_symbols_cannot_escape_the_attribute(self, hostile):
        from views.common import logo
        out = logo(hostile)
        assert "<script" not in out
        assert "onerror=" not in out
        # Nothing but ticker characters survives into the src.
        assert "/../" not in out

    def test_a_symbol_of_only_punctuation_renders_nothing(self):
        from views.common import logo
        assert logo("///") == ""

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_symbol_renders_nothing(self, blank):
        from views.common import logo
        assert logo(blank) == ""

    def test_alt_is_empty_so_a_failed_fetch_shows_the_monogram(self):
        # With alt text, a 404 would print the alt over the initials.
        from views.common import logo
        assert "alt=''" in logo("AAPL")

    def test_size_is_an_integer_in_the_style(self):
        from views.common import logo
        assert "--tik:30px" in logo("AAPL", 30)
