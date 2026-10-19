"""Check calendar-independent fixture construction without loading model weights."""

import json
import unittest
from unittest.mock import patch

from run_baseline import make_fixture, render_prompt
from run_experiment7_profile import PROFILE_CHAT_TEMPLATE_DATE, parse_args


class CalendarTokenizer:
    """Model the template's date input; this is not a Llama tokenizer test."""

    def __init__(self, today):
        self.today = today
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        date = kwargs.get("date_string", self.today)
        return list((date + json.dumps(messages)).encode())


class PromptDateTests(unittest.TestCase):
    def test_pinned_fixture_survives_calendar_change(self):
        first = CalendarTokenizer("19 Oct 2026")
        second = CalendarTokenizer("20 Oct 2026")
        a = make_fixture(first, 512, "summarization", PROFILE_CHAT_TEMPLATE_DATE)
        b = make_fixture(second, 512, "summarization", PROFILE_CHAT_TEMPLATE_DATE)
        self.assertEqual(a["token_ids"], b["token_ids"])
        self.assertGreater(len(first.calls), 3)  # Includes candidate rendering.
        for call in first.calls + second.calls:
            self.assertEqual(call["date_string"], "04 Oct 2026")

    def test_other_callers_keep_automatic_date_behavior(self):
        first = CalendarTokenizer("19 Oct 2026")
        second = CalendarTokenizer("20 Oct 2026")
        a, _ = render_prompt(first, "notes", "summarize")
        b, _ = render_prompt(second, "notes", "summarize")
        self.assertNotEqual(a, b)
        self.assertNotIn("date_string", first.calls[0])

    def test_profile_default_and_explicit_override(self):
        with patch("sys.argv", ["profile"]):
            self.assertEqual(parse_args().chat_template_date, "04 Oct 2026")
        with patch("sys.argv", ["profile", "--chat-template-date", "05 Oct 2026"]):
            self.assertEqual(parse_args().chat_template_date, "05 Oct 2026")


if __name__ == "__main__":
    unittest.main()
