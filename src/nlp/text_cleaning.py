"""
Text cleaning (synopsis §5.4). Strips noise from raw Reddit text while
keeping sentence structure intact, since sentence boundaries are what
the entity-linking step relies on.
"""

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")  # [text](url) -> text
MARKDOWN_EMPHASIS_PATTERN = re.compile(r"(\*\*|\*|__|_|~~)")     # bold/italic/strike markers
REDDIT_QUOTE_PATTERN = re.compile(r"^&gt;.*$", re.MULTILINE)     # quoted-reply lines
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
BLANK_LINES_PATTERN = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = URL_PATTERN.sub("", text)
    text = REDDIT_QUOTE_PATTERN.sub("", text)
    text = MARKDOWN_EMPHASIS_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = BLANK_LINES_PATTERN.sub("\n", text)
    return text.strip()
