import re
from typing import cast, Callable

from srt import sort_and_reindex

from app.common.consts import SrtString

_rtl_punctuation_pattern = re.compile(r'([\u0590-\u05FF]+)([,.])')


def _correct_punctuation_alignment(subtitles: str | SrtString):
    corrected_lines = []
    for line in subtitles.split('\n'):
        # Adjusting punctuation placement for Hebrew text
        line = _rtl_punctuation_pattern.sub(r'\1\2', line)
        corrected_lines.append(line)

    corrected_subtitles = '\n'.join(corrected_lines)
    return corrected_subtitles


def rows_to_srt(
        *,
        rows: list['SRTBlock'],
        user_revision: bool = False,
        target_language: str | None = None,
        reindex=True,
        start_index=1,
        strict=True,
        eol=None
) -> SrtString:
    r"""
    Convert an iterator of :py:class:`Subtitle` objects to a string of joined
    SRT blocks.

    .. doctest::

        >>> from datetime import timedelta
        >>> start = timedelta(seconds=1)
        >>> end = timedelta(seconds=2)
        >>> subs = [
        ...     SRTBlock(index=1, start=start, end=end, content='x'),
        ...     SRTBlock(index=2, start=start, end=end, content='y'),
        ... ]
        >>> compose(subs)  # doctest: +ELLIPSIS
        '1\n00:00:01,000 --> 00:00:02,000\nx\n\n2\n00:00:01,000 --> ...'

    :param rows: list of SRTBlocks to construct the SRT from
    :param user_revision: if True, will use the revised translation
    :param target_language: if None will deafult to original text (self.content)
    :param bool reindex: Whether to reindex subtitles based on start time
    :param int start_index: If reindexing, the index to start reindexing from
    :param bool strict: Whether to enable strict mode, see
                        :py:func:`Subtitle.to_srt` for more information
    :param str eol: The end of line string to use (default "\\n")
    :returns: A single SRT formatted string, with each input
              :py:class:`Subtitle` represented as an SRT block
    """
    subtitles = sort_and_reindex(subtitles=rows, start_index=start_index, in_place=True) if reindex else rows
    ret = "".join(
        subtitle.to_srt(strict=strict, eol=eol, target_language=target_language, revision=user_revision)
        for subtitle in subtitles
    )
    if target_language in ('Hebrew', 'heb', 'he'):
        ret = _correct_punctuation_alignment(ret)
    return cast(SrtString, ret)
