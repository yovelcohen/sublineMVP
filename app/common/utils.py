import re
from typing import cast, Callable

from srt import sort_and_reindex

from common.consts import SrtString

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
        translated: bool = True,
        reindex=True,
        start_index=1,
        strict=True,
        eol=None,
        target_language: str | None = None
) -> SrtString:
    r"""
    Convert an iterator of :py:class:`Subtitle` objects to a string of joined
    SRT blocks.

    :param rows: list of SRTBlocks to construct the SRT from
    :param user_revision: if True, will use the revised translation
    :param translated: if True, will use the translated text
    :param bool reindex: Whether to reindex subtitles based on start time
    :param int start_index: If reindexing, the index to start reindexing from
    :param bool strict: Whether to enable strict mode, see
                        :py:func:`Subtitle.to_srt` for more information
    :param str eol: The end of line string to use (default "\\n")
    :returns: A single SRT formatted string, with each input
              :py:class:`Subtitle` represented as an SRT block
    """
    subtitles = sort_and_reindex(subtitles=rows, start_index=start_index, in_place=True) if reindex else rows
    ret = "".join(subtitle.to_srt(strict=strict, eol=eol, translated=translated) for subtitle in subtitles)
    if target_language in ('Hebrew', 'heb', 'he'):
        ret = _correct_punctuation_alignment(ret)
    return cast(SrtString, ret)


def chunks(li, n=200):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(li), n):
        yield li[i:i + n]
