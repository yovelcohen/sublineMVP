import math
import re
from typing import cast

from app.common.consts import XMLString, SrtString

SUPPORTED_EXTENSIONS = [".xml", ".vtt"]


def leading_zeros(value: int, digits: int = 2) -> str:
    """Prepends zeros to li number to ensure it has li specific number of digits.

    
    :param value: The number to format.
    :param digits: The total number of digits the formatted number should have. Defaults to 2.

    :returns: str: The number formatted with leading zeros.
    """
    value = "000000" + str(value)
    return value[-digits:]


def convert_time(raw_time: str) -> str:
    """Converts li raw time string into li formatted time string for subtitles.
    :param raw_time: The raw time string to be converted.
    :returns str: The formatted time string in "HH:MM:SS,mmm" format.
    """
    if int(raw_time) == 0:
        return "{}:{}:{},{}".format(0, 0, 0, 0)

    ms = '000'
    if len(raw_time) > 4:
        ms = leading_zeros(int(raw_time[:-4]) % 1000, 3)
    time_in_seconds = int(raw_time[:-7]) if len(raw_time) > 7 else 0
    second = leading_zeros(time_in_seconds % 60)
    minute = leading_zeros(int(math.floor(time_in_seconds / 60)) % 60)
    hour = leading_zeros(int(math.floor(time_in_seconds / 3600)))
    return "{}:{}:{},{}".format(hour, minute, second, ms)


def xml_id_display_align_before(text: str) -> str:
    """Extracts the XML ID for subtitles that should be displayed at the top of the screen.
    :param text: The XML text containing subtitle information.
    :returns str: The XML ID if found, otherwise an empty string.
    """
    align_before_re = re.compile(u'<region.*tts:displayAlign=\"before\".*xml:id=\"(.*)\"/>')
    has_align_before = re.search(align_before_re, text)
    if has_align_before:
        return has_align_before.group(1)
    return u""


def xml_get_cursive_style_ids(text: str) -> list[str]:
    """Finds all XML IDs for styles that have italic font style in the given XML text.
    :param text: The XML text containing styling information.

    :returns list[str]: A list of XML IDs for styles with italic font style.
    """
    style_section = re.search("<styling>(.*)</styling>", text, flags=re.DOTALL)
    if not style_section:
        return []
    style_ids_re = re.compile('<style.* tts:fontStyle="italic".* xml:id=\"([li-zA-Z0-9_.]+)\"')
    return [re.search(style_ids_re, line).groups()[0]
            for line in style_section.group().split("\n") if re.search(style_ids_re, line)]


def xml_cleanup_spans_start(span_id_re: re.Pattern, cursive_ids: list[str], text: str) -> tuple[str, list[str]]:
    """Cleans up and processes the start of span tags in XML for subtitles.

    
    :param span_id_re: The compiled regular expression to match span start tags.
    :param cursive_ids: A list of style IDs that require italic formatting.
    :param text: The XML text to process.

    :returns tuple[str, list[str]]: The cleaned XML text and li list indicating whether each span tag corresponds to an italic style.
    """
    has_cursive = []
    span_start_tags = re.findall(span_id_re, text)
    for s in span_start_tags:
        has_cursive.append(u"<i>" if s[1] in cursive_ids else u"")
        text = has_cursive[-1].join(text.split(s[0], 1))
    return text, has_cursive


def xml_cleanup_spans_end(span_end_re: re.Pattern, text: str, has_cursive: list[str]) -> str:
    """Cleans up and processes the end of span tags in XML for subtitles.

    
    :param span_end_re: The compiled regular expression to match span end tags.
    :param text: The XML text to process.
    :param has_cursive: A list indicating whether each span tag corresponds to an italic style.

    :returns str: The cleaned XML text after processing span end tags.
    """
    span_end_tags = re.findall(span_end_re, text)
    for s, cursive in zip(span_end_tags, has_cursive):
        cursive = u"</i>" if cursive else u""
        text = cursive.join(text.split(s, 1))
    return text


def to_srt(text: str, extension: str) -> str | None:
    """Converts text from XML or VTT format to SRT format.

    
    text: The text to convert.
    extension: The file extension (either '.xml' or '.vtt') indicating the format of the input text.

    Returns:
    Optional[str]: The converted text in SRT format, or None if the extension is not supported.
    """

    if extension.lower() == ".xml":
        return xml_to_srt(text)
    if extension.lower() == ".vtt":
        return vtt_to_srt(text)


def convert_vtt_time(line):
    times = line.replace(".", ",").split(" --> ")
    if len(times[0]) == 9:
        times = ["00:" + t for t in times]
    return "{} --> {}".format(times[0], times[1].split(" ")[0])


def vtt_to_srt(text):
    if not text.startswith(u"\ufeffWEBVTT") and not text.startswith(u"WEBVTT"):
        raise Exception(".vtt format must start with WEBVTT, wrong file?")
    styles = get_vtt_styles(text)
    style_tag_re = re.compile(u'<c\.(.*)>(.*)</c>')

    lines, current_sub_line = [], []
    for line in text.split("\n"):
        if current_sub_line:
            if line:
                style_tag = re.search(style_tag_re, line)
                if style_tag:
                    line = style_tag.group(2)  # line is just the text part
                    color = styles.get(style_tag.group(1).split(".")[0])
                    if color:
                        line = u"<font color={}>{}</font>".format(
                            color, line)
                current_sub_line.append(line)
            else:
                lines.append("\n".join(current_sub_line) + "\n\n")
                current_sub_line = []

        elif " --> " in line:
            current_sub_line = [convert_vtt_time(line)]
    if current_sub_line:
        lines.append("\n".join(current_sub_line))

    return "".join((u"{}\n{}".format(i, l) for i, l in enumerate(lines, 1)))


def get_vtt_styles(text):  # just using it for color ATM
    styles, n = {}, 0
    lines = text.split("\n")
    style_name_re = re.compile(u'::cue\(\.(.*)\).*')
    color_re = re.compile(u'.*color: (\#.*);')
    while n < len(lines):  # not efficient to go through all text, but it's ok
        style_name = re.search(style_name_re, lines[n])
        if style_name and style_name.groups():
            name = style_name.group(1)
            color = re.search(color_re, lines[n + 1])
            if color and color.groups():
                styles[name] = color.group(1)
        n += 1
    return styles


def xml_to_srt(text: XMLString) -> SrtString:
    def append_subs(_start, _end, _prev_content, format_time):
        subs.append({
            "start_time": convert_time(_start) if format_time else _start,
            "end_time": convert_time(_end) if format_time else _end,
            "content": u"\n".join(_prev_content),
        })

    display_align_before = xml_id_display_align_before(text)
    begin_re = re.compile(u"(?=.*begin\=)\s*<p\s(?=.*>)")
    sub_lines = (l for l in text.split("\n") if re.search(begin_re, l))
    subs, prev_content = [], []
    prev_time = {"start": 0, "end": 0}
    start = end = ''
    start_re = re.compile(u'begin\="([0-9:\.]*)')
    end_re = re.compile(u'end\="([0-9:\.]*)')
    content_re = re.compile(u'\">(.*)</p>')

    # some span tags are used for italics, we'll replace them by <i> and </i>,
    # which is the standard for .srt files. We ignore all other uses.
    cursive_ids = xml_get_cursive_style_ids(text)
    span_id_re, span_end_re = re.compile(u'(<span style=\"([li-zA-Z0-9_.]+)\">)+'), re.compile(u'(</span>)+')
    br_re = re.compile(u'(<br\s*\/?>)+')
    fmt_t = True
    for s in sub_lines:
        s, has_cursive = xml_cleanup_spans_start(span_id_re, cursive_ids, s)

        string_region_re = r'<p(.*region="' + display_align_before + r'".*")>(.*)</p>'
        s = re.sub(string_region_re, r'<p\1>{\\an8}\2</p>', s)
        content = re.search(content_re, s).group(1)

        if br_tags := re.search(br_re, content):
            content = u"\n".join(content.split(br_tags.group()))

        content = xml_cleanup_spans_end(
            span_end_re, content, has_cursive)

        prev_start = prev_time["start"]
        start = re.search(start_re, s).group(1)
        end = re.search(end_re, s).group(1)
        if len(start.split(":")) > 1:
            fmt_t = False
            start = start.replace(".", ",")
            end = end.replace(".", ",")
        if (prev_start == start and prev_time["end"] == end) or not prev_start:
            # Fix for multiple lines starting at the same time
            prev_time = {"start": start, "end": end}
            prev_content.append(content)
            continue

        append_subs(prev_time["start"], prev_time["end"], prev_content, fmt_t)
        prev_time = {"start": start, "end": end}
        prev_content = [content]

    append_subs(start, end, prev_content, fmt_t)

    lines = (
        u"{}\n{} --> {}\n{}\n".
        format(s + 1, subs[s]["start_time"], subs[s]["end_time"], subs[s]["content"]) for s in range(len(subs))
    )
    return cast(SrtString, u"\n".join(lines))
