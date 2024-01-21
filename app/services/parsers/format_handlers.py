import re
from typing import cast

import orjson as json
from datetime import timedelta
from xml.etree import ElementTree as ET
import srt
import webvtt

from app.common.consts import SrtString, JsonStr, XMLString
from app.common.models.translation import SRTBlock, Translation
from app.common.utils import rows_to_srt
from app.services.parsers.convertors import xml_to_srt
from app.services.sync import sync_video_to_file


class SubtitlesResults:
    __slots__ = ('translation_obj',)

    def __init__(self, *, translation_obj: Translation):
        self.translation_obj = translation_obj

    @property
    def target_language(self):
        return self.translation_obj.target_language

    @property
    def rows(self):
        return self.translation_obj.subtitles

    def to_srt(self, *, translated: bool = True, reindex=True, start_index=1) -> SrtString:
        return rows_to_srt(rows=self.translation_obj.subtitles, translated=translated, reindex=reindex,
                           start_index=start_index, target_language=self.target_language)

    def to_json(self) -> JsonStr:
        return cast(JsonStr, self.translation_obj.model_dump_json())

    def to_xml(self, root: ET, parent_map: dict[str, ET.Element]) -> XMLString:
        parent_map = {k.strip(): v for k, v in parent_map.items()}
        text_to_translation = {row.content: row.translations.content for row in self.translation_obj.subtitles}
        for original_text, translated_text in text_to_translation.items():
            parent_map[original_text].text = translated_text
        xml = cast(XMLString, ET.tostring(root, encoding='utf-8'))
        return xml.decode('utf-8')

    def to_webvtt(self):
        raise NotImplementedError


class BaseHandler:
    __slots__ = '_results', 'raw_content', 'translation_obj', '_config', 'version'

    DEFAULT_CHUNK_SIZE: int = 10

    def __init__(
            self, *,
            raw_content: str | bytes | SrtString | XMLString,
            translation_obj: Translation,
            version: 1 | 2 | 3 = 3
    ):
        if isinstance(raw_content, bytes):
            raw_content = raw_content.decode('utf-8')
        self.raw_content = raw_content
        self.translation_obj = translation_obj
        self._results: SubtitlesResults | None = None
        self._config = self._extract_config()
        self.version = version

    def _extract_config(self):
        return

    @property
    def config(self):
        return self._config

    def to_rows(self) -> list[SRTBlock]:
        raise NotImplementedError

    def parse_output(self, *, output, **kwargs):
        return output

    @property
    def results(self):
        return self._results


class QTextHandler(BaseHandler):
    def _extract_config(self):
        config_pattern = r"\{.*?\}"
        config_lines = re.findall(config_pattern, self.raw_content)
        config = ' '.join(config_lines)
        config_dict = dict(re.findall(r"{(.*?):(.*?)}", config))
        return config_dict

    def to_rows(self) -> list[SRTBlock]:
        subtitle_pattern = r"\[\d{2}:\d{2}:\d{2}\.\d{2}\].*?\n.*?\n\[\d{2}:\d{2}:\d{2}\.\d{2}\]\n?"
        subtitles = re.findall(subtitle_pattern, self.raw_content, re.DOTALL)
        srt_blocks = []

        def parse_time(time_str):
            hours, minutes, seconds = map(float, time_str.split(':'))
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)

        for index, block in enumerate(subtitles, start=1):
            start_time, content, end_time = [s for s in re.split(r"\n|\[|\]", block.strip()) if  # noqa
                                             s not in ('', '\n', '\r')]
            start_td, end_td = parse_time(start_time), parse_time(end_time)
            srt_block = SRTBlock(index=index, start=start_td, end=end_td, content=content)
            srt_blocks.append(srt_block)

        return srt_blocks

    def parse_output(self, output: SubtitlesResults, *args, **kwargs):
        def format_time(td):
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            milliseconds = td.microseconds // 10000
            return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:02}"

        config_str = ' '.join([f"{{{k}:{v}}}" for k, v in self.config.items()])

        subtitles_str = ""
        for block in output.rows:
            start_str = format_time(block.start)
            end_str = format_time(block.end)
            subtitles_str += f"[{start_str}]\n{block.content}\n[{end_str}]\n\n"

        return config_str + "\n" + subtitles_str


class XMLHandler(BaseHandler):
    def __init__(
            self, *,
            raw_content: str | bytes | SrtString | XMLString,
            translation_obj: Translation,
            version: 1 | 2 | 3 = 3
    ):
        super().__init__(raw_content=raw_content, translation_obj=translation_obj, version=version)
        self.elements = []
        self.root = ET.fromstring(self.raw_content)
        self.parent_map = {}
        self.elements = []

        self._extract_texts(element=self.root, elements=self.elements, parent_map=self.parent_map)

    def _extract_texts(
            self, *, element: ET.Element, elements: list, parent_map: dict[str, ET.Element]
    ) -> None:
        """
        Recursively extract text from an XML element and its children.

        :param element: The current XML element.
        :param elements: list to store the extracted Element objects.
        :param parent_map: Map to store the relationship between texts and their parent elements.
        """
        if element.text and element.text.strip():
            elements.append(element)
            parent_map[element.text] = element
        for child in element:
            self._extract_texts(element=child, elements=elements, parent_map=parent_map)

    def to_rows(self) -> list[SRTBlock]:
        # TODO: Setup PROPER XML parsing + config parsing
        srt_string = xml_to_srt(self.raw_content)
        return srt_to_rows(raw_content=srt_string)

    def parse_output(self, output: SubtitlesResults) -> XMLString:  # noqa
        return output.to_xml(root=self.root, parent_map=self.parent_map)


def srt_to_rows(raw_content: SrtString) -> list[SRTBlock]:
    return sorted(
        [SRTBlock(index=row.index, start=row.start, end=row.end, content=row.content)
         for row in srt.parse(raw_content)],
        key=lambda x: x.start
    )


class SRTHandler(BaseHandler):
    def to_rows(self) -> list[SRTBlock]:
        return srt_to_rows(raw_content=self.raw_content)

    def parse_output(self, *, output: SubtitlesResults, video_file_path=None) -> SrtString:
        original_translation = output.to_srt()
        if video_file_path:
            rows = sync_video_to_file(video_path=self.translation_obj.video_path, srt_string=original_translation)
            return rows_to_srt(rows=rows, target_language=self.translation_obj.target_language)
        return original_translation


class JSONHandler(BaseHandler):
    def to_rows(self) -> list[SRTBlock]:
        return [SRTBlock(**row) for row in json.loads(self.raw_content)]

    def parse_output(self, output: SubtitlesResults, **kwargs) -> JsonStr:
        return output.to_json()


class VTTHandler(BaseHandler):

    def to_rows(self) -> list[SRTBlock]:
        return [
            SRTBlock(index=i, start=row.start, end=row.end, content=row.text)
            for i, row in enumerate(webvtt.read_buffer(self.raw_content), start=1)
        ]


class PACHandler(BaseHandler):
    def to_rows(self) -> list[SRTBlock]:
        ...
