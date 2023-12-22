import logging
import re

import orjson as json
from datetime import timedelta
from xml.etree import ElementTree as ET
import srt
import streamlit as st
import webvtt

from common.consts import SrtString, JsonStr, XMLString, VTTString
from common.context_vars import total_stats
from common.models.core import SRTBlock, TranslationStates, Translation
from common.utils import rows_to_srt
from services.constructor import SubtitlesResults, TranslatorV1
from services.sync import sync_video_to_file


class BaseHandler:
    __slots__ = '_results', 'raw_content', 'translation_obj', '_config'

    def __init__(self, *, raw_content: str | bytes | SrtString | XMLString, translation_obj: Translation):
        if isinstance(raw_content, bytes):
            raw_content = raw_content.decode('utf-8')
        self.raw_content = raw_content
        self.translation_obj = translation_obj
        self._results: SubtitlesResults | None = None
        self._config = self._extract_config()

    def _extract_config(self):
        return

    @property
    def config(self):
        return self._config

    def to_rows(self) -> list[SRTBlock]:
        raise NotImplementedError

    def parse_output(self, *, output, **kwargs):
        return output

    async def _set_state(self, state: TranslationStates):
        self.translation_obj.state = state
        await self.translation_obj.save()

    async def _run(self):
        self.translation_obj.subtitles = set(self.to_rows())
        translator = TranslatorV1(translation_obj=self.translation_obj)
        await self.translation_obj.save()
        return await translator(num_rows_in_chunk=25)

    async def _recover(self):
        rows = list(self.translation_obj.rows_missing_translation)
        logging.info(f"Running in recovery mode, found {len(rows)} rows to translate")
        st.toast(f'Running in recovery mode, found {len(rows)} rows to translate')
        translator = TranslatorV1(translation_obj=self.translation_obj)
        await translator.translate_missing(num_rows_in_chunk=25)
        return SubtitlesResults(translation_obj=self.translation_obj)

    async def run(self, recovery_mode: bool = False):
        try:
            await self._set_state(state=TranslationStates.IN_PROGRESS)
            self._results = await self._recover() if recovery_mode else await self._run()
            self._results.translation_obj.tokens_cost = total_stats.get()
            self._results.translation_obj.state = TranslationStates.DONE
            await self._results.translation_obj.replace()
            return self._results
        except Exception as e:
            await self._set_state(state=TranslationStates.FAILED)
            logging.error(f"Failed to run translation", exc_info=True)
            raise e

    @property
    def results(self):
        return self._results


class QTtextHandler(BaseHandler):
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
    def __init__(self, *, raw_content: str | SrtString | XMLString, translation_obj: Translation):
        super().__init__(raw_content=raw_content, translation_obj=translation_obj)
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
        def parse_ttml_timestamp(timestamp_str):
            if not timestamp_str:
                return
            milliseconds_str = timestamp_str.rstrip("t")
            milliseconds = int(milliseconds_str)
            return timedelta(milliseconds=milliseconds)

        ids = [elem.attrib[key] for elem in self.elements for key in elem.attrib if key.endswith("id")]
        blocks = [
            SRTBlock(
                content=elem.text.strip(),
                index=pk,
                style=elem.attrib.get("style"),
                region=elem.attrib.get("region"),
                start=parse_ttml_timestamp(elem.attrib.get("begin")),
                end=parse_ttml_timestamp(elem.attrib.get("end")),
            )
            for pk, elem in zip(ids, self.elements)
        ]
        return blocks

    def parse_output(self, output: SubtitlesResults, use_revision: bool) -> XMLString:  # noqa
        return output.to_xml(root=self.root, parent_map=self.parent_map, use_revision=use_revision)


class SRTHandler(BaseHandler):
    def to_rows(self) -> list[SRTBlock]:
        return [
            SRTBlock(index=row.index, start=row.start, end=row.end, content=row.content)
            for row in srt.parse(self.raw_content)
        ]

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
            SRTBlock(index=i, start=row.start)
            for i, row in enumerate(webvtt.read_buffer(self.raw_content), start=1)
        ]
