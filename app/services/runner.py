import asyncio
import json
import logging
import re
from datetime import timedelta

import magic
import srt
from xml.etree import ElementTree as ET
import streamlit as st
from beanie import PydanticObjectId

from common.consts import SrtString, XMLString, JsonStr
from common.models.core import Translation, SRTBlock, TranslationStates
from services.constructor import TranslatorV1, SubtitlesResults, TranslationAuditor
from services.convertors import xml_to_srt


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

    def parse_output(self, output, *args, **kwargs):
        return output

    async def _set_state(self, state: TranslationStates):
        self.translation_obj.state = state
        await self.translation_obj.save()

    async def run(self, *, smart_audit: bool = False, raw_results: bool = False, recovery_mode: bool = False):
        try:
            await self._set_state(state=TranslationStates.IN_PROGRESS)
            if recovery_mode:
                self._results = SubtitlesResults(translation_obj=self.translation_obj)
                rows = list(
                    TranslatorV1(translation_obj=self.translation_obj).rows_missing_translations(
                        self.translation_obj.subtitles
                    )
                )
                logging.info(f"Running in recovery mode, found {len(rows)} rows to translate")
                st.toast(f'Running in recovery mode, found {len(rows)} rows to translate')
            else:
                rows = self.to_rows()
            self.translation_obj.subtitles = set(rows)
            await self.translation_obj.save()
            translator = TranslatorV1(translation_obj=self.translation_obj)
            self._results = await translator(num_rows_in_chunk=25)

            if smart_audit:
                try:
                    logging.info("Running Auditor app")
                    await self._set_state(state=TranslationStates.SMART_AUDIT)
                    auditor = TranslationAuditor(translation_obj=self._results.translation_obj, rows=self._results.rows)
                    self._results = await auditor()
                    logging.info('finished running auditor')
                except Exception as e:
                    st.warning(f"Failed to run smart audit, skipping")
                    logging.error(f"Failed to run smart audit, skipping", exc_info=True)

            if raw_results:
                return self._results
            return self.parse_output(self._results)
        except Exception as e:
            await self._set_state(state=TranslationStates.FAILED)
            logging.error(f"Failed to run translation", exc_info=True)
            raise e

    @property
    def results_holder(self):
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
            start_time, content, end_time = [s for s in re.split(r"\n|\[|\]", block.strip()) if
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

    def parse_output(self, output: SubtitlesResults) -> SrtString:  # noqa
        return output.to_srt()


class JSONHandler(BaseHandler):
    def to_rows(self) -> list[SRTBlock]:
        return [SRTBlock(**row) for row in json.loads(self.raw_content)]

    def parse_output(self, output: SubtitlesResults, **kwargs) -> JsonStr:
        return output.to_json()


async def run_translation(
        task: Translation,
        blob_content,
        mime_type: str = None,
        raw_results: bool = False
) -> SubtitlesResults | SrtString | JsonStr | XMLString:
    """

    :param task: a translation DB obj, associated with this run
    :param blob_content: raw string of the subtitles file
    :param mime_type: if not provided, will try to detect the mime type
    :param raw_results: if True, outputs the raw results of the file

    :returns SubtitlesResults if raw_results is True,
             otherwise returns a string of translated subtitles in the uploaded format.
    """
    if not mime_type:
        mime_type = magic.Magic()
        detected_mime_type = mime_type.from_buffer(blob_content)
        detected_mime_type = detected_mime_type.lower()
    else:
        detected_mime_type = mime_type.lower()

    if 'xml' in detected_mime_type:
        blob_content = xml_to_srt(text=blob_content)
        handler = SRTHandler
        # handler = XMLHandler
    elif 'json' in detected_mime_type:
        handler = JSONHandler
    elif 'srt' in detected_mime_type:
        handler = SRTHandler
    elif 'qtext' in detected_mime_type:
        handler = QTtextHandler
    else:
        raise ValueError(f"Unsupported file type: {detected_mime_type}")
    return await handler(raw_content=blob_content, translation_obj=task).run(raw_results=raw_results)


async def main(**name_to_paths):
    from common.config import settings
    from common.db import init_db
    await init_db(settings, [Translation])
    ret = []
    for name, path in name_to_paths.items():
        with open(path, 'r') as f:
            data = f.read()

        task = Translation(target_language='Hebrew', source_language='English', subtitles=[],
                           project_id=PydanticObjectId(), name=name)
        await task.save()
        ret.append(await run_translation(task=task, blob_content=data, raw_results=True, mime_type='srt'))
    return ret


def logging_setup():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s %(name)s:%(message)s",
        force=True,
    )  # Change these settings for your own purpose, but keep force=True at least.
    logging.getLogger('httpcore').setLevel(logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)


if __name__ == '__main__':
    logging_setup()
    _paths = {
        'Suits 0104': '/Users/yovel.c/PycharmProjects/services/sublineStreamlit/srts/suits0104/original_en.srt',
        'Suits 0108': '/Users/yovel.c/PycharmProjects/services/sublineStreamlit/srts/suits0108/Suits - 1x08 - Identity Crisis.HDTV.L0L.en.srt'
    }
    _ret = asyncio.run(main(**_paths))
    print(_ret)
