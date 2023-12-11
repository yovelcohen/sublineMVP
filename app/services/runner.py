import asyncio
import json
import logging
from datetime import timedelta
from typing import Literal

import magic
import srt
from xml.etree import ElementTree as ET

from beanie import PydanticObjectId

from app.common.consts import SrtString, XMLString, JsonStr
from app.common.models.core import Translation, SRTBlock, TranslationStates
from app.services.constructor import SRTTranslator, SubtitlesResults, TranslationRevisor
from common.config import settings
from common.db import init_db


class BaseHandler:
    _results: SubtitlesResults | None = None

    def __init__(self, *, raw_content: str | SrtString | XMLString, translation_obj: Translation):
        self.raw_content = raw_content
        self.translation_obj = translation_obj

    def to_rows(self) -> list[SRTBlock]:
        raise NotImplementedError

    def parse_output(self, output, *args, **kwargs):
        return output

    async def _set_state(self, state: TranslationStates):
        self.translation_obj.state = state
        await self.translation_obj.save()

    async def run(self, revise: bool = False, raw_results: bool = False, model: Literal['good', 'best'] = 'good'):
        await self._set_state(state=TranslationStates.IN_PROGRESS)

        translator = SRTTranslator(translation_obj=self.translation_obj, rows=self.to_rows(), model=model)
        self._results = await translator(num_rows_in_chunk=75)
        if revise:
            logging.info("Running Revisor app")
            await self._set_state(state=TranslationStates.IN_REVISION)
            revisor = TranslationRevisor(translation_obj=self._results.translation_obj, rows=self._results.rows,
                                         model=model)
            self._results = await revisor(num_rows_in_chunk=75)
            logging.info('finished running revisor')
            await self._set_state(state=TranslationStates.DONE)

        if raw_results:
            return self._results
        return self.parse_output(self._results)

    @property
    def results_holder(self):
        return self._results


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

    def parse_output(self, output: SubtitlesResults, use_revision: bool) -> SrtString:  # noqa
        return output.to_srt(target_language=output.target_language, revision=use_revision)


class JSONHandler(BaseHandler):
    def to_rows(self) -> list[SRTBlock]:
        return [SRTBlock(**row) for row in json.loads(self.raw_content)]

    def parse_output(self, output: SubtitlesResults, **kwargs) -> JsonStr:
        return output.to_json()


async def run_translation(
        task: Translation,
        model,
        blob_content,
        revision: bool = False,
        raw_results: bool = False
) -> SubtitlesResults | SrtString | JsonStr | XMLString:
    """

    :param task: a translation DB obj, associated with this run
    :param model: the GPT to use
    :param blob_content: raw string of the subtitles file
    :param revision: if True, will run a revision round on the results
    :param raw_results: if True, outputs the raw results of the file

    :returns SubtitlesResults if raw_results is True,
             otherwise returns a string of translated subtitles in the uploaded format.
    """
    mime_type = magic.Magic()
    detected_mime_type = mime_type.from_buffer(blob_content)

    detected_mime_type = detected_mime_type.lower()
    if 'xml' in detected_mime_type:
        handler = XMLHandler(raw_content=blob_content, translation_obj=task)
    elif 'json' in detected_mime_type:
        handler = JSONHandler(raw_content=blob_content, translation_obj=task)
    elif 'text' in detected_mime_type:
        handler = SRTHandler(raw_content=blob_content, translation_obj=task)
    else:
        raise ValueError(f"Unsupported file type: {detected_mime_type}")

    results = await handler.run(revise=revision, model=model, raw_results=raw_results)
    return results


async def main(blob, revision=True):
    await init_db(settings, [Translation])
    task = Translation(target_language='Hebrew', source_language='English', subtitles=[], project_id=PydanticObjectId())
    await task.save()
    return await run_translation(task=task, model='good', blob_content=blob, revision=revision, raw_results=True)


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
    with open(
            '/Users/yovel.c/PycharmProjects/services/sublineStreamlit/srts/theOffice0409/original_english.srt', 'r'
    ) as f:
        data = f.read()
    ret = asyncio.run(main(blob=data))
    print(ret)
