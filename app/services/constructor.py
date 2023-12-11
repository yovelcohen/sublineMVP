import asyncio
import logging
from typing import cast, Literal, NoReturn
from xml.etree import ElementTree as ET  # noqa

from beanie.odm.operators.update.general import Set

from app.common.consts import SrtString, JsonStr, XMLString, LanguageCode
from app.common.models.core import SRTBlock, Translation, TranslationContent, TranslationStates
from app.common.utils import rows_to_srt
from app.common.context_vars import total_stats
from app.services.llm.translator import translate_via_openai_func, TokenCountTooHigh, encoder
from services.llm.revisor import review_revisions_via_openai


def pct(a, b):
    return round((a / b) * 100, 2)


class SubtitlesResults:
    __slots__ = 'translation_obj', '_srts'

    def __init__(self, *, translation_obj: Translation):
        self.translation_obj = translation_obj
        self._srts = sorted([s for s in translation_obj.subtitles if s.index], key=lambda row: row.index)

    @classmethod
    def from_dict(cls, data: JsonStr):
        translation_obj = Translation.model_validate_json(data)
        return cls(translation_obj=translation_obj)

    @property
    def rows(self) -> list[SRTBlock]:
        return self._srts

    @property
    def target_language(self):
        return self.translation_obj.target_language

    def to_srt(
            self,
            *,
            revision: bool = False,
            target_language: str | None = None,
            reindex=True,
            start_index=1,
            strict=True,
            eol=None
    ) -> SrtString:
        return rows_to_srt(user_revision=revision, rows=self._srts, target_language=target_language, reindex=reindex,
                           start_index=start_index, strict=strict, eol=eol)

    def to_json(self) -> JsonStr:
        return cast(JsonStr, self.translation_obj.model_dump_json())

    def to_xml(self, root: ET, parent_map: dict[str, ET.Element], use_revision: bool = False) -> XMLString:
        parent_map = {k.strip(): v for k, v in parent_map.items()}
        text_to_translation = {row.content: row.translations.get(revision=use_revision)
                               for row in self.translation_obj.subtitles}
        for original_text, translated_text in text_to_translation.items():
            parent_map[original_text].text = translated_text
        xml = cast(XMLString, ET.tostring(root, encoding='utf-8'))
        return xml.decode('utf-8')

    def to_webvtt(self):
        raise NotImplementedError

    def rows_missing_translations(self, is_revision=False):
        if is_revision:
            return [row for row in self.rows if row.translations and row.translations.revision is None]
        return [row for row in self.rows if row.translations is None]


# TODO: Adjust to accept multi options from openai for each chunk.
class SRTTranslator:
    JSON_PARSING_ERROR = 'jp'
    TOKEN_LIMIT_ERROR = 'tl'

    __slots__ = (
        'rows', 'make_function_translation', 'failed_rows', 'sema', 'model', 'translation_obj', 'is_revision'
    )

    def __init__(
            self, *,
            translation_obj: Translation,
            rows: list[SRTBlock] = None,
            model: Literal['best', 'good'] = 'best',
            is_revision: bool = False
    ):
        if not rows and not translation_obj.subtitles:
            raise ValueError('either rows or translation_obj.subtitles must be provided')

        self.translation_obj = translation_obj
        self.rows = rows or translation_obj.subtitles
        self.failed_rows: dict[str, list[SRTBlock]] = dict()
        self.sema = asyncio.BoundedSemaphore(12)
        self.model = model
        self.is_revision = is_revision

    @property
    def name(self):
        return self.translation_obj.project_id

    @property
    def target_language(self) -> LanguageCode:
        return cast(LanguageCode, self.translation_obj.target_language)

    async def __call__(self, *, num_rows_in_chunk: int = 500) -> SubtitlesResults:
        return await self._translate(num_rows_in_chunk=num_rows_in_chunk)

    async def _translate(self, *, num_rows_in_chunk: int = 500) -> SubtitlesResults:
        await self._run_and_update(num_rows_in_chunk=num_rows_in_chunk, rows=self.rows)
        logging.debug('finished translating main chunks, handling failed rows')

        await self._handle_failed_rows(original_num_rows_in_chunk=num_rows_in_chunk)

        results = SubtitlesResults(translation_obj=self.translation_obj)

        await self._translate_missing(translation_holder=results, num_rows_in_chunk=num_rows_in_chunk)
        results.translation_obj.tokens_cost = total_stats.get()
        results.translation_obj.state = TranslationStates.DONE
        await results.translation_obj.save()
        return results

    def _add_translation_to_rows(self, *, rows: list[SRTBlock], results: dict[str, str]):
        """
        attaches translation to each indexed row
        """
        for row in rows:
            if translation := results.get(str(row.index), None):
                row.translations = TranslationContent(content=translation)

    async def _update_translations(self, translations: dict, chunk: list[SRTBlock]) -> NoReturn:
        self._add_translation_to_rows(rows=chunk, results=translations)
        self.translation_obj.subtitles.extend(chunk)
        subtitles = sorted(self.translation_obj.subtitles, key=lambda row: int(row.index))
        self.translation_obj = await self.translation_obj.update(Set({Translation.subtitles: subtitles}))

    async def _run_translation_hook(self, chunk):
        """
        Hook that receives the chunk to work on and calls the proper translation function
        """
        return await translate_via_openai_func(rows=chunk, target_language=self.target_language, model=self.model)

    async def _run_one_chunk(self, *, chunk: list[SRTBlock], chunk_id: int = None) -> NoReturn:
        self_name = self.__class__.name
        async with self.sema:
            try:
                logging.debug(f'{self_name}: starting to translate chunk number {chunk_id} via openai')
                answer = await self._run_translation_hook(chunk=chunk)
                await self._update_translations(translations=answer, chunk=chunk)
                progress = pct(len(self.translation_obj.subtitles), len(self.rows))
                logging.debug(f'{self_name}: Completed Rows: {progress}%')

            except TokenCountTooHigh:
                logging.exception('%cls: failed to translate chunk %s because chunk was too big.', self_name, chunk_id)
                self.failed_rows[self.TOKEN_LIMIT_ERROR][1].extend(chunk)

            except Exception as e:
                raise e

    async def _run_and_update(
            self, *,
            rows: list[SRTBlock],
            num_rows_in_chunk: int
    ):
        text_chunks = self._divide_rows(rows=rows, num_rows_in_chunk=num_rows_in_chunk)
        logging.debug('amount of chunks: %s', len(text_chunks))
        tasks = [self._run_one_chunk(chunk=chunk, chunk_id=i) for i, chunk in enumerate(text_chunks, start=1)]
        await asyncio.gather(*tasks)

    async def _handle_failed_rows(self, *, original_num_rows_in_chunk: int = 500):
        if token_limit_errors := self.failed_rows.get(self.TOKEN_LIMIT_ERROR, False):
            num_rows_in_chunk = original_num_rows_in_chunk // 2
            logging.info('divided %s rows into %s chunks because of token limit reached',
                         len(token_limit_errors), int(len(token_limit_errors) // num_rows_in_chunk))
            await self._run_and_update(
                num_rows_in_chunk=num_rows_in_chunk,
                rows=token_limit_errors
            )

        if parsing_errors := self.failed_rows.get(self.JSON_PARSING_ERROR, False):
            await self._run_and_update(
                num_rows_in_chunk=original_num_rows_in_chunk,
                rows=parsing_errors
            )

    async def _translate_missing(
            self, *,
            translation_holder: SubtitlesResults,
            num_rows_in_chunk: int
    ):
        """
        get the missing translations_result
        """
        missing = translation_holder.rows_missing_translations(is_revision=self.is_revision)
        if len(missing) == 0:
            return
        logging.debug(
            f'found %s missing translations result on LLM: {self.__class__.name}, translating them now', len(missing)
        )
        await self._run_and_update(
            rows=missing,
            num_rows_in_chunk=num_rows_in_chunk
        )

        return await self._translate_missing(translation_holder=translation_holder, num_rows_in_chunk=num_rows_in_chunk)

    def _divide_rows(self, *, rows: list[SRTBlock], num_rows_in_chunk: int) -> list[list[SRTBlock]]:
        return [rows[i:i + num_rows_in_chunk] for i in range(0, len(rows), num_rows_in_chunk)]

    def _divide_rows_by_token_limit(self, *, rows, token_limit):
        sublists, current_str, current_token_count = [[]], str(), 0

        for row in rows:
            if (current_token_count + row.num_tokens) < token_limit:
                sublists[-1].append(repr(row))
                current_token_count += row.num_tokens
            else:
                current_token_count = 0
                sublists.append([repr(row)])

        return ['\n'.join(sublist) for sublist in sublists]


class TranslationRevisor(SRTTranslator):

    def __init__(self, *,
                 translation_obj: Translation,
                 rows: list[SRTBlock] = None,
                 model: Literal['best', 'good'] = 'best'
                 ):
        super().__init__(translation_obj=translation_obj, rows=rows, model=model, is_revision=True)

    async def _run_translation_hook(self, chunk):
        return await review_revisions_via_openai(rows=chunk, target_language=self.target_language, model=self.model)

    def _add_translation_to_rows(self, *, rows: list[SRTBlock], results: dict[str, str]):
        results = {k.strip(): v for k, v in results.items()}
        for row in rows:
            if translation := results.get(row.content.strip(), None):
                row.translations.revision = translation
