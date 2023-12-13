import asyncio
import logging
import re
from collections import defaultdict
from typing import cast, Literal, NoReturn
from xml.etree import ElementTree as ET  # noqa

import streamlit as st
from httpx import AsyncClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.consts import SrtString, JsonStr, XMLString, LanguageCode
from common.models.core import SRTBlock, Translation, TranslationContent, TranslationStates
from common.utils import rows_to_srt
from common.context_vars import total_stats
from services.llm.auditor import audit_results_via_openai
from services.llm.translator import translate_via_openai_func
from services.llm.revisor import review_revisions_via_openai

logger = logging.getLogger(__name__)


async def LLM_sentence_matcher(source_sentence: str, candidates: list[str]):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"
    headers = {"Authorization": "Bearer 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"}
    payload = {"inputs": {"source_sentence": source_sentence, "sentences": candidates}}
    async with AsyncClient(headers=headers) as client:
        response = await client.post(API_URL, headers=headers, json=payload)
    return response.json()


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
        text_to_translation = {row.content: row.translations.get_selected(revision_fallback=use_revision)
                               for row in self.translation_obj.subtitles}
        for original_text, translated_text in text_to_translation.items():
            parent_map[original_text].text = translated_text
        xml = cast(XMLString, ET.tostring(root, encoding='utf-8'))
        return xml.decode('utf-8')

    def to_webvtt(self):
        raise NotImplementedError

    @classmethod
    def _conditional_iter_rows(cls, rows, is_revision, negate=False):
        for row in rows:
            translated = row.is_translated(is_revision=is_revision)
            if negate:
                if not translated:
                    yield row
            else:
                if translated:
                    yield row

    @classmethod
    def rows_missing_translations(cls, rows, is_revision=False):
        return cls._conditional_iter_rows(rows=rows, is_revision=is_revision, negate=True)

    @classmethod
    def rows_with_translations(cls, rows, is_revision=False):
        return cls._conditional_iter_rows(rows=rows, is_revision=is_revision, negate=False)


add_space = lambda s: re.sub(r'(?<!^)(?=[A-Z])', ' ', s)


class BaseLLMTranslator:
    __slots__ = (
        'rows', 'make_function_translation', 'failed_rows', 'sema', 'model', 'translation_obj', 'is_revision',
        'iterations'
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
        self.iterations = 0

    @property
    def name(self):
        return self.translation_obj.project_id

    @classmethod
    def class_name(cls):
        return add_space(cls.__qualname__)

    @property
    def target_language(self) -> LanguageCode:
        return cast(LanguageCode, self.translation_obj.target_language)

    async def __call__(
            self, *,
            num_rows_in_chunk: int = 500,
            start_progress_val: int = 30,
            middle_progress_val: int = 45,
            end_progress_val: int = 60
    ) -> SubtitlesResults:
        return await self._translate(
            num_rows_in_chunk=num_rows_in_chunk,
            start_progress_val=start_progress_val,
            middle_progress_val=middle_progress_val,
            end_progress_val=end_progress_val
        )

    async def _translate(self, *, num_rows_in_chunk: int = 75, **kwargs):
        raise NotImplementedError


class TranslatorV1(BaseLLMTranslator):

    async def _translate(
            self, *,
            num_rows_in_chunk: int = 500,
            start_progress_val: int = 30,
            middle_progress_val: int = 45,
            end_progress_val: int = 60
    ) -> SubtitlesResults:
        """
        Class entry point, runs the translation process
        """
        cls_name = self.class_name()
        st.session_state['currentVal'] = start_progress_val
        st.session_state['bar'].progress(start_progress_val, f'Running {cls_name} Translation')
        await self._run_and_update(num_rows_in_chunk=num_rows_in_chunk, rows=self.rows)
        logging.debug('finished translating main chunks, handling failed rows')

        st.session_state['bar'].progress(middle_progress_val, f'Finished generating {cls_name} Translation')

        results = SubtitlesResults(translation_obj=self.translation_obj)
        num_errors = len(tuple(results.rows_missing_translations(rows=self.rows, is_revision=self.is_revision)))
        st.session_state['bar'].progress(middle_progress_val,
                                         f'Handling {cls_name} Translation {num_errors} Errors')
        st.session_state['currentVal'] = middle_progress_val
        await self._translate_missing(translation_holder=results, num_rows_in_chunk=num_rows_in_chunk)
        st.session_state['bar'].progress(end_progress_val,
                                         f'Finished Fixing {cls_name} Translations {num_errors} Errors')
        st.session_state['currentVal'] = end_progress_val

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
                if isinstance(translation, dict):
                    translation = {k.lower(): v for k, v in translation.items()}
                    translation = translation['hebrew']
                row.translations = TranslationContent(content=translation)

    async def _update_translations(self, translations: dict, chunk: list[SRTBlock]) -> NoReturn:
        self._add_translation_to_rows(rows=chunk, results=translations)
        subtitles = self.translation_obj.subtitles.copy() | set(chunk)
        subtitles = set(sorted(list(subtitles), key=lambda row: int(row.index)))
        self.translation_obj = await self.translation_obj.set({Translation.subtitles: subtitles})  # noqa

    async def _run_translation_hook(self, chunk):
        """
        Hook that receives the chunk to work on and calls the proper translation function
        """
        return await translate_via_openai_func(rows=chunk, target_language=self.target_language, model=self.model)

    async def _run_one_chunk(self, *, chunk: list[SRTBlock], chunk_id: int = None) -> NoReturn:
        self_name = self.class_name()
        async with self.sema:
            try:
                logging.debug(f'{self_name}: starting to translate chunk number {chunk_id} via openai')
                answer = await self._run_translation_hook(chunk=chunk)
                await self._update_translations(translations=answer, chunk=chunk)
                progress = pct(
                    len(list(SubtitlesResults.rows_with_translations(rows=chunk, is_revision=self.is_revision))),
                    len(self.rows)
                )
                msg = f'{self_name}: Completed Rows: {progress}%'
                logging.debug(f'Finished chunk number {chunk_id} via openai')
                logging.debug(msg)
                st.session_state['bar'].progress(st.session_state['currentVal'], msg)

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
        self.iterations += len(tasks)

    async def _translate_missing(
            self, *,
            translation_holder: SubtitlesResults,
            num_rows_in_chunk: int,
            recursion_count=0
    ):
        """
        fill the missing translations result recursively
        """
        missing = list(
            translation_holder.rows_missing_translations(is_revision=self.is_revision, rows=translation_holder.rows)
        )
        if len(missing) == 0:
            return

        recursion_count += 1
        logging.debug(
            f'found %s missing translations result on LLM: {self.__class__.__qualname__}, translating them now',
            len(missing)
        )
        if recursion_count > 3:
            if not self.is_revision:
                # TODO: Handle this case, But for now, allow second translation to be missing,
                #       The Issue stems from the Revisor Prompt and model output format.
                logging.warning('failed to translate missing rows, recursion count exceeded')
                st.warning('failed to translate missing rows, recursion exceeded, stopping')
                raise RuntimeError('failed to translate missing rows, recursion exceeded')
            return

        await self._run_and_update(
            rows=missing,
            num_rows_in_chunk=num_rows_in_chunk
        )

        return await self._translate_missing(translation_holder=translation_holder, num_rows_in_chunk=num_rows_in_chunk,
                                             recursion_count=recursion_count)

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


vectorizer = TfidfVectorizer()


def find_most_similar_sentence(source_sentence, sentences_list):
    try:
        combined_sentences = [source_sentence] + sentences_list
        tfidf_matrix = vectorizer.fit_transform(combined_sentences)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        most_similar_index = similarities[0].argmax()
        most_similar_sentence = sentences_list[most_similar_index]
        similarity_score = similarities[0][most_similar_index]
        return most_similar_sentence, similarity_score
    except ValueError as e:
        raise e


class TranslationRevisor(TranslatorV1):
    __slots__ = TranslatorV1.__slots__ + ('unmatched',)

    def __init__(
            self, *,
            translation_obj: Translation,
            rows: list[SRTBlock] = None,
            model: Literal['best', 'good'] = 'best'
    ):
        super().__init__(translation_obj=translation_obj, rows=rows, model=model, is_revision=True)
        self.unmatched = defaultdict(list)

    async def _run_translation_hook(self, chunk):
        return await review_revisions_via_openai(rows=chunk, target_language=self.target_language, model=self.model)

    def _add_translation_to_rows(self, *, rows: list[SRTBlock], results: dict[str, str]):
        """
        attaches translation to each indexed row
        """
        unmatched, success = 0, 0
        for row in rows:
            if translation := results.get(str(row.index), None):
                if row.translations is not None:
                    row.translations.revision = translation
                    success += 1
                else:
                    row.translations = TranslationContent(content=translation)
                    info = {'row_index': row.index, 'row_content': row.content,
                            'revision_translation': translation}
                    logger.debug('Row didnt have V1 translation, assigning revision as V1',
                                 extra=info)
                    self.unmatched['Row didnt have V1 translation, assigning revision as V1'].append(info)
                    unmatched += 1
            else:
                self.unmatched['Translation Not Found'].append({'row_index': row.index, 'row_content': row.content})
                unmatched += 1

        logger.debug(f'added {success} translations to rows, {unmatched} rows were unmatched')


class TranslationAuditor(TranslatorV1):
    def __init__(self, *, translation_obj: Translation, **kwargs):
        kwargs['model'] = 'best'
        super().__init__(translation_obj=translation_obj, **kwargs)

    async def _run_translation_hook(self, chunk):
        return await audit_results_via_openai(rows=set(chunk), target_language=self.target_language)

    def _add_translation_to_rows(self, *, rows: list[SRTBlock], results: dict[str, str]):
        for row in rows:
            if selection := results.get(str(row.index)):
                if selection == '1':
                    row.translations.selected = '1'
                elif selection == '2':
                    row.translations.selected = '2'
                elif isinstance(selection, str) and not selection.isdigit():
                    row.translations.selected = selection
                else:
                    raise ValueError(f'Invalid selection: {selection}')
