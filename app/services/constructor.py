import asyncio
import datetime
import logging
import re
from collections import defaultdict
from typing import cast, NoReturn
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
from streamlit_utils import stqdm

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
            translated: bool = True,
            reindex=True,
            start_index=1,
            strict=True,
            eol=None
    ) -> SrtString:
        return rows_to_srt(rows=self._srts, translated=translated, reindex=reindex,
                           start_index=start_index, strict=strict, eol=eol, target_language=self.target_language)

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
    def rows_missing_translations(cls, rows):
        for row in rows:
            if row.translations is None:
                yield row

    @classmethod
    def rows_with_translations(cls, rows):
        for row in rows:
            if row.translations is not None and row.translations.content is not None:
                yield row


class BaseLLMTranslator:
    __slots__ = (
        'rows', 'make_function_translation', 'failed_rows', 'sema', 'translation_obj', 'iterations', '_results'
    )

    def __init__(self, translation_obj: Translation, *, rows: list[SRTBlock] = None):
        if not rows and not translation_obj.subtitles:
            raise ValueError('either rows or translation_obj.subtitles must be provided')

        self.translation_obj = translation_obj
        rows = rows or translation_obj.subtitles
        self.rows = sorted(list(rows), key=lambda row: row.index)
        self.failed_rows: dict[str, list[SRTBlock]] = dict()
        self.sema = asyncio.BoundedSemaphore(30)
        self.iterations = 0
        self._results = dict()

    def rows_missing_translations(self, rows):
        for row in rows:
            if row.index not in self._results:
                yield row

    def rows_with_translations(self, rows):
        for row in rows:
            if row.index in self._results:
                yield row

    @classmethod
    def class_name(cls):
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', cls.__qualname__)

    @property
    def target_language(self) -> LanguageCode:
        return cast(LanguageCode, self.translation_obj.target_language)

    async def __call__(self, num_rows_in_chunk: int = 500) -> SubtitlesResults:
        return await self._translate(num_rows_in_chunk=num_rows_in_chunk)

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
        st.session_state['currentVal'] = middle_progress_val

        await self._translate_missing(num_rows_in_chunk=num_rows_in_chunk)
        self.translation_obj.tokens_cost = total_stats.get()
        self.translation_obj.state = TranslationStates.DONE
        await self.translation_obj.save()

        st.session_state['bar'].progress(end_progress_val, f'Finished {cls_name} Translations')
        st.session_state['currentVal'] = end_progress_val

        return SubtitlesResults(translation_obj=self.translation_obj)

    def _add_translation_to_rows(
            self, *, rows: list[SRTBlock], results: dict[str, str]
    ) -> tuple[set[SRTBlock], list[SRTBlock]]:
        """
        attaches translation to each indexed row
        """
        missed, copy = list(), set()
        for row in rows:
            if translation := results.get(str(row.index), None):
                if isinstance(translation, dict):
                    translation = {k.lower(): v for k, v in translation.items()}
                    if 'hebrew' in translation:
                        translation = translation['hebrew']
                    else:
                        if f'{row.index}_1' in translation:
                            translation = ' '.join(translation.values())
                try:
                    row.translations = TranslationContent(content=translation)
                    copy.add(row.model_copy(deep=True))
                    self._results[row.index] = translation
                except Exception as e:
                    print('b')
            else:
                missed.append(row)

        return copy, missed

    async def _update_translations(self, translations: dict, chunk: list[SRTBlock]) -> NoReturn:
        rows, missed = self._add_translation_to_rows(rows=chunk, results=translations)
        existing_translation_c = len(list(self.rows_with_translations(rows=self.translation_obj.subtitles)))
        logging.debug(
            f'before translation object with new translations, len: {len(rows) - len(missed)}, missed: {len(missed)}, on_object: {existing_translation_c}'
        )
        new = self.translation_obj.subtitles | rows
        self.translation_obj.subtitles = new
        self.translation_obj.updated_at = datetime.datetime.utcnow()
        await self.translation_obj.save()
        existing_translation_c1 = len(
            list(self.rows_with_translations(rows=self.translation_obj.subtitles))
        )
        logging.debug(
            f'after translation object with new translations, len: {len(rows) - len(missed)}, missed: {len(missed)}, on_object: {existing_translation_c1}'
        )

    async def _run_translation_hook(self, chunk):
        """
        Hook that receives the chunk to work on and calls the proper translation function
        """
        return await translate_via_openai_func(rows=chunk, target_language=self.target_language)

    async def _run_one_chunk(self, *, chunk: list[SRTBlock], chunk_id: int = None) -> NoReturn:
        self_name = self.class_name()
        async with self.sema:
            try:
                logging.debug(f'{self_name}: starting to translate chunk number {chunk_id} via openai')
                answer = await self._run_translation_hook(chunk=chunk)
                await self._update_translations(translations=answer, chunk=chunk)
                progress = pct(
                    len(list(SubtitlesResults.rows_with_translations(rows=chunk))),
                    len(self.rows)
                )
                logging.debug(f'Finished chunk number {chunk_id} via openai')
                logging.debug(f'{self_name}: Completed Rows: {progress}%')
                st.session_state['bar'].progress(st.session_state['currentVal'], 'Translating...')

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
        await stqdm.gather(*tasks, total=len(tasks))
        self.iterations += len(tasks)

    async def _translate_missing(self, *, num_rows_in_chunk: int, recursion_count=1):
        """
        fill the missing translations result recursively
        """
        missing = list(self.rows_missing_translations(rows=self.rows))
        if len(missing) < 2:
            return

        recursion_count += 1
        logging.debug(
            f'found %s missing translations result on LLM: {self.class_name()}, translating them now',
            len(missing)
        )
        if recursion_count == 3:
            logging.warning('failed to translate missing rows, recursion count exceeded')
            st.warning('failed to translate missing rows, recursion exceeded, stopping')
            raise RuntimeError('failed to translate missing rows, recursion exceeded')

        await self._run_and_update(
            rows=missing,
            num_rows_in_chunk=num_rows_in_chunk
        )

        return await self._translate_missing(num_rows_in_chunk=num_rows_in_chunk, recursion_count=recursion_count)

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
    ):
        super().__init__(translation_obj=translation_obj, rows=rows)
        self.unmatched = defaultdict(list)

    async def _run_translation_hook(self, chunk):
        return await review_revisions_via_openai(rows=chunk, target_language=self.target_language)

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
