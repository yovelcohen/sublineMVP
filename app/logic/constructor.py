import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import NewType, cast, Literal

import streamlit as st
import httpx
import tiktoken
from openai import BadRequestError
from srt import compose, Subtitle

from logic.function import SRTBlock, translate_via_openai_func, TokenCountTooHigh

encoder = tiktoken.encoding_for_model('gpt-4')

logger = logging.getLogger(__name__)

SrtString = NewType('SrtString', str)
JsonStr = NewType('JsonStr', str)


async def process_with_openai(lst, process_func, max_attempts=5, split_factor=2):
    """
    Process a list with an OpenAI API function, automatically retrying with smaller lists if the token limit is exceeded.

    Args:
    lst: The list to be processed.
    process_func: The OpenAI API function to use for processing.
    max_attempts: Maximum number of attempts to retry.
    split_factor: Factor by which to split the list on each retry.

    Returns:
    The processed result.
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            # Attempt to process the list
            return await process_func(lst)
        except BadRequestError as e:
            # Check if the error is due to token limit exceeded
            if 'context_length_exceeded' in str(e):
                if len(lst) <= 1:
                    raise ValueError("Unable to process the list as it's already at minimum size.")
                # Split the list into smaller parts
                mid_index = len(lst) // split_factor
                lst = lst[:mid_index]
                attempt += 1
                logger.info(f"Attempt {attempt}: List size reduced to {len(lst)} to avoid token limit.")
            else:
                # If the error is not due to token limit, re-raise it
                raise
        except Exception as e:
            raise
        await asyncio.sleep(1)  # Adding a short delay before retrying

    raise RuntimeError("Maximum attempts reached without success.")


class TranslatedSRT:
    def __init__(self, constructor: 'ConstructTranslatedSRT'):
        self.constructor = constructor

    @property
    def srt(self) -> SrtString:
        rows = [
            Subtitle(index=row.index, start=row.start, end=row.end, content=row.translation)
            for row in self.constructor.srts
        ]
        return cast(SrtString, compose(rows))

    @property
    def json(self) -> JsonStr:
        return cast(JsonStr, json.dumps([row.to_dict() for row in self.constructor.srts]))

    @property
    def webvtt(self):
        raise NotImplementedError

    @classmethod
    def from_constructor(cls, constructor):
        return cls(constructor=constructor)


class ConstructTranslatedSRT:

    def __init__(self, name: str, srts: list[SRTBlock], translated_text: dict[str, str]):
        self.name = name
        self.srts = sorted([s for s in srts if s.index], key=lambda row: row.index)
        self.translated_text = translated_text
        self.missing_translations = False

    @property
    def results(self) -> TranslatedSRT:
        return TranslatedSRT(self)

    def add_translation_to_rows(self):
        """
        attaches translation to each indexed row
        """
        missing_count = 0
        for row in self.srts:
            if translation := self.translated_text.get(str(row.index), None):
                row.translation = translation
            else:
                missing_count += 1
        if missing_count:
            logger.info(f"missing {missing_count} translations")
            self.missing_translations = True


def back_up_raw_jsons(name: str, original_rows: list[SRTBlock], answer: dict, target_language: str):
    """
    back up the raw jsons from openai in case we need to debug
    """
    answer = {int(k): v for k, v in answer.items()}
    rows = [r.to_dict() for r in original_rows if r.index in answer]
    for row in rows:
        row['translation'] = answer[row['index']]
    with open(f'./raw_jsons/{name}_{target_language}.json', 'w') as f:
        f.write(json.dumps(answer))


class SRTTranslator:
    JSON_PARSING_ERROR = 'jp'
    TOKEN_LIMIT_ERROR = 'tl'

    __slots__ = (
        'name', 'target_language', 'rows', 'make_function_translation', 'failed_rows', 'sema', 'model',
        'finished_chunks')

    def __init__(self, project_name: str, target_language: str, deepgram_results: dict = None,
                 rows: list[SRTBlock] = None, model: Literal['best', 'good'] = 'best'):
        self.name = project_name
        self.target_language = target_language
        self.rows = rows or [
            SRTBlock(index=i, start=utterance['start'], end=utterance['end'], content=utterance['transcript'],
                     speaker=utterance['speaker'], num_tokens=len(encoder.encode(utterance['transcript'])))
            for i, utterance in enumerate(deepgram_results['results']['utterances'], start=1)
        ]
        self.failed_rows: defaultdict[str, list[SRTBlock]] = defaultdict(list)
        self.sema = asyncio.BoundedSemaphore(12)
        self.model = model
        self.finished_chunks: int = 0

    async def __call__(self, num_rows_in_chunk: int = 500) -> TranslatedSRT:
        return await self._translate(num_rows_in_chunk=num_rows_in_chunk)

    async def _translate(self, num_rows_in_chunk: int = 500) -> TranslatedSRT:
        text_chunks = self._divide_rows(self.rows, num_rows_in_chunk)
        amount_chunks = len(text_chunks)
        logger.debug('amount of chunks: %s', len(text_chunks))
        per_bar = 100 // amount_chunks
        tasks = [self._run_one_chunk(chunk, i=i, per_bar=per_bar, num_chunks=amount_chunks)
                 for i, chunk in enumerate(text_chunks, start=1)]
        translated_chunks = await asyncio.gather(*tasks)
        translations = dict()
        for res in translated_chunks:
            translations.update(res)

        logger.debug('finished translating main chunks, handling failed rows')

        await self._handle_failed_rows(translations)

        constructor = ConstructTranslatedSRT(name=self.name, srts=self.rows, translated_text=translations)
        constructor.add_translation_to_rows()
        await self.translate_missing(constructor)

        return constructor.results

    async def _run_one_chunk(self, chunk: list[SRTBlock], num_chunks, per_bar, i: int = None):
        async with self.sema:
            try:
                logger.debug('starting to translate chunk number %s via openai', i)
                t1 = time.time()
                answer, last_idx = await translate_via_openai_func(chunk, target_language=self.target_language,
                                                                   model=self.model)
                t2 = time.time()
                logger.info('finished translating chunk %s via openai, took %s seconds', i, t2 - t1)
                self.finished_chunks += 1
                if per_bar and num_chunks:
                    st.session_state['bar'].progress(
                        min((per_bar * self.finished_chunks), 95),
                        text=f'Finished Translating Chunk No. {self.finished_chunks} Out of {num_chunks} Chunks'
                    )
                # back_up_raw_jsons(name=self.name, original_rows=chunk, answer=answer,
                #                   target_language=self.target_language)
                if last_idx != -1:
                    retry_idx = int(last_idx) - 1
                    self.failed_rows[self.JSON_PARSING_ERROR].extend(chunk[retry_idx:])
                return answer
            except TokenCountTooHigh:
                logger.exception('failed to translate chunk %s because chunk was too big.', i)
                self.failed_rows[self.TOKEN_LIMIT_ERROR].extend(chunk)
                return dict()
            except Exception as e:
                raise e

    async def _handle_failed_rows(self, translations: dict, original_num_rows_in_chunk: int = 500):

        async def update_results(_translations, _chunks):
            tasks = [self._run_one_chunk(chunk, None, None, None) for chunk in _chunks]
            translated_chunks = await asyncio.gather(*tasks)
            for result in translated_chunks:
                _translations.update(result)

        amount_errors = len(self.failed_rows[self.TOKEN_LIMIT_ERROR]) + len(self.failed_rows[self.JSON_PARSING_ERROR])
        st.session_state['bar'].progress(97, text=f'Fixing {amount_errors} Translation Error')

        if token_limit_errors := self.failed_rows.pop(self.TOKEN_LIMIT_ERROR, False):
            num_rows_in_chunk = original_num_rows_in_chunk // 2
            chunks = self._divide_rows(token_limit_errors, num_rows_in_chunk)
            logger.info('divided %s rows into %s chunks because of token limit reached',
                        len(token_limit_errors), len(chunks))
            await update_results(translations, chunks)

        if parsing_errors := self.failed_rows.pop(self.JSON_PARSING_ERROR, False):
            chunks = self._divide_rows(parsing_errors, original_num_rows_in_chunk)
            await update_results(translations, chunks)

    async def translate_missing(self, constructor):
        """
        get the missing translations
        """
        if constructor.missing_translations:
            missing = [row for row in constructor.srts if not row.translation]
            if len(missing) == 0:
                return
            logger.debug('found %s missing translations, translating them now', len(missing))
            translated = await self._run_one_chunk(missing)
            constructor.translated_text.update(translated)
            constructor.add_translation_to_rows()

            if len(missing) > 0:
                await self.translate_missing(constructor=constructor)

            return

    async def _translate_locally(self):

        API_URL = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"
        headers = {"Authorization": "Bearer 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"}
        client = httpx.AsyncClient()

        async def query(payload):
            async with client:
                response = await client.post(API_URL, headers=headers, json=payload)
                return response.json()

        output = query({
            "inputs": "Меня зовут Вольфганг и я живу в Берлине",
        })

    def _divide_rows(self, rows, num_rows_in_chunk: int) -> list[list[SRTBlock]]:
        return [rows[i:i + num_rows_in_chunk] for i in range(0, len(rows), num_rows_in_chunk)]

    def _divide_rows_by_token_limit(self, rows, token_limit):
        sublists, current_str, current_token_count = [[]], str(), 0

        for row in rows:
            if (current_token_count + row.num_tokens) < token_limit:
                sublists[-1].append(repr(row))
                current_token_count += row.num_tokens
            else:
                current_token_count = 0
                sublists.append([repr(row)])

        return ['\n'.join(sublist) for sublist in sublists]


def parse_srt(srt_content: SrtString) -> list[SRTBlock]:
    """
    Parses an SRT file content string and returns a list of SRTBlock objects.

    :param srt_content: SRT file content as a string.
    :return: List of SRTBlock objects.
    """

    def srt_time_to_seconds(srt_time: str) -> float:
        """
        Converts SRT time format to seconds.
        """
        datetime_obj = datetime.strptime(srt_time, '%H:%M:%S,%f')
        return datetime_obj.hour * 3600 + datetime_obj.minute * 60 + datetime_obj.second + datetime_obj.microsecond / 1e6

    srt_rows = []
    for block in filter(None, srt_content.split('\n\n')):
        lines = block.split('\n')
        if len(lines) >= 3:
            index = int(lines[0])
            start_end = lines[1].split(' --> ')
            start = srt_time_to_seconds(start_end[0].replace('\r', ''))
            end = srt_time_to_seconds(start_end[1].replace('\r', ''))
            text = ' '.join(lines[2:])
            srt_rows.append(SRTBlock(index, start, end, text))

    return srt_rows


async def main(num_rows_in_chunk):
    t1 = time.time()
    with open('/Users/yovel.c/PycharmProjects/services/engine/transcriptions/avatar.json', 'r') as f:
        deepgram_data = json.loads(f.read())

    translator = SRTTranslator(target_language='he', deepgram_results=deepgram_data, project_name='avatar')
    _ret = await translator(num_rows_in_chunk=num_rows_in_chunk)
    t2 = time.time()
    logger.info(f'took {t2 - t1} seconds')
    return _ret, t2 - t1


async def from_srt(name, file_path, num_rows_in_chunk):
    t1 = time.time()
    with open(file_path, 'r') as f:
        srt = f.read()

    rows = parse_srt(srt)  # noqa
    translator = SRTTranslator(target_language='he', project_name=name, rows=rows)
    _ret = await translator(num_rows_in_chunk=num_rows_in_chunk)
    t2 = time.time()
    logger.info(f'took {t2 - t1} seconds')
    return _ret, t2 - t1
