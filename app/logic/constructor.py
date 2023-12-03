import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import cast, Literal, TypedDict, DefaultDict, Annotated

import streamlit as st
import tiktoken
from pydantic import BaseModel, model_validator, Field
from srt import compose, timedelta_to_srt_timestamp, make_legal_content, sort_and_reindex

from logic.function import translate_via_openai_func, TokenCountTooHigh
from logic.consts import LanguageCode, SrtString, JsonStr, Version

logger = logging.getLogger(__name__)

encoder = tiktoken.encoding_for_model('gpt-4')


class SRTRowDict(TypedDict):
    index: int | str
    start: str
    end: str
    text: str
    translation: str | None
    speaker: int | None


class Subtitle(BaseModel):
    r"""
    The metadata relating to a single subtitle. Subtitles are sorted by start
    time by default. If no index was provided, index 0 will be used on writing
    an SRT block.

    :param index: The SRT index for this subtitle
    :type index: int or None
    :param start: The time that the subtitle should start being shown
    :type start: :py:class:`datetime.timedelta`
    :param end: The time that the subtitle should stop being shown
    :type end: :py:class:`datetime.timedelta`
    :param str proprietary: Proprietary metadata for this subtitle
    :param str content: The subtitle content. Should not contain OS-specific
                        line separators, only \\n. This is taken care of
                        already if you use :py:func:`srt.parse` to generate
                        Subtitle objects.
    """

    index: str | int
    start: timedelta
    end: timedelta
    content: str
    proprietary: str = ''

    def __hash__(self):
        return hash(frozenset(vars(self).items()))

    def __eq__(self, other):
        return vars(self) == vars(other)

    def __lt__(self, other):
        return (self.start, self.end, self.index) < (other.start, other.end, other.index)

    def __repr__(self):
        return f'SRT Block No. {self.index}\nContent: {self.content}'


TranslationResults = DefaultDict[LanguageCode, Annotated[dict, Field(default_factory=dict[Version, str])]]


class SRTBlock(Subtitle):
    index: int | str
    translations: TranslationResults = Field(default_factory=lambda: defaultdict(dict))
    start: timedelta
    end: timedelta
    content: str
    speaker: int | None = None
    region: str | None = None
    style: str | None = None
    num_tokens: int | None = None
    translation: str | None = None

    @model_validator(mode='before')
    @classmethod
    def validate_fields(cls, data: dict):
        index = data.pop('index')
        if isinstance(index, str) and index.lower().startswith('subtitle'):
            index = int(index.split('subtitle')[-1].strip())
        data['index'] = int(index)
        return data

    def to_dict(self) -> SRTRowDict:
        return SRTRowDict(index=self.index, speaker=self.speaker, start=timedelta_to_srt_timestamp(self.start),
                          end=timedelta_to_srt_timestamp(self.end), text=self.content, translation=self.translation)

    def to_srt(self, strict=True, eol="\n", target_language=None, version=None):
        r"""
        Convert the current :py:class:`Subtitle` to an SRT block.

        :param bool strict: If disabled, will allow blank lines in the content
                            of the SRT block, which is a violation of the SRT
                            standard and may cause your media player to explode
        :param str eol: The end of line string to use (default "\\n")
        :param target_language: if provided, tries to get the content string by language in self.translations_result
        :param version: if provided, tries to get the content string by version in self.translations_result
        
        :returns: The metadata of the current :py:class:`Subtitle` object as an
                  SRT formatted subtitle block
        :rtype: str
        """
        if target_language:
            output_content = self.translations[target_language].get(version)
            if not output_content:
                output_content = list(self.translations[target_language].values())[0]
        else:
            output_content = self.content

        output_proprietary = self.proprietary
        if output_proprietary:
            # output_proprietary is output directly next to the timestamp, so
            # we need to add the space as a field delimiter.
            output_proprietary = " " + output_proprietary

        if strict:
            output_content = make_legal_content(output_content)

        if eol is None:
            eol = "\n"
        elif eol != "\n":
            output_content = output_content.replace("\n", eol)

        template = "{idx}{eol}{start} --> {end}{prop}{eol}{content}{eol}{eol}"
        return template.format(
            idx=self.index or 0,
            start=timedelta_to_srt_timestamp(self.start),
            end=timedelta_to_srt_timestamp(self.end),
            prop=output_proprietary,
            content=output_content,
            eol=eol,
        )


class SubtitlesResults:
    __slots__ = 'name', 'target_language', '_srts', 'translated_text'

    def __init__(
            self, *,
            name: str,
            target_language: LanguageCode,
            srts: list[SRTBlock],
            translated_text: defaultdict[int, dict[str, str]]
    ):
        self.name = name
        self.target_language = target_language
        self._srts = sorted([s for s in srts if s.index], key=lambda row: row.index)
        self.translated_text = translated_text

    @classmethod
    def from_translator(
            cls, *,
            translator: 'SRTTranslator',
            translations: defaultdict[int, dict[LanguageCode, str]]
    ):
        return cls(name=translator.name, target_language=translator.target_language, srts=translator.rows,
                   translated_text=translations)

    @property
    def rows(self) -> list[SRTBlock]:
        return self._srts

    def to_srt(
            self,
            version: int = 1,
            target_language: str | None = None,
            reindex=True,
            start_index=1,
            strict=True,
            eol=None
    ) -> SrtString:
        r"""
        Convert an iterator of :py:class:`Subtitle` objects to a string of joined
        SRT blocks.

        .. doctest::

            >>> from datetime import timedelta
            >>> start = timedelta(seconds=1)
            >>> end = timedelta(seconds=2)
            >>> subs = [
            ...     Subtitle(index=1, start=start, end=end, content='x'),
            ...     Subtitle(index=2, start=start, end=end, content='y'),
            ... ]
            >>> compose(subs)  # doctest: +ELLIPSIS
            '1\n00:00:01,000 --> 00:00:02,000\nx\n\n2\n00:00:01,000 --> ...'

        :param target_language: if None will
        :param bool reindex: Whether to reindex subtitles based on start time
        :param int start_index: If reindexing, the index to start reindexing from
        :param bool strict: Whether to enable strict mode, see
                            :py:func:`Subtitle.to_srt` for more information
        :param str eol: The end of line string to use (default "\\n")
        :returns: A single SRT formatted string, with each input
                  :py:class:`Subtitle` represented as an SRT block
        """
        if not version and len(self.translated_text) > 1:
            raise ValueError('version must be provided when there are more than 1 versions')

        subtitles = sort_and_reindex(subtitles=self.rows, start_index=start_index,
                                     in_place=True) if reindex else self.rows
        ret = "".join(
            subtitle.to_srt(strict=strict, eol=eol, target_language=target_language, version=version)
            for subtitle in subtitles
        )
        if self.target_language in ('Hebrew', 'heb', 'he'):
            ret = self._correct_punctuation_alignment(ret)
        return cast(SrtString, ret)

    @property
    def json(self) -> JsonStr:
        return cast(JsonStr, json.dumps([row.to_dict() for row in self._srts]))

    @property
    def webvtt(self):
        raise NotImplementedError

    @property
    def xml(self):
        raise NotImplementedError

    def add_translation_to_rows(self):
        """
        attaches translation to each indexed row
        """
        for version in self.translated_text:
            version = cast(Version, version)
            for row in self._srts:
                if translation := self.translated_text[version].get(str(row.index), None):
                    row.translations[self.target_language][version] = translation

    @property
    def rows_missing_translations(self):
        return [row for row in self.rows if not row.translations.get(self.target_language)]

    # Regular expression to match Hebrew characters followed by a comma or period
    _rtl_punctuation_pattern = re.compile(r'([\u0590-\u05FF]+)([,.])')

    def _correct_punctuation_alignment(self, subtitles: str | SrtString):
        corrected_lines = []
        for line in subtitles.split('\n'):
            # Adjusting punctuation placement for Hebrew text
            line = self._rtl_punctuation_pattern.sub(r'\1\2', line)
            corrected_lines.append(line)

        corrected_subtitles = '\n'.join(corrected_lines)
        return corrected_subtitles


# TODO: Adjust to accept multi options from openai for each chunk.
class SRTTranslator:
    JSON_PARSING_ERROR = 'jp'
    TOKEN_LIMIT_ERROR = 'tl'

    __slots__ = (
        'name', 'target_language', 'rows', 'make_function_translation', 'failed_rows',
        'sema', 'model', 'finished_chunks'
    )

    def __init__(
            self, *,
            project_name: str,
            target_language: LanguageCode,
            deepgram_results: dict = None,
            rows: list[SRTBlock] = None,
            model: Literal['best', 'good'] = 'best'
    ):
        self.name = project_name
        self.target_language = target_language
        if not rows and not deepgram_results:
            raise ValueError('either rows or deepgram_results must be provided')
        self.rows = rows or [
            SRTBlock(index=i, start=utterance['start'], end=utterance['end'], content=utterance['transcript'],
                     speaker=utterance['speaker'], num_tokens=len(encoder.encode(utterance['transcript'])))
            for i, utterance in enumerate(deepgram_results['results']['utterances'], start=1)
        ]
        self.failed_rows: defaultdict[str, defaultdict[int, list[SRTBlock]]] = defaultdict(lambda: defaultdict(list))
        self.sema = asyncio.BoundedSemaphore(12)
        self.model = model
        self.finished_chunks: int = 0

    async def __call__(self, *, num_rows_in_chunk: int = 500, num_options: int = 1) -> SubtitlesResults:
        num_options = num_options or 1
        return await self._translate(num_rows_in_chunk=num_rows_in_chunk, num_options=num_options)

    async def _translate(self, *, num_rows_in_chunk: int = 500, num_options: int = 1) -> SubtitlesResults:
        translations = defaultdict(dict)
        await self._run_and_update(
            translations_results=translations,
            num_rows_in_chunk=num_rows_in_chunk,
            rows=self.rows, num_options=num_options
        )
        logger.debug('finished translating main chunks, handling failed rows')
        await self._handle_failed_rows(
            translations=translations,
            original_num_rows_in_chunk=num_rows_in_chunk,
            num_options=num_options
        )

        results = SubtitlesResults.from_translator(translator=self, translations=translations)
        results.add_translation_to_rows()

        await self._translate_missing(
            translation_holder=results,
            num_rows_in_chunk=num_rows_in_chunk,
            num_options=num_options
        )
        return results

    async def _run_one_chunk(
            self, *,
            results: dict,
            chunk: list[SRTBlock],
            num_chunks,
            per_bar,
            chunk_id: int = None,
            num_options=None
    ):
        async with self.sema:
            try:
                logger.debug('starting to translate chunk number %s via openai', chunk_id)
                t1 = time.time()
                answer = await translate_via_openai_func(
                    rows=chunk, target_language=self.target_language, model=self.model, num_options=num_options
                )
                # mapping from version number to tuple of translations_result and last_idx the json was loaded in
                answer: dict[int, tuple[dict, int]]
                t2 = time.time()
                logger.info('finished translating chunk %s via openai, took %s seconds', chunk_id, t2 - t1)
                self.finished_chunks += 1
                if per_bar and num_chunks and st.session_state.get('bar', False):
                    st.session_state['bar'].progress(
                        min((per_bar * self.finished_chunks), 95),
                        text=f'Finished Translating Chunk No. {self.finished_chunks} Out of {num_chunks} Chunks'
                    )
                for version, (translations, version_last_idx) in answer.items():
                    if version_last_idx != -1:
                        retry_idx = int(version_last_idx) - 1
                        self.failed_rows[self.JSON_PARSING_ERROR][version].extend(chunk[retry_idx:])
                    results[version].update(translations)

            except TokenCountTooHigh:
                logger.exception('failed to translate chunk %s because chunk was too big.', chunk_id)
                self.failed_rows[self.TOKEN_LIMIT_ERROR][version].extend(chunk)
                return dict()
            except Exception as e:
                raise e

    async def _run_and_update(
            self, *,
            translations_results: defaultdict[int, dict],
            rows: list[SRTBlock],
            num_rows_in_chunk: int,
            num_options=None
    ):
        text_chunks = self._divide_rows(rows=rows, num_rows_in_chunk=num_rows_in_chunk)
        amount_chunks = len(text_chunks)
        logger.debug('amount of chunks: %s', len(text_chunks))
        per_bar = 100 // amount_chunks
        tasks = [
            self._run_one_chunk(results=translations_results, chunk=chunk, chunk_id=i, per_bar=per_bar,
                                num_chunks=amount_chunks, num_options=num_options)
            for i, chunk in enumerate(text_chunks, start=1)
        ]
        await asyncio.gather(*tasks)

    async def _handle_failed_rows(
            self, *,
            translations: defaultdict[int, dict],
            original_num_rows_in_chunk: int = 500,
            num_options: int = 1
    ):

        amount_errors = len(self.failed_rows[self.TOKEN_LIMIT_ERROR]) + len(self.failed_rows[self.JSON_PARSING_ERROR])
        if st.session_state.get('bar'):
            st.session_state['bar'].progress(97, text=f'Fixing {amount_errors} Translation Error')

        for option in range(1, num_options + 1):
            if token_limit_errors := self.failed_rows.get(self.TOKEN_LIMIT_ERROR, {}).pop(option, False):
                num_rows_in_chunk = original_num_rows_in_chunk // 2
                logger.info('divided %s rows into %s chunks because of token limit reached',
                            len(token_limit_errors), int(token_limit_errors // num_rows_in_chunk))
                await self._run_and_update(
                    translations_results=translations,
                    num_rows_in_chunk=num_rows_in_chunk,
                    rows=token_limit_errors,
                    num_options=num_options
                )

            if parsing_errors := self.failed_rows.get(self.JSON_PARSING_ERROR, {}).pop(option, False):
                await self._run_and_update(
                    translations_results=translations,
                    num_rows_in_chunk=original_num_rows_in_chunk,
                    rows=parsing_errors,
                    num_options=num_options
                )

    async def _translate_missing(
            self, *,
            translation_holder: SubtitlesResults,
            num_rows_in_chunk: int,
            num_options: int = 1
    ):
        """
        get the missing translations_result
        """
        missing = translation_holder.rows_missing_translations
        if not bool(missing):
            return
        logger.debug('found %s missing translations_result, translating them now', len(missing))
        await self._run_and_update(
            translations_results=translation_holder.translated_text,
            rows=missing,
            num_rows_in_chunk=num_rows_in_chunk,
            num_options=num_options
        )
        translation_holder.add_translation_to_rows()
        if bool(translation_holder.rows_missing_translations):
            await self._translate_missing(translation_holder=translation_holder, num_rows_in_chunk=num_rows_in_chunk)
        return

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


async def main(name_to_fpath, _target_language):
    import srt as srt_lib
    for name, file in name_to_fpath.items():
        with open(file, 'r', ) as f:
            string_data = cast(SrtString, f.read())
        parent_dir = Path(file).parent

        rows = [SRTBlock(index=row.index, start=row.start, end=row.end, content=row.content)
                for row in srt_lib.parse(string_data)]
        logger.info(f'working on file: {name}')
        translator = SRTTranslator(target_language=_target_language, project_name=name, rows=rows, model='good')
        num_options = 1
        results = await translator(num_rows_in_chunk=75, num_options=num_options)
        for v in range(1, num_options + 1):
            with open(parent_dir / f'{name}_{_target_language}_V{v}.srt', 'w') as f:
                f.write(results.to_srt(target_language=_target_language, version=v))
        logger.info(f'finished working on file: {name}')

# if __name__ == '__main__':
#     # Then set our logger in a normal way
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(levelname)s %(asctime)s %(name)s:%(message)s",
#         force=True,
#     )  # Change these settings for your own purpose, but keep force=True at least.
#
#     streamlit_handler = logging.getLogger("streamlit")
#     streamlit_handler.setLevel(logging.DEBUG)
#     logging.getLogger('httpcore').setLevel(logging.INFO)
#     logging.getLogger('openai').setLevel(logging.INFO)
#     logging.getLogger('watchdog.observers').setLevel(logging.INFO)
#     lang = 'Hebrew'
#     ntop = {
#         'TheOffice0409': '/Users/yovel.c/PycharmProjects/services/sublineStreamlit/srts/theOffice0409/original_english.srt'
#     }
#     asyncio.run(main(ntop, lang))
