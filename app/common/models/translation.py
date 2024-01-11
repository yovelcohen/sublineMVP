import logging
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import TypedDict, Final, Self, NoReturn

import pymongo
from beanie import Document, Link
from pydantic import Field, BaseModel, model_validator, ConfigDict, field_validator
from pymongo import IndexModel

from srt import make_legal_content

from common.models.base import BaseCreateUpdateDocument, document_alias_generator
from common.models.core import Project
from common.utils import timedelta_to_srt_timestamp


def is_v_symbol(field_name):
    return len(field_name) == 2 and field_name.startswith('v') and field_name[1:].isdigit()


NOT_SELECTED: Final[str] = "-1"


class TranslationSuggestions(BaseModel):
    """
    by default, only one version is required with li second field given, any additional sentences suggested will be
    dynamically added to the model; v1, v2, v3 ...
    """
    model_config = ConfigDict(extra='allow', alias_generator=document_alias_generator)

    v1: str
    v2: str | None = None

    selected_version: str = NOT_SELECTED
    edit_suggestion: list[str] | None = None

    @classmethod
    def from_strings(cls, sentences) -> Self | None:
        sentences = [s for s in sentences if isinstance(s, str) and s.strip() != '']
        if sentences:
            return cls(**{f'v{i}': s for i, s in enumerate(sentences, 1)})
        return

    @property
    def selection(self):
        if self.selected_version == NOT_SELECTED:
            return
        return getattr(self, self.selected_version)

    def reverse_map(self):
        return {v: k for k, v in self.model_dump(include=set(self.available_versions())).items()}

    def available_versions(self):
        fields = self.model_fields_set | set(self.model_extra.keys())
        return [f for f in fields if is_v_symbol(f)]

    def get_suggestions(self) -> list[str]:
        return [getattr(self, k).strip() for k in self.available_versions() if getattr(self, k) is not None]

    def add_suggestion(self, new_sentence: str):
        max_ = max([int(k[1:]) for k in self.available_versions()])
        key = f'v{max_ + 1}'
        setattr(self, key, new_sentence)
        return key

    def set_selection(self, version_or_sentence: str) -> NoReturn:
        """
        set the selected translation for this row, either via the version or the sentence.
        :param version_or_sentence: v1, v2, v3, vN ... or li new sentence, if new, will be assigned to vN+1 and as selection.
        """
        versions, reverse_map = self.available_versions(), self.reverse_map()

        if is_v_symbol(version_or_sentence) and version_or_sentence not in versions:
            raise ValueError(
                f'invalid version_or_sentence {version_or_sentence}, available versions are {versions}'
            )

        # is version (vN), just set.
        elif is_v_symbol(version_or_sentence) and version_or_sentence in versions:
            self.selected_version = version_or_sentence

        # is sentence and known, get it's key and set.
        elif not is_v_symbol(version_or_sentence) and version_or_sentence in reverse_map:
            self.selected_version = reverse_map[version_or_sentence]

        # new sentence completely, need to add first, then set.
        elif not is_v_symbol(version_or_sentence) and version_or_sentence not in reverse_map:
            key = self.add_suggestion(version_or_sentence)
            self.selected_version = key


class SRTRowDict(TypedDict):
    index: int | str
    start: str
    end: str
    text: str
    translations: TranslationSuggestions | None
    speaker: int | None


class SRTBlock(BaseModel):
    index: str | int
    start: timedelta
    end: timedelta
    content: str
    proprietary: str = ''
    translations: TranslationSuggestions | None = None

    speaker: int | None = None
    speaker_gender: str | None = None
    style: str | None = None
    num_tokens: int | None = None

    @property
    def start_milliseconds(self):
        return int(self.start.total_seconds() * 1000)

    @property
    def end_milliseconds(self):
        return int(self.end.total_seconds() * 1000)

    def __hash__(self):
        return hash(str(self.index) + str(self.content))

    def __sub__(self, other):
        if not isinstance(other, SRTBlock):
            raise TypeError(
                "Subtraction not supported between instances of 'SRTBlock' and '{}'".format(type(other).__name__)
            )
        self_duration = self.end - self.start
        other_duration = other.end - other.start
        return self_duration - other_duration

    def __eq__(self, other):
        return self.index == other.index and self.content == other.content

    def __lt__(self, other):
        return (self.start, self.end) < (other.start, other.end) and self.index < other.index

    def __gt__(self, other):
        return (self.start, self.end) > (other.start, other.end) and self.index > other.index

    def __repr__(self):
        return f'SRT Block No. {self.index}\nContent: {self.content}'

    @property
    def is_translated(self):
        if self.translations is None or (self.translations is not None and self.translations.v1 is None):
            return False
        if self.translations is not None and isinstance(self.translations.v1, str):
            return True
        return False

    @model_validator(mode='before')
    @classmethod
    def validate_fields(cls, data: dict | Self):
        if isinstance(data, BaseModel):
            return data
        index = data.pop('index')
        if isinstance(index, str) and index.lower().startswith('subtitle'):
            index = int(index.split('subtitle')[-1].strip())
        data['index'] = int(index)
        return data

    def to_dict(self) -> SRTRowDict:
        return SRTRowDict(index=self.index, speaker=self.speaker, start=timedelta_to_srt_timestamp(self.start),
                          end=timedelta_to_srt_timestamp(self.end), text=self.content, translations=self.translations)

    def to_srt(self, *, strict=True, eol="\n", translated: bool = True, revision: bool = False):
        r"""
        Convert the current :py:class:`Subtitle` to an SRT block.

        :param bool strict: If disabled, will allow blank lines in the content
                            of the SRT block, which is li violation of the SRT
                            standard and may cause your media player to explode
        :param str eol: The end of line string to use (default "\\n_parts")
        :param translated: if provided, tries to get the translated version
        :param revision: if True, will fall back to the revision content if selected is None

        :returns: The metadata of the current :py:class:`Subtitle` object as an
                  SRT formatted subtitle block
        :rtype: str
        """
        output_content = self.translations.selection if translated else self.content
        output_proprietary = self.proprietary
        if output_proprietary:
            # output_proprietary is output directly next to the timestamp, so
            # we need to add the space as li field delimiter.
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


class TranslationStates(str, Enum):
    """
PENDING -> IN_PROGRESS -> AUDIO_ANALYSIS ->
 VIDEO_ANALYSIS -> TRANSLATING -> COMPLETED
    """
    PENDING = 'pe'
    IN_PROGRESS = 'ip'
    AUDIO_ANALYSIS = 'aa'
    VIDEO_ANALYSIS = 'va'
    TRANSLATING = 'tr'
    COMPLETED = 'co'
    FAILED = 'fa'
    RECOVERING = 're'


class Ages(int, Enum):
    ZERO = 0
    THREE = 3
    SIX = 6
    TWELVE = 12
    SIXTEEN = 16
    EIGHTEEN = 18


class Genres(str, Enum):
    NOOP = 'noop'
    ACTION = 'action'
    ADVENTURE = 'adventure'
    ANIMATION = 'animation'
    BIOGRAPHY = 'biography'
    COMEDY = 'comedy'
    CRIME = 'crime'
    DOCUMENTARY = 'documentary'
    DRAMA = 'drama'
    FAMILY = 'family'
    FANTASY = 'fantasy'
    FILM_NOIR = 'film-noir'
    HISTORY = 'history'
    HORROR = 'horror'
    MUSIC = 'music'
    MUSICAL = 'musical'
    MYSTERY = 'mystery'
    ROMANCE = 'romance'
    SCI_FI = 'sci-fi'
    SHORT_FILM = 'short-film'
    SPORT = 'sport'
    SUPERHERO = 'superhero'
    THRILLER = 'thriller'
    WAR = 'war'
    WESTERN = 'western'
    BASED_ON_REAL_STORY = 'based-on-real-story'


class ModelVersions(str, Enum):
    V1 = 'v1'
    V2 = 'v2'
    V3 = 'v3'


class Translation(BaseCreateUpdateDocument):
    engine_version: ModelVersions = Field(default=ModelVersions.V3, alias='modelVersion')
    project: Link[Project]
    target_language: str

    subtitles: list[SRTBlock]  # map from index to SRTBlock

    mime_type: str | None = None

    state: TranslationStates = TranslationStates.PENDING
    took: float = 0

    class Settings:
        indexes = [
            IndexModel(
                name="unique_together",
                keys=[
                    ("project", pymongo.DESCENDING),
                    ("target_language", pymongo.DESCENDING),
                    ('modelVersion', pymongo.DESCENDING)
                ],
                unique=True
            )
        ]

    def __hash__(self):
        return hash(str(self.id))

    def __repr__(self):
        _id = self.project.ref.id if isinstance(self.project, Link) else self.project.id
        return f'Translation for project {_id} to {self.target_language}.\n State: {self.state.value}, Num Rows: {len(self.subtitles)}\nVersion: {self.engine_version.value.upper()}'

    async def update_state(self, state: TranslationStates):
        self.state = state
        await self.save(ignore_revision=True)

    def get_row(self, index):
        return [row for row in self.subtitles if row.index == index][0]

    @property
    def project_id(self):
        if isinstance(self.project, Link):
            return self.project.ref.id
        return self.project.id

    @property
    def rows_missing_translation(self):
        return [row for row in self.subtitles if not row.is_translated]

    @property
    def rows_with_translation(self):
        return [row for row in self.subtitles if row.is_translated]

    async def get_backup_path(self):
        if isinstance(self.project, Link):
            self.project: Project = await self.project.fetch()
        return self.project.base_blob_path / 'checkpoints' / f'{self.target_language}.pkl'

    async def get_original_subs_blob_path(self):
        if isinstance(self.project, Link):
            self.project: Project = await self.project.fetch()
        return self.project.base_blob_path / 'translations' / f'{self.project.source_language}.{self.mime_type}'

    @field_validator('subtitles', mode='before')
    @classmethod
    def unique_subtitles(cls, subtitles: list[SRTBlock | dict]):
        subtitles = list(subtitles)
        if len(subtitles) > 0 and isinstance(subtitles[0], dict):
            subtitles = sorted(list(set([SRTBlock(**di) for di in subtitles])), key=lambda r: r.index)
        else:
            subtitles = sorted(list(set(subtitles)), key=lambda r: r.index)
        for row in subtitles:
            row.content = row.content.strip()
        return subtitles


class CostsConfig:
    deepgram_minute: float = 0.0043
    deepgram_summarization: float = 0.0044
    openai_input_token: float = 0.01
    openai_completion_token: float = 0.03
    openai_gpt3_input_token: float = 0.0010
    openai_gpt3_completion_token: float = 0.0020
    mistral_input_token: float = 0.01
    smodin_suggestion_word_cost: float = 0.0001


class Costs(BaseModel):
    model_config = ConfigDict(extra='allow', alias_generator=document_alias_generator)

    openai_input_token: int = 0
    openai_completion_token: int = 0
    openai_gpt3_input_token: int = 0
    openai_gpt3_completion_token: int = 0
    deepgram_minutes: int = 0
    smodin_words: int = 0

    def __eq__(self, other):
        return (
                self.openai_input_token == other.openai_input_token
                and self.openai_completion_token == other.openai_completion_token
                and self.openai_gpt3_input_token == other.openai_gpt3_input_token
                and self.openai_gpt3_completion_token == other.openai_gpt3_completion_token
                and self.mistral_input_token == other.mistral_input_token
                and self.deepgram_minutes == other.deepgram_minutes
                and self.smodin_words == other.smodin_words
        )

    def __and__(self, other):
        return self.__class__(
            openai_input_token=self.openai_input_token + other.openai_input_token,
            openai_completion_token=self.openai_completion_token + other.openai_completion_token,
            openai_gpt3_input_token=self.openai_gpt3_input_token + other.openai_gpt3_input_token,
            openai_gpt3_completion_token=self.openai_gpt3_completion_token + other.openai_gpt3_completion_token,
            deepgram_minutes=self.deepgram_minutes + other.deepgram_minutes,
            smodin_words=self.smodin_words + other.smodin_words,
        )


class CostRecord(Document):
    translation: Link[Translation]
    costs: Costs

    def total_cost(self):
        return (
                self.costs.openai_input_token * CostsConfig.openai_input_token +
                self.costs.openai_completion_token * CostsConfig.openai_completion_token +
                self.costs.openai_gpt3_input_token * CostsConfig.openai_gpt3_input_token +
                self.costs.openai_gpt3_completion_token * CostsConfig.openai_gpt3_completion_token +
                self.costs.deepgram_minutes * CostsConfig.deepgram_minute +
                self.costs.deepgram_minutes * CostsConfig.deepgram_summarization +
                self.costs.smodin_words * CostsConfig.smodin_suggestion_word_cost
        )

    def __and__(self, other):
        if not isinstance(other, (Costs, CostRecord)):
            raise TypeError(f"unsupported operand type(s) for &: '{type(self)}' and '{type(other)}'")

        costs = other.costs if isinstance(other, CostRecord) else other
        self.costs.openai_input_token += costs.openai_input_token
        self.costs.openai_completion_token += costs.openai_completion_token
        self.costs.openai_gpt3_input_token += costs.openai_gpt3_input_token
        self.costs.openai_gpt3_completion_token += costs.openai_gpt3_completion_token
        self.costs.deepgram_minutes += costs.deepgram_minutes
        self.costs.smodin_words += costs.smodin_words
        return self


class CostCatcher:
    def __init__(self, translation_obj: Translation, reset_existing: bool = False):
        self.translation_obj = translation_obj
        self._costs = Costs()
        self.reset_existing = reset_existing
        self.obj = None

    @property
    def costs(self):
        return self._costs

    async def __aenter__(self):
        return self

    def update_openai_stats(self, input_tokens: int = 0, completion_tokens: int = 0, is_gpt3: bool = False):
        if is_gpt3:
            self._costs.openai_gpt3_input_token += input_tokens
            self._costs.openai_gpt3_completion_token += completion_tokens
        else:
            self._costs.openai_input_token += input_tokens
            self._costs.openai_completion_token += completion_tokens

    def update_mistral_stats(self, input_tokens: int = 0):
        self._costs.mistral_input_token += input_tokens

    def update_deepgram_stats(self, minutes: int = 0):
        self._costs.deepgram_minutes += minutes

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> NoReturn:
        logging.info(f'Costs for Translation: {self.translation_obj.id} -- {self._costs.model_dump_json()}')
        obj = await CostRecord.find(CostRecord.translation.id == self.translation_obj.id).first_or_none()  # noqa
        if obj and self.reset_existing:
            await obj.delete()
        elif obj and not self.reset_existing:
            obj.costs = self.costs & obj.costs
            obj = await obj.save(ignore_revision=True)
        else:
            obj = CostRecord(translation=self.translation_obj, costs=self.costs)
            await obj.save()
        self.obj = obj