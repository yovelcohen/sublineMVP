import logging
from datetime import timedelta, datetime
from enum import Enum
from typing import Final, Self, NoReturn, Iterable, NotRequired

import pymongo
import typing_extensions
from beanie import Document, Link
from pydantic import Field, BaseModel, model_validator, ConfigDict, field_validator
from pymongo import IndexModel

from srt import make_legal_content

from common.models.base import BaseCreateUpdateDocument, document_alias_generator  # noqa
from common.models.core import Project  # noqa
from common.utils import timedelta_to_srt_timestamp  # noqa


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

    def get_suggestion(self, version):
        return getattr(self, version)

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


class SRTBlock(BaseModel):
    index: str | int
    start: timedelta
    end: timedelta
    content: str
    proprietary: str = ''
    translations: TranslationSuggestions | None = None

    speaker: int | str | None = None
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

    def __eq__(self, other):
        return self.index == other.index and self.content == other.content

    def __sub__(self, other):
        if not isinstance(other, SRTBlock):
            raise TypeError(
                "Subtraction not supported between instances of 'SRTBlock' and '{}'".format(type(other).__name__)
            )
        self_duration = self.end - self.start
        other_duration = other.end - other.start
        return self_duration - other_duration

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

    def to_srt(self, *, strict=True, eol="\n", translated: bool = True):
        r"""
        Convert the current :py:class:`Subtitle` to an SRT block.

        :param bool strict: If disabled, will allow blank lines in the content
                            of the SRT block, which is li violation of the SRT
                            standard and may cause your media player to explode
        :param str eol: The end of line string to use (default "\\n_parts")
        :param translated: if provided, tries to get the translated version

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


class TranslationSteps(str, Enum):
    """
    PENDING -> IN_PROGRESS -> AUDIO_ANALYSIS ->
        VIDEO_ANALYSIS -> TRANSLATING -> COMPLETED
    |||--> FAILED
    """
    PENDING = 'pe'
    IN_PROGRESS = 'ip'
    AUDIO_ANALYSIS = 'aa'
    VIDEO_ANALYSIS = 'va'
    TRANSLATING = 'tr'
    COMPLETED = 'co'
    FAILED = 'fa'


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
    V031 = 'v3.0.1'
    V032 = 'v0.3.2'
    V033 = 'v0.3.3'
    V034 = 'v0.3.4'
    V035 = 'v0.3.5'
    V036 = 'v0.3.6'
    V037 = 'v0.3.7'
    V038 = 'v0.3.8'
    V039 = 'v0.3.9'
    V0310 = 'v0.3.10'
    V0310_G = 'v0.3.10-g'
    V0311_GENDER = 'v0.3.11-g'
    V0311 = 'v0.3.11'
    V0312 = 'v0.3.12'
    V310_SBS = 'v0.3.10-sbs'

    LATEST = V0311


is_multi_modal = lambda v: v.value.startswith('v3') or '.3.' in v.value


def load_unique_subtitles(subtitles: list[SRTBlock | dict]):
    subtitles = list(subtitles)
    if len(subtitles) > 0 and isinstance(subtitles[0], dict):
        subtitles = sorted(list(set([SRTBlock(**di) for di in subtitles])), key=lambda r: r.index)
    else:
        subtitles = sorted(list(set(subtitles)), key=lambda r: r.index)
    for row in subtitles:
        row.content = row.content.strip()
    return subtitles


class TranslationState(BaseModel):
    execution_id: str | None = Field(
        None, description='Azure Execution ID, can be used to query state and progress', alias='executionId'
    )
    audio_flow_execution_id: str | None = Field(None, alias='audioFlowExecutionId')
    state: TranslationSteps = TranslationSteps.PENDING
    took: float = 0.0


class Translation(BaseCreateUpdateDocument):
    engine_version: ModelVersions = Field(default=ModelVersions.LATEST, alias='modelVersion')
    project: Link[Project]
    target_language: str | None

    # TODO: This is very bad for mongo, we need to move this to blob or something asap
    subtitles: list[SRTBlock]  # map from index to SRTBlock

    mime_type: str | None = None

    flow_state: TranslationState | None = Field(default_factory=TranslationState)

    class Settings:
        indexes = [
            IndexModel(
                name="unique_together",
                keys=[
                    ("project", pymongo.DESCENDING),
                    ("targetLanguage", pymongo.DESCENDING),
                    ('modelVersion', pymongo.DESCENDING)
                ],
                unique=True
            )
        ]

    def __hash__(self):
        return hash(str(self.id))

    def __repr__(self):
        _id = self.project.ref.id if isinstance(self.project, Link) else self.project.id  # noqa
        return f'Translation for project {_id} to {self.target_language}.\n State: {self.flow_state.state.value}, Num Rows: {len(self.subtitles)}\nVersion: {self.engine_version.value.upper()}'

    async def update_state(self, state: TranslationSteps):
        if self.flow_state is None:
            self.flow_state = TranslationState(state=state)
        else:
            self.flow_state.state = state
        await self.save(ignore_revision=True)

    def get_row(self, index):
        return [row for row in self.subtitles if row.index == index][0]

    @property
    def project_id(self):
        if isinstance(self.project, Link):
            return self.project.ref.id
        return self.project.id  # noqa

    @property
    def rows_missing_translation(self):
        return [row for row in self.subtitles if not row.is_translated]

    @property
    def rows_with_translation(self):
        return [row for row in self.subtitles if row.is_translated]

    @field_validator('subtitles', mode='before')
    @classmethod
    def unique_subtitles(cls, subtitles: list[SRTBlock | dict]):
        return load_unique_subtitles(subtitles)


class SpeakerUtterance(BaseModel):
    id: int
    start: timedelta | int | float
    end: timedelta | int | float
    text: str

    @model_validator(mode='before')
    @classmethod
    def validate_time_stamps(cls, data: dict):
        for k in ('start', 'end'):
            if not isinstance(data[k], timedelta):
                data[k] = timedelta(seconds=data[k])
        return data


class AudioIntel(BaseModel):
    speakers: list[SpeakerUtterance]
    description: str


class Chunk(BaseModel):
    id: int
    rows: list[SRTBlock] | None = None
    video_description: str | None = None
    audio_intel: AudioIntel | None = None
    sample_frames: list[str] = Field(default_factory=list)

    def __eq__(self, other):
        if not other:
            return False
        return self.id == other.id and self.start_time == other.start_time and self.end_time == other.end_time

    @field_validator('rows', mode='before')
    @classmethod
    def validate_rows(cls, rows):
        return sorted(rows, key=lambda row: row.index)

    @classmethod
    def lazy_init_from_rows(cls, rows: Iterable[SRTBlock], _id: int):
        # TODO: TMP WORKAROUND Because of failed isinstance checks by pydantic validators on SRTBlock,
        #       Those little fuckers fail because of delayed annotations and imports.
        #       should be fixed when organizing the files.
        klass = cls(id=_id)
        klass.rows = sorted(list(rows), key=lambda row: row.index)
        return klass

    @property
    def start_time(self) -> timedelta:
        return self.rows[0].start

    @property
    def end_time(self) -> timedelta:
        return self.rows[-1].end

    @property
    def min_index(self) -> int:
        return self.rows[0].index

    @property
    def max_index(self) -> int:
        return self.rows[-1].index


class TranslationError(str, Enum):
    GENDER_MISTAKE = 'Gender Mistake'
    TIME_TENSES = 'Time Tenses'
    SLANG = 'Slang'
    PREPOSITIONS = 'Prepositions'
    TYPO = 'Typo'
    NAME_AS_IS = 'Name "as is"'
    NOT_FIT_IN_CONTEXT = 'Not fit in context'
    NOT_FIT_IN_CONTEXT1 = 'not fit in context'
    PLAIN_WRONG_TRANSLATION = 'Plain Wrong Translation'
    NAMES = 'Names'


class IntTranslationError(int, Enum):
    GENDER_MISTAKE = 1
    TIME_TENSES = 2
    SLANG = 3
    PREPOSITIONS = 4
    TYPO = 5
    NAME_AS_IS = 6
    NOT_FIT_IN_CONTEXT = 7
    NOT_FIT_IN_CONTEXT1 = NOT_FIT_IN_CONTEXT
    PLAIN_WRONG_TRANSLATION = 9
    NAMES = 10


class MarkedRow(typing_extensions.TypedDict):
    index: NotRequired[int]
    error: TranslationError
    original: str
    translation: str | None
    fixed: NotRequired[bool]
    score: NotRequired[float]
    guessedErrors: NotRequired[list[TranslationError]]
    correctForm: NotRequired[str]


def align_errors_names(marked_rows: list[MarkedRow]):
    for row in marked_rows:
        if isinstance(row['error'], str):
            row['error'] = TranslationError(row['error'])
        if row['error'] == TranslationError.NOT_FIT_IN_CONTEXT1:
            row['error'] = TranslationError.NOT_FIT_IN_CONTEXT


class TranslationFeedbackV2(Document):
    name: str
    version: ModelVersions = Field(default=ModelVersions.LATEST, alias='engine_version')
    total_rows: int
    marked_rows: list[MarkedRow]
    duration: timedelta | None = None

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('marked_rows', mode='before')
    @classmethod
    def validate_marked_rows(cls, marked_rows: list[MarkedRow]):
        align_errors_names(marked_rows)
        return marked_rows

    async def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        align_errors_names(self.marked_rows)
        return await super().save(*args, **kwargs)

    class Settings:
        indexes = [
            IndexModel(
                name="unique_together",
                keys=[("name", pymongo.DESCENDING), ("engine_version", pymongo.DESCENDING)],
                unique=True
            )
        ]
