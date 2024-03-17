import re
from datetime import timedelta, datetime
from enum import Enum
from pathlib import Path
from typing import Final, Self, Never, cast, NotRequired, Iterable, Literal

import pymongo
import typing_extensions
import yaml
from beanie import Link, PydanticObjectId, Document
from pydantic import Field, BaseModel, model_validator, ConfigDict, field_validator, TypeAdapter
from pydantic.dataclasses import dataclass
from pymongo import IndexModel

from srt import make_legal_content

from common.consts import SrtString, LanguageCode, VSymbolType, SentenceType, RowIndex, AllowedLanguagesForTranslation
from common.models.base import BaseCreateUpdateDocument, document_alias_generator
from common.models.core import Project
from common.models.consts import TranslationSteps, ModelVersions
from common.utils import timedelta_to_srt_timestamp


def is_v_symbol(field_name: str | VSymbolType) -> bool:
    return len(field_name) == 2 and field_name.startswith('v') and field_name[1:].isdigit()


MAX_SENTENCE_LEN: Final[int] = 32


def split_sentences(sentence: str) -> str:
    words = re.findall(r'\S+|\s+', sentence)
    split_sentences, lines, current_line = list(), list(), ""

    for word in words:
        if len(current_line) + len(word) <= 32:
            current_line += word
        else:
            lines.append(current_line.strip())
            current_line = word

    if current_line:
        lines.append(current_line.strip())

    # Join the lines into a single string
    split_sentence = "\n".join(lines)
    split_sentences.append(split_sentence)

    return '\n'.join(split_sentences)


NOT_SET: Final[str] = "-1"
OtherTypeError = lambda operand, other: TypeError(
    f"unsupported operand {operand} for: 'SRTBlock' and '{type(other)}'"
)

tag_re = re.compile(r'<[^>]+>')


def strip_tags(xml_str) -> str:
    return tag_re.sub('', xml_str)


class TranslationSuggestions(BaseModel):
    """
    by default, only one version is required with second field given, any additional sentences suggested will be
    dynamically added to the model; v1, v2, v3 ...

    Preferably initiate vie TranslationSuggestions.from_strings
    """
    model_config = ConfigDict(extra='allow', alias_generator=document_alias_generator)

    v1: SentenceType
    v2: SentenceType | None = None

    selected_version: VSymbolType | Literal['-1'] = NOT_SET
    edit_suggestion: list[str] | None = None

    @classmethod
    def from_strings(cls, sentences: list[SentenceType]) -> Self | None:
        """
        Create a TranslationSuggestions object from a list of sentences.
        The sentences list will maintain the original input order.
        :returns: a new TranslationSuggestions object or None if no sentences were provided.
        """
        _cls = cls(
            v1=cls.prepare_sentence(sentences.pop(0)),
            v2=cls.prepare_sentence(sentences.pop(0)) if len(sentences) > 1 else None
        )
        for sentence in sentences:
            _cls.add_suggestion(sentence)
        return _cls

    @classmethod
    def prepare_sentence(cls, sentence: SentenceType):
        sentence = sentence.strip()
        sentence = strip_tags(sentence)
        # TODO: the split should be used only on outputting subtitles in a format that requires it.
        #       when doing it here, it just leads to extra tokens and might confuse the llm if appears much.
        # return split_sentences(sentence) if (len(sentence) > MAX_SENTENCE_LEN and '\n' not in sentence) else sentence
        return sentence

    def __repr__(self):
        versions_map = self.as_suggestions_dict()
        if self.selected_version != NOT_SET:
            versions_map[self.selected_version] = f'**{versions_map[self.selected_version]}**'
        return f'TranslationSuggestions:\n{yaml.dump(versions_map)}'

    @property
    def is_set(self):
        return self.selected_version != NOT_SET

    def merge_from_others(
            self,
            others: list['TranslationSuggestions'],
            *,
            selection_priority: int | None = None,
            inplace=True
    ) -> Self:
        """
        merge the suggestions from other TranslationSuggestions objects into with this one inplace or into a new object.

        :param others: other TranslationSuggestions objects.
        :param selection_priority: index of the TranslationSuggestions (in "others") object to take the selection from.
                                   if none, will not set a selection on the new/updated object.
        :param inplace: if True, merges others into Self, if False, creates a new TranslationSuggestions instance.
        """
        others, all_versions = list(others), {i.lower(): i for i in self.get_suggestions()}

        def recursively_extract_all_versions(_others):
            for idx, other in enumerate(_others):
                if isinstance(other, list):
                    return recursively_extract_all_versions(other)
                else:
                    suggestions = other.get_suggestions()
                    all_versions.update({s.lower(): s for s in suggestions})

        if inplace:
            for v in all_versions.values():
                self.add_suggestion(v)
            if selection_priority is not None and selection_priority > 0:
                selection = others[selection_priority].selection
                self.set_selection(selection)
            return self
        else:
            new = TranslationSuggestions.from_strings(list(all_versions.values()))
            if selection_priority is not None:
                selection = self.selection if selection_priority == 0 else others[selection_priority].selection
                new.set_selection(selection)
            return new

    @property
    def selection(self) -> SentenceType | None:
        return getattr(self, self.selected_version) if self.selected_version != NOT_SET else None

    def as_suggestions_dict(self) -> dict[VSymbolType, SentenceType]:
        return {k: getattr(self, k) for k in self.available_versions()}

    def reverse_map(self) -> dict[SentenceType, VSymbolType]:
        return {v: k for k, v in self.model_dump(include=self.available_versions()).items()}

    def available_versions(self) -> set[VSymbolType]:
        return {f for f in (self.model_fields_set | set(self.model_extra.keys())) if is_v_symbol(f)}

    def get_suggestions(self) -> list[SentenceType]:
        return [getattr(self, k).strip() for k in self.available_versions() if getattr(self, k) is not None]

    def add_suggestion(self, new_sentence: SentenceType) -> VSymbolType:
        new_sentence = self.prepare_sentence(new_sentence)
        if new_sentence.lower() in map(lambda x: x.lower(), self.get_suggestions()):
            return self.reverse_map()[new_sentence]
        max_ = max([int(k[1:]) for k in self.available_versions()])
        key = f'v{max_ + 1}'
        setattr(self, key, new_sentence)
        return key

    def set_selection(self, version_or_sentence: RowIndex | VSymbolType | SentenceType) -> Never:
        """
        set the selected translation for this row, either via the version or the sentence.
        :param version_or_sentence: v1, v2, v3, vN ... ornew sentence, if new, will be assigned to vN+1 and as selection.
        """
        if isinstance(version_or_sentence, int):
            version_or_sentence, item_is_v_symbol = f'v{version_or_sentence}', True
        else:
            item_is_v_symbol = is_v_symbol(version_or_sentence)

        if item_is_v_symbol:
            versions = self.available_versions()
            if version_or_sentence in versions:  # is version (vN), just set.
                self.selected_version = version_or_sentence
            else:
                raise ValueError(
                    f'invalid version_or_sentence {version_or_sentence}, available versions are {versions}'
                )
        else:
            reverse_map = self.reverse_map()
            if version_or_sentence in reverse_map:  # is a known sentence, get it's key and set.
                self.selected_version = reverse_map[version_or_sentence]
            else:  # new sentence completely, need to add first, then set.
                key = self.add_suggestion(version_or_sentence)
                self.selected_version = key

    def re_rank(self, ranks: list[VSymbolType]) -> Self:
        original = self.as_suggestions_dict()
        new_strings = [original.pop(f'v{rank}') for rank in ranks]
        ts = TranslationSuggestions(v1=new_strings.pop(0), v2=new_strings.pop(0) if new_strings else None)
        ts.set_selection('v1')
        for sentence in new_strings:
            ts.add_suggestion(sentence)
        return ts


@dataclass(config=ConfigDict(alias_generator=document_alias_generator, extra='forbid', populate_by_name=True))
class SubtitlePosition:
    x: float
    y: float


class SubtitleStyle(BaseModel):
    position: SubtitlePosition
    font_name: str = "Arial"
    font_size: int = 24
    font_color: str = "white"
    background_color: str = "transparent"
    bold: bool = False
    italic: bool = False
    underline: bool = False

    model_config = ConfigDict(extra='forbid', alias_generator=document_alias_generator, populate_by_name=True)

    @field_validator('position', mode='before')
    @classmethod
    def validate_position(cls, v):
        if isinstance(v, tuple) and len(v) == 2:
            return SubtitlePosition(x=v[0], y=v[1])
        elif isinstance(v, (SubtitlePosition, dict)):
            return v
        raise ValueError('Position must be a tuple with two elements or a Position namedtuple')


class SRTBlock(BaseModel):
    index: RowIndex
    start: timedelta
    end: timedelta
    content: SentenceType
    translations: TranslationSuggestions | None = None

    speaker: str | int | None = None
    speaker_gender: str | None = None
    style: str | None | SubtitleStyle = None
    num_tokens: int | None = None

    model_config = ConfigDict(json_encoders={timedelta: lambda x: str(x)}, validate_assignment=True)

    @model_validator(mode='before')
    @classmethod
    def validate_fields(cls, data: dict | Self):
        if isinstance(data, BaseModel):
            return data
        index = data.pop('index')
        if isinstance(index, str) and index.lower().startswith('subtitle'):
            index = int(index.split('subtitle')[-1].strip())
        data['index'] = int(index)
        style = data.pop('style', None)
        if isinstance(style, (SubtitleStyle, dict)):
            data['style'] = style
        if content := data.get('content'):
            data['content'] = strip_tags(content.strip())
        return data

    @property
    def duration_seconds(self):
        return self.end.total_seconds() - self.start.total_seconds()

    @property
    def start_milliseconds(self):
        return int(self.start.total_seconds() * 1000)

    @property
    def end_milliseconds(self):
        return int(self.end.total_seconds() * 1000)

    @property
    def is_translated(self) -> bool:
        return self.translations is not None and isinstance(self.translations.v1, str)

    def __hash__(self):
        return hash(str(self.index) + str(self.content))

    def __sub__(self, other):
        if not isinstance(other, SRTBlock):
            raise OtherTypeError("subtraction", other)
        self_duration = self.end - self.start
        other_duration = other.end - other.start
        return self_duration - other_duration

    def __eq__(self, other):
        if not isinstance(other, SRTBlock):
            raise OtherTypeError("equality", other)
        return self.index == other.index and self.content == other.content

    def __lt__(self, other):
        if not isinstance(other, SRTBlock):
            raise OtherTypeError("less than", other)
        return (self.start, self.end) < (other.start, other.end) and self.index < other.index

    def __gt__(self, other):
        if isinstance(other, timedelta):
            return self.start > other
        if not isinstance(other, SRTBlock):
            raise OtherTypeError("greater than", other)
        return (self.start, self.end) > (other.start, other.end) and self.index > other.index

    def __le__(self, other):
        if isinstance(other, timedelta):
            return self.start < other
        if not isinstance(other, SRTBlock):
            raise OtherTypeError("less than or equal", other)
        return (self.start, self.end) <= (other.start, other.end) and self.index <= other.index

    def __repr__(self):
        return f'SRT Block No. {self.index}\nContent: {self.content}'

    def to_srt(self, translated: bool = True, *, strict=True, eol="\n") -> SrtString | None:
        r"""
        Convert self to an SRT format string

        :param translated: if provided, tries to get the translated version
        :param bool strict: If disabled, will allow blank lines in the content
                            of the SRT block, which isviolation of the SRT
                            standard and may cause your media player to explode
        :param str eol: The end of line string to use (default "\\n_parts")
        :rtype: str
        """
        if translated and not self.translations:
            return
        output_content = self.translations.selection if translated else self.content

        if strict:
            output_content = make_legal_content(output_content)

        if eol is None:
            eol = "\n"
        elif eol != "\n":
            output_content = output_content.replace("\n", eol)

        template = "{idx}{eol}{start} --> {end}{eol}{content}{eol}{eol}"
        return cast(
            SrtString,
            template.format(
                idx=int(self.index),
                start=timedelta_to_srt_timestamp(self.start),
                end=timedelta_to_srt_timestamp(self.end),
                content=output_content,
                eol=eol,
            )
        )


SRTBlockAdapter = TypeAdapter(list[SRTBlock])


def load_unique_subtitles(subtitles: list[SRTBlock | dict]) -> list[SRTBlock]:
    subtitles = list(subtitles)
    if len(subtitles) > 0 and isinstance(subtitles[0], dict):
        subtitles = sorted(list(set([SRTBlock(**di) for di in subtitles])), key=lambda r: r.index)
    else:
        subtitles = sorted(list(set(subtitles)), key=lambda r: r.index)
    return subtitles


class TranslationState(BaseModel):
    execution_id: str | None = Field(
        None, description='Azure Execution ID, can be used to query state and progress', alias='executionId'
    )
    audio_flow_execution_id: str | None = Field(None, alias='audioFlowExecutionId')
    state: TranslationSteps = TranslationSteps.PENDING
    took: float = 0.0
    failure_reason: str | None = None


class Translation(BaseCreateUpdateDocument):
    model_config = ConfigDict(**BaseCreateUpdateDocument.model_config, extra='ignore')

    engine_version: ModelVersions = Field(default=ModelVersions.LATEST, alias='modelVersion')
    project: Link[Project]
    target_language: AllowedLanguagesForTranslation

    # TODO: This is very bad for mongo, we need to move this to blob or something asap
    subtitles: list[SRTBlock]  # map from index to SRTBlock
    results_path: Path | None = None
    flow_state: TranslationState | None = Field(default_factory=TranslationState)

    # class Settings:
    #     indexes = [
    #         IndexModel(
    #             name="unique_together",
    #             keys=[
    #                 ("project", pymongo.DESCENDING),
    #                 ("targetLanguage", pymongo.DESCENDING),
    #                 ('modelVersion', pymongo.DESCENDING)
    #             ],
    #             unique=True
    #         )
    #     ]

    def __hash__(self):
        return hash(str(self.id))

    def __repr__(self):
        return f'Translation for project {self.project_id} to {self.target_language}.\n State: {self.flow_state.state.value}, Num Rows: {len(self.subtitles)}\nVersion: {self.engine_version.value.upper()}'

    __str__ = __repr__

    async def update_state(self, state: TranslationSteps, *, reason: str | None = None, save: bool = True):
        if self.flow_state is None:
            self.flow_state = TranslationState(state=state)
        else:
            self.flow_state.state = state
        if reason:
            self.flow_state.failure_reason = reason
        self.updated_at = datetime.now()
        if save:
            await self.save(ignore_revision=True)

    async def fetch_project(self) -> Project:
        """
        Fetch the project object from the database if needed
        """
        if isinstance(self.project, Link):
            await self.fetch_link('project')
        return self.project  # noqa

    def get_row(self, index) -> SRTBlock:
        index = 1 if index == 1 else index - 1
        return self.subtitles[index]

    @property
    def project_id(self) -> PydanticObjectId:
        if isinstance(self.project, Link):
            return self.project.ref.id
        return self.project.id  # noqa

    @property
    def rows_missing_translation(self) -> list[SRTBlock]:
        return [row for row in self.subtitles if not row.is_translated]

    @property
    def rows_with_translation(self) -> list[SRTBlock]:
        return [row for row in self.subtitles if row.is_translated]

    async def translations_json_blob_path(self) -> Path:
        p = await self.fetch_project()
        tl = self.target_language.value if self.target_language else self.target_language
        return p.base_blob_path / 'translations' / f'{tl}_{self.engine_version.value}.json'

    @field_validator('subtitles', mode='before')
    @classmethod
    def unique_subtitles(cls, subtitles: list[SRTBlock | dict]):
        return load_unique_subtitles(subtitles)

    @field_validator('target_language', mode='before')
    @classmethod
    def validate_target_language(cls, v: str | LanguageCode):
        if v is None:
            return
        if v.lower() == 'hebrew':
            return LanguageCode.he
        if isinstance(v, str):
            return LanguageCode(v)
        return v


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
