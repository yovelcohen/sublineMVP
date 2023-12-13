from datetime import datetime, timedelta
from enum import Enum
from typing import TypedDict, Self

import pymongo
from beanie import Document, PydanticObjectId
from beanie.odm.operators.find.logical import Or
from pydantic import Field, BaseModel, model_validator
from pymongo import IndexModel

from srt import make_legal_content

from common.models.base import BaseDocument, BaseCreateUpdateDocument

SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60
HOURS_IN_DAY = 24
MICROSECONDS_IN_MILLISECOND = 1000


def timedelta_to_srt_timestamp(timedelta_timestamp):
    r"""
    Convert a :py:class:`~datetime.timedelta` to an SRT timestamp.

    .. doctest::

        >>> import datetime
        >>> delta = datetime.timedelta(hours=1, minutes=23, seconds=4)
        >>> timedelta_to_srt_timestamp(delta)
        '01:23:04,000'

    :param datetime.timedelta timedelta_timestamp: A datetime to convert to an
                                                   SRT timestamp
    :returns: The timestamp in SRT format
    :rtype: str
    """

    hrs, secs_remainder = divmod(timedelta_timestamp.seconds, SECONDS_IN_HOUR)
    hrs += timedelta_timestamp.days * HOURS_IN_DAY
    mins, secs = divmod(secs_remainder, SECONDS_IN_MINUTE)
    msecs = timedelta_timestamp.microseconds // MICROSECONDS_IN_MILLISECOND
    return "%02d:%02d:%02d,%03d" % (hrs, mins, secs, msecs)


class ClientTypes(str, Enum):
    PRO = 'pro'
    EXPERTS = 'experts'
    ENTERPRISE = 'enterprise'


class ClientSpecification(BaseModel):
    client_type: ClientTypes = ClientTypes.PRO


class Client(Document):
    name: str
    slug: str | None = None
    client_spec: ClientSpecification = Field(default_factory=ClientSpecification)
    created_at: datetime = Field(default_factory=datetime.now)


class ProjectTypes(str, Enum):
    MOVIE = 'movie'
    SERIES = 'series'


class BaseProject(BaseDocument):
    model_config = {**BaseDocument.model_config, 'use_enum_values': True}
    name: str
    type: ProjectTypes
    description: str | None = None
    source_language: str

    season: int | None = None
    episode: int | None = None

    class Settings:
        is_root = True

    @property
    def blob_path(self) -> str:
        base = f'{self.type}/{self.name}/'
        if self.type == ProjectTypes.SERIES:
            base += f's{self.season}e{self.episode}/'
        return base


class Project(BaseProject):
    uploader_id: str | PydanticObjectId
    client: Client
    allowed_editors: list[str | PydanticObjectId] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        indexes = [
            "unique_together",
            [
                ("client", pymongo.DESCENDING),
                ("name", pymongo.DESCENDING),
                ("type", pymongo.DESCENDING),
            ],
            IndexModel(
                [("unique_name", pymongo.DESCENDING)],
                name="unique_name_index_DESCENDING",
            ),
        ]

    @classmethod
    async def get_authorized_project(cls, project_id, user_id) -> Self | None:
        query = {
            '_id': project_id,
            '$or': [{'uploader_id': user_id}, {'allowed_editors': user_id}]
        }
        return await cls.find_one(Or(Project.uploader_id == user_id, Project.allowed_editors == user_id))


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
        return hash(str(self.index) + str(self.content))

    def __eq__(self, other):
        return self.index == other.index and self.content == other.content

    def __lt__(self, other):
        return (self.start, self.end, self.index) < (other.start, other.end, other.index)

    def __repr__(self):
        return f'SRT Block No. {self.index}\nContent: {self.content}'


class TranslationContent(BaseModel):
    content: str
    revision: str | None = None

    def get(self, revision: bool = False):
        if revision:
            return self.revision or self.content
        return self.content


class SRTRowDict(TypedDict):
    index: int | str
    start: str
    end: str
    text: str
    translations: TranslationContent | None
    speaker: int | None


class SRTBlock(Subtitle):
    index: int | str
    start: timedelta
    end: timedelta
    content: str
    speaker: int | None = None
    region: str | None = None
    style: str | None = None
    num_tokens: int | None = None
    translations: TranslationContent | None = None

    def __hash__(self):
        return hash(str(self.index) + str(self.content))

    def is_translated(self, is_revision: bool = False):
        if not is_revision:
            return self.translations is not None and self.translations.content
        return self.translations is not None and self.translations.revision

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
                          end=timedelta_to_srt_timestamp(self.end), text=self.content, translations=self.translations)

    def to_srt(self, *, strict=True, eol="\n", target_language=None, revision: bool = False):
        r"""
        Convert the current :py:class:`Subtitle` to an SRT block.

        :param bool strict: If disabled, will allow blank lines in the content
                            of the SRT block, which is a violation of the SRT
                            standard and may cause your media player to explode
        :param str eol: The end of line string to use (default "\\n")
        :param target_language: if provided, tries to get the content string by language in self.translations_result
        :param revision: if True, will return the revision content

        :returns: The metadata of the current :py:class:`Subtitle` object as an
                  SRT formatted subtitle block
        :rtype: str
        """
        if target_language:
            output_content = self.translations.get(revision=revision)
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


class TranslationStates(str, Enum):
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    IN_REVISION = 'in_revision'
    DONE = 'done'
    FAILED = 'failed'


class Translation(BaseCreateUpdateDocument):
    project_id: PydanticObjectId
    target_language: str
    subtitles: set[SRTBlock]  # map from index to SRTBlock
    state: TranslationStates = TranslationStates.PENDING
    tokens_cost: dict = Field(default_factory=dict)

    class Settings:
        indexes = ["orderSubtitles", [("subtitles.index", pymongo.ASCENDING)]]
        is_root = True

    def __repr__(self):
        return f'Translation for project {self.project_id} to {self.target_language}. \n State: {self.state.value}, Num Rows: {len(self.subtitles)}'

    @property
    def task_id(self):
        return self.id

    def get_blob_path(self, project: Project, file_mime):
        return project.blob_path + f'{self.target_language}.{file_mime}'

    async def get_subtitles(self, offset, limit) -> list[SRTBlock]:
        raise NotImplementedError
        # pipeline = [
        #     {'$match': {'task_id': self.task_id}},
        #     {'$unwind': '$subtitles'},
        #     {'$sort': {'subtitles.index': 1}},
        #     {'$group': {'_id': '$_id', 'subtitles': {'$push': '$subtitles'}}},
        #     {'$project': {'subtitlesChunk': {'$slice': ['$subtitles', offset, limit]}}}
        # ]
        # ret = await self.aggregate(pipeline).to_list()
        # return [SRTBlock(**di['v']) for di in ret[0]['subtitlesChunk']]