from datetime import datetime
from enum import Enum

import pymongo
from beanie import Document, PydanticObjectId, Link, Indexed
from pydantic import Field, BaseModel, model_validator, ConfigDict
from pymongo import IndexModel, DESCENDING

from common.consts import LanguageCode  # noqa
from common.models.base import document_alias_generator, DEFAULT_CONFIG, id_from_ref_or_obj  # noqa


class ClientTypes(str, Enum):
    PRO = 'pro'
    EXPERTS = 'experts'
    ENTERPRISE = 'enterprise'


class ClientSpecification(BaseModel):
    client_type: ClientTypes = ClientTypes.PRO


class Client(Document):
    model_config = DEFAULT_CONFIG
    name: Indexed(str, unique=True)
    slug: str | None = None
    client_spec: ClientSpecification = Field(default_factory=ClientSpecification)
    created_at: datetime = Field(default_factory=datetime.now)


class ClientChannel(Document):
    model_config = DEFAULT_CONFIG

    name: str
    email: str = ''
    phone: str = ''
    client: Link[Client]

    class Settings:
        indexes = [
            IndexModel(name='client_name_unique',
                       keys=[('client', DESCENDING), ('name', DESCENDING)],
                       unique=True)
        ]


class ProjectTypes(str, Enum):
    MOVIE = 'movie'
    SERIES = 'series'


class Ages(int, Enum):
    ZERO = 0
    THREE = 3
    SIX = 6
    TWELVE = 12
    SIXTEEN = 16
    EIGHTEEN = 18

    NOOP = ZERO


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


class ProjectDescription(BaseModel):
    model_config = ConfigDict(alias_generator=document_alias_generator, populate_by_name=True)

    age: Ages | None = Ages.ZERO
    main_genre: Genres | None = Genres.NOOP
    additional_genres: list[Genres] | None = Field(default_factory=list)
    user_description: str | None = None

    @model_validator(mode='before')
    @classmethod
    def validate_fields(cls, data: dict):
        if not data:
            return data
        for k in ('age', 'main_genre'):
            data[k] = Ages.ZERO if k == 'age' else Genres.NOOP
            if k in data and isinstance(data[k], str):
                _cls = Ages if k == 'age' else Genres
                data[k] = _cls(data[k])
        if data.get('additional_genres', None) is None:
            data['additional_genres'] = list()
        if 'additional_genres' in data and isinstance(data['additional_genres'], list):
            data['additional_genres'] = [Genres(g) for g in data['additional_genres']]
        return data


class ProjectMediaMetaData(BaseModel):
    mime_type: str | None = Field(default=None, description='Video mime type')
    audio_mime_type: str | None = Field(default=None, description='Audio mime type')
    original_subs_mime_type: str | None = Field(default=None, description='Original subtitles mime type')
    season: int | None = None
    episode: int | None = None

    video_duration: float | None = Field(default=None, description='Video duration in seconds')
    amount_rows: int | None = Field(default=None, description='Amount of rows in the original subtitles')

    model_config = ConfigDict(alias_generator=document_alias_generator, populate_by_name=True)


class BaseProject(BaseModel):
    name: str
    source_language: LanguageCode | None = None
    type: ProjectTypes | None = None
    description: ProjectDescription | None = Field(default_factory=lambda: ProjectDescription())


class Project(Document, BaseProject):
    model_config = {**DEFAULT_CONFIG, 'use_enum_values': True}

    # TODO: Delete these fields
    media_meta: ProjectMediaMetaData = Field(default_factory=lambda: ProjectMediaMetaData())

    client: Link[Client]
    channel: Link[ClientChannel] | None = None

    parent: Link['Project'] | None = None
    path: str | None = None

    uploader_id: PydanticObjectId | None = None
    allowed_editors: list[PydanticObjectId] = Field(
        default_factory=list, description="List of user ids that are allowed to edit this request_data"
    )

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # translations: list[NestedTranslation] = Field(default_factory=list)

    class Settings:
        indexes = [
            # IndexModel(
            #     name="unique_together",
            #     keys=[
            #         ("client", pymongo.DESCENDING),
            #         ('parent', pymongo.DESCENDING),
            #         ("name", pymongo.DESCENDING),
            #         ("source_language", pymongo.DESCENDING),
            #     ],
            #     unique=True,
            # ),
            IndexModel(
                keys=[("name", pymongo.DESCENDING)],
                name="name_index_DESCENDING"
            )
        ]

    def __repr__(self):
        base = f'<Project {self.id} {self.name}'
        if self.type == ProjectTypes.SERIES:
            if self.media_meta.season:
                base += f' S{self.media_meta.season}'
            if self.media_meta.episode:
                base += f' E{self.media_meta.episode}'
        base += ' Last updated at: ' + str(self.updated_at)
        return base + '>'

    @property
    def parent_id(self) -> PydanticObjectId | None:
        return id_from_ref_or_obj(self.parent)

    @property
    def client_id(self) -> PydanticObjectId:
        return id_from_ref_or_obj(self.client)

    @property
    def channel_id(self) -> PydanticObjectId | None:
        return id_from_ref_or_obj(self.channel)

    @property
    def is_root(self) -> bool:
        return not self.project.parent
