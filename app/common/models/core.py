from datetime import datetime
from enum import Enum
from pathlib import Path

import pymongo
from beanie import Document, PydanticObjectId, Link
from pydantic import Field, BaseModel, model_validator, ConfigDict
from pymongo import IndexModel

from common.models.base import BaseDocument, document_alias_generator


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


class ClientChannel(Document):
    model_config = BaseDocument.model_config

    name: str
    email: str = ''
    phone: str = ''
    client: Link[Client]

    class Settings:
        indexes = [
            IndexModel(name='client_name_unique',
                       keys=[('client', pymongo.DESCENDING), ('name', pymongo.DESCENDING)],
                       unique=True)
        ]


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


class Project(Document):
    model_config = {**BaseDocument.model_config, 'use_enum_values': True}

    name: str
    source_language: str | None

    description: ProjectDescription | None = Field(default_factory=lambda: ProjectDescription())

    type: ProjectTypes | None = None
    season: int | None = None
    episode: int | None = None
    mime_type: str | None = None

    client: Link[Client]
    parent: Link['Project'] | None = None
    channel: Link[ClientChannel] | None = None

    uploader_id: PydanticObjectId | None = None
    allowed_editors: list[PydanticObjectId] = Field(
        default_factory=list, description="List of user ids that are allowed to edit this request_data"
    )

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        indexes = [
            # IndexModel(
            #     name="unique_together",
            #     keys=[
            #         ("client", pymongo.DESCENDING),
            #         ('parent', pymongo.DESCENDING),
            #         ("name", pymongo.DESCENDING),
            #         ("type", pymongo.DESCENDING),
            #         ("season", pymongo.DESCENDING),
            #         ("episode", pymongo.DESCENDING),
            #     ],
            #     unique=True,
            # ),
            IndexModel(
                keys=[("name", pymongo.DESCENDING)],
                name="name_index_DESCENDING"
            )
        ]

    @property
    def is_root(self) -> bool:
        return not self.parent

    @property
    def qualifying_name(self) -> str:
        base = self.name
        if self.type == ProjectTypes.SERIES:
            base += f' S{self.season}E{self.episode}'
        return base

    @property
    def base_blob_path(self) -> Path:
        return Path(str(self.id))

    @property
    def video_blob_path(self) -> Path:
        return self.base_blob_path / 'video' / f'video.{self.mime_type}'

    @property
    def audio_blob_path(self) -> Path:
        return self.base_blob_path / 'audio' / 'audio.wav'
