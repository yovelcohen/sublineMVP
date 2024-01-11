from enum import Enum
from typing import Self

from beanie import Document, Link, PydanticObjectId
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class UserAccess(BaseModel):
    model_config = ConfigDict(extra='allow', use_enum_values=True)

    disabled: bool = False
    is_admin: bool = False
    can_edit: bool = False
    can_view: bool = True

    class Levels(str, Enum):
        ADMIN = 'admin'
        EDITOR = 'editor'
        VIEWER = 'viewer'

    def __eq__(self, other: Self) -> bool:
        return (self.disabled == other.disabled and
                self.is_admin == other.is_admin and
                self.can_edit == other.can_edit and
                self.can_view == other.can_view)

    @property
    def is_editor(self):
        return self.is_admin or self.can_edit

    def set(self, level: Levels):
        if level == self.Levels.ADMIN:
            self.is_admin = True
            self.can_edit = True
            self.can_view = True
        elif level == self.Levels.EDITOR:
            self.is_admin = False
            self.can_edit = True
            self.can_view = True
        elif level == self.Levels.VIEWER:
            self.is_admin = False
            self.can_edit = False
            self.can_view = True


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Document):
    username: str
    first_name: str
    last_name: str
    email: str | None = None
    hashed_password: str
    client: Link['Client']
    access: UserAccess = Field(default_factory=UserAccess)
    avatar_blob_path: str | None = None

    @classmethod
    async def create_user(
            cls, *,
            first_name: str,
            last_name: str,
            email: str,
            username: str,
            password: str,
            client: 'Client',  # noqa
            access: UserAccess | None = None
    ) -> Self:
        access = access or UserAccess()
        if isinstance(client, Link):
            client = client.fetch()
        try:
            obj = cls(username=username, hashed_password=cls.get_password_hash(password), client=client.model_dump(),
                      access=access, first_name=first_name, last_name=last_name, email=email, id=PydanticObjectId())
        except ValidationError as e:
            raise e
        await obj.insert()
        return obj

    @classmethod
    def get_password_hash(cls, password):
        return pwd_context.hash(password)
