from __future__ import annotations
from enum import Enum

from beanie import Document, Link
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, Field

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


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
    client: Link["Client"] # noqa
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
            client,
            access: UserAccess | None = None
    ):
        access = access or UserAccess()
        obj = cls(username=username, hashed_password=cls.get_password_hash(password), client=client, access=access,
                  first_name=first_name, last_name=last_name, email=email)
        await obj.insert()
        return obj

    @classmethod
    def verify_password(cls, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    @classmethod
    def get_password_hash(cls, password):
        return pwd_context.hash(password)

    @classmethod
    async def get_user(cls, username: str):
        return await cls.find(User.username == username).first_or_none()

    @classmethod
    async def authenticate_user(cls, *, username: str, password: str):
        user = await cls.get_user(username)
        if not user:
            return False
        if not cls.verify_password(password, user.hashed_password):
            return False
        return user