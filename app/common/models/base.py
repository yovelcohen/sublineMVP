from datetime import datetime

from beanie import Document, Link, PydanticObjectId
from beanie.odm.documents import json_schema_extra
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel


def document_alias_generator(s: str) -> str:
    if s == "id":
        return "_id"
    return to_camel(s)


def id_from_ref_or_obj(ref_or_obj: Link | Document | None) -> PydanticObjectId | None:
    return None if not ref_or_obj else ref_or_obj.ref.id if isinstance(ref_or_obj, Link) else ref_or_obj.id


model_config = ConfigDict(
    json_schema_extra=json_schema_extra,
    populate_by_name=True,
    alias_generator=document_alias_generator,
    validate_assignment=True
)

DEFAULT_CONFIG = model_config


class BaseCreateUpdateDocument(Document):
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = model_config

    async def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return await super().save(*args, **kwargs)
