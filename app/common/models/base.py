from datetime import datetime
from typing import Any

from beanie import Document, Link
from beanie.odm.documents import json_schema_extra
from pydantic import ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


def document_alias_generator(s: str) -> str:
    if s == "id":
        return "_id"
    return to_camel(s)


class BaseDocument(Document):
    model_config = ConfigDict(
        json_schema_extra=json_schema_extra,
        populate_by_name=True,
        alias_generator=document_alias_generator,
        validate_assignment=True
    )


class BaseCreateUpdateDocument(BaseDocument):
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    async def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return await super().save(*args, **kwargs)
