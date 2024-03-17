import datetime
import logging
from types import MappingProxyType

import pymongo
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import Literal, TypeAlias

from beanie import Document, PydanticObjectId
from pydantic import create_model, Field, ConfigDict
from pymongo import IndexModel

from common.models.core import Project
from common.consts import AllowedLanguagesForTranslation
from common.models.consts import ModelVersions
from common.models.base import document_alias_generator
from common.models.translation import Translation

granularity_serializers = {
    'per_1000': lambda amount, cost: (amount / 1000) * cost,
    'second': lambda amount, cost: amount * cost,
    'minute': lambda amount, cost: (amount / 60) * cost,
    '1': lambda amount, cost: amount * cost
}

BasicConfig = ConfigDict(
    alias_generator=document_alias_generator,
    arbitrary_types_allowed=True,
    populate_by_name=True
)


@pydantic_dataclass(config=BasicConfig, frozen=True, slots=True, repr=True)
class CostField:
    name: str
    price_per_unit: float
    unit: Literal['token', 'time_in_seconds', 'plain_amount']
    price_granularity: Literal['per_1000', 'minute', 'second', '1']

    def __post_init__(self):
        if self.unit == 'token' and self.price_granularity != 'per_1000':
            raise ValueError("token as unit must have price_granularity as per_1000, not time")

    def compute_cost(self, amount):
        return granularity_serializers[self.price_granularity](amount, self.price_per_unit)


CostFieldName: TypeAlias = str

_FrozenCostsConfigV2: dict[str, CostField] = dict(
    deepgram_minute=CostField(
        name='deepgram_minute',
        price_per_unit=0.0043,
        unit='time_in_seconds',
        price_granularity='minute'
    ),
    gpt4_input_token=CostField(
        name='gpt4_input_token',
        price_per_unit=0.01,
        unit='token',
        price_granularity='per_1000'
    ),
    gpt4_completion_token=CostField(
        name='gpt4_completion_token',
        price_per_unit=0.03,
        unit='token',
        price_granularity='per_1000'
    ),
    gpt3_input_token=CostField(
        name='gpt3_input_token',
        price_per_unit=0.001,
        unit='token',
        price_granularity='per_1000'
    ),
    gpt3_completion_token=CostField(
        name='gpt3_completion_token',
        price_per_unit=0.002,
        unit='token',
        price_granularity='per_1000'
    ),
    presentid_rapidapi=CostField(
        name='presentid_rapidapi',
        price_per_unit=0.01,
        unit='plain_amount',
        price_granularity='per_1000'
    )
)

CostsConfigV2 = MappingProxyType(_FrozenCostsConfigV2)  # ensures a frozen dict


def __costeq__(self, other):
    self_dict, other_dict = self.model_dump(), other.model_dump()
    return all([self_dict[k] == other_dict.get(k, 0.0) for k in self_dict])


def __costiadd__(self, other):
    self_dict, other_dict = self.model_dump(), other.model_dump()
    for k in self_dict:
        self_dict[k] += other_dict.get(k, 0.0)
    return self.__class__(**self_dict)


Costs = create_model('Costs', **{k: (float, 0) for k in CostsConfigV2.keys()})
Costs.__eq__ = __costeq__
Costs.__iadd__ = __costiadd__


class CostsInfo(Document):
    project_name: str
    project_id: PydanticObjectId
    video_duration: float
    target_language: AllowedLanguagesForTranslation
    engine_version: ModelVersions
    costs: Costs
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    model_config = BasicConfig

    class Settings:
        indexes = [
            IndexModel(
                name="pid_tl_unique_index",
                keys=[
                    ("projectId", pymongo.ASCENDING),
                    ("targetLanguage", pymongo.ASCENDING),
                    ("modelVersion", pymongo.ASCENDING)
                ],
                unique=True
            )
        ]

    def calculate_total_cost(self) -> float | int:
        sum_ = 0
        for k, v in self.costs.model_dump().items():
            sum_ += CostsConfigV2[k].compute_cost(v)
        return sum_

    def calculate_cost_by_field_name(self, field_name: CostFieldName) -> int | float:
        conf = CostsConfigV2[field_name]
        value = getattr(self.costs, field_name)
        return conf.compute_cost(value)


class CostCatcher:

    def __init__(self, translation_obj: Translation, project: Project, reset_existing: bool = False):
        self.translation_obj = translation_obj
        self.project = project
        self._costs: Costs = Costs()
        self.reset_existing = reset_existing
        self.obj: CostsInfo | None = None

    @property
    def costs(self) -> Costs:
        return self._costs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> CostsInfo:
        obj = await CostsInfo.find(
            CostsInfo.project_id == self.translation_obj.project_id,
            CostsInfo.target_language == self.translation_obj.target_language,
            CostsInfo.engine_version == self.translation_obj.engine_version
        ).first_or_none()  # noqa

        if obj and self.reset_existing:
            obj.costs = self.costs
        elif obj and not self.reset_existing:
            obj.costs += self.costs
            obj = await obj.save(ignore_revision=True)
        else:
            obj = CostsInfo(
                costs=self.costs, project_name=self.project.name,
                target_language=self.translation_obj.target_language,
                project_id=self.translation_obj.project_id,
                video_duration=self.project.media_meta.video_duration,
                engine_version=self.translation_obj.engine_version
            )
            await obj.save()

        self.obj = obj
        logging.info(f'Costs for Translation: {self.translation_obj.id} -- {self._costs.model_dump_json()}')
        logging.info(f'Total cost: {self.obj.calculate_total_cost()}')
        return self.obj

    def update_openai_stats(
            self,
            input_tokens: int | float = 0,
            completion_tokens: int | float = 0,
            is_gpt3: bool = False
    ):
        if is_gpt3:
            self._costs.openai_gpt3_input_token += input_tokens
            self._costs.openai_gpt3_completion_token += completion_tokens
        else:
            self._costs.gpt4_input_token += input_tokens
            self._costs.gpt4_completion_token += completion_tokens

    def update_deepgram_stats(self, minutes: int | float = 0):
        self._costs.deepgram_minute += minutes

    def update_assembly_ai_seconds(self, seconds: int | float):
        self._costs.assembly_ai_seconds += seconds

    def update_assembly_ai_tokens(self, input_tokens: int | float = 0, completion_tokens: int | float = 0):
        self._costs.lemur_input_tokens += input_tokens
        self._costs.lemur_output_tokens += completion_tokens

    def update_antrhopic_stats(self, input_tokens: int | float = 0, completion_tokens: int | float = 0):
        self._costs.antrophic_input_tokens += input_tokens
        self._costs.antrophic_completion_tokens += completion_tokens

    def update_gender_api_stats(self, amount_requests: int):
        self._costs.presentid_rapidapi += amount_requests
