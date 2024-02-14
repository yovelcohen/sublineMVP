import logging
from dataclasses import dataclass, asdict

from beanie import Document, Link
from pydantic import create_model, ConfigDict

from common.models.base import document_alias_generator
from common.models.translation import Translation


def _init_costs_config() -> dataclass:
    @dataclass(frozen=True, slots=True, repr=True)
    class _CostsConfig:
        deepgram_minute: float = 0.0043
        deepgram_summarization: float = 0.0044
        openai_input_token: float = 0.01
        openai_completion_token: float = 0.03
        openai_gpt3_input_token: float = 0.0010
        openai_gpt3_completion_token: float = 0.0020
        anthropic_input_tokens: float = 0.008
        anthropic_completion_tokens: float = 0.024
        anthropic_instant_input_tokens: float = 0.0008
        anthropic_instant_completion_tokens: float = 0.0024
        assembly_ai_seconds: float = 0.0001028
        lemur_input_tokens: float = 0.015
        lemur_output_tokens: float = 0.043
        smodin_suggestion_word_cost: float = 0.0001

    return _CostsConfig()


CostsConfig = _init_costs_config()


def __costeq__(self, other):
    self_dict: dict = self.model_dump()
    other_dict: dict = other.model_dump()
    return all([self_dict[k] == other_dict.get(k, 0.0) for k in self_dict.keys()])


def __costiadd__(self, other):
    self_dict: dict = self.model_dump()
    other_dict: dict = other.model_dump()
    for k in self_dict.keys():
        self_dict[k] += other_dict.get(k, 0.0)
    return self.__class__(**self_dict)


def units_of_1k(number: int | float) -> int | float:
    if number < 1000:  # less than 1000, get reminder, not billed, but shown for clarity.
        return number / 1000
    units = number // 1000
    if number % 1000 > 500:
        units += 1
    return units


Costs = create_model(
    __model_name='Costs', **{k: (float, 0) for k in asdict(CostsConfig).keys() if not k.startswith('_')}
)
Costs.__eq__ = __costeq__
Costs.__iadd__ = __costiadd__
Costs.units_of_1k = staticmethod(units_of_1k)


class CostRecord(Document):
    translation: Link[Translation]
    costs: Costs

    def total_cost(self):
        return sum([Costs.units_of_1k(v) * getattr(CostsConfig, k) for k, v in self.costs.model_dump().items()])

    @property
    def translation_id(self):
        if isinstance(self.translation, Link):
            return self.translation.ref.id
        return self.translation.id
