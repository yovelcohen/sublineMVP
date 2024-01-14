import asyncio
from datetime import timedelta

import beanie.exceptions
import pandas as pd
from beanie import PydanticObjectId
from pydantic import BaseModel, Field

import streamlit as st

from common.config import mongodb_settings
from common.db import init_db
from common.models.core import Project, Client
from common.models.translation import CostRecord, Translation, CostsConfig


class TranslationProjection(BaseModel):
    id: PydanticObjectId = Field(..., alias='_id')
    name: str


def format_length(delta: timedelta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


format_multiplied = lambda x, y: f'{x * y:.2f}'  # noqa


def get_root_form(num):
    if 0 <= num <= 50000:
        root_form = num / 1000.0
        return round(root_form, 3)
    else:
        return "Input out of range (0 to 50,000)"


async def _costs_panel():
    async def get_costs():
        return await CostRecord.find_all(ignore_cache=True).to_list()

    try:
        costs = await get_costs()
    except beanie.exceptions.CollectionWasNotInitialized as e:
        await init_db(mongodb_settings, [CostRecord, Translation, Project, Client])
        costs = await get_costs()

    costs: list[CostRecord]

    names = {proj.id: proj.name for proj in await Project.find_all().to_list()}
    id_to_translation: dict[PydanticObjectId, Translation] = {
        tr.id: tr for tr in await Translation.find_all().to_list()
    }
    data = []

    for record in costs:
        if record.translation_id in id_to_translation:
            rec_costs = record.costs
            di = {
                'Name': names[id_to_translation[record.translation_id].project_id],
                "Length": format_length(id_to_translation[record.translation_id].length),
                'Num Rows': len(id_to_translation[record.translation_id].subtitles),
                'Total Cost': get_root_form(record.total_cost()),
                'OpenAI Input Cost (tokens)': f'{format_multiplied(rec_costs.openai_input_token, CostsConfig.openai_input_token)} (t: {record.costs.openai_input_token})',
                'OpenAI Output Cost (tokens)': f'{format_multiplied(rec_costs.openai_completion_token, CostsConfig.openai_completion_token)} (t: {record.costs.openai_completion_token})',
            }
            if rec_costs.assembly_ai_second != 0:
                di['Assembly AI Cost (seconds)'] = f'{format_multiplied(rec_costs.assembly_ai_second, CostsConfig.assembly_ai_second)} (t: {record.costs.assembly_ai_second})'
            if rec_costs.deepgram_minutes != 0:
                di['Deepgram Cost (minutes)'] = f'{format_multiplied(rec_costs.deepgram_minutes, CostsConfig.deepgram_minute)} (r: {record.costs.deepgram_minutes})'

            data.append(di)

    df = pd.DataFrame(data)
    st.data_editor(df, disabled=True, use_container_width=True)


def costs_panel():
    return asyncio.run(_costs_panel())
