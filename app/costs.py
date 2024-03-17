import asyncio
from datetime import timedelta
from typing import Callable

import beanie.exceptions
import pandas as pd
from beanie import PydanticObjectId
from pydantic import BaseModel, Field

import streamlit as st

from common.config import mongodb_settings
from common.db import init_db
from common.models.costs import CostsInfo


class TranslationProjection(BaseModel):
    id: PydanticObjectId = Field(..., alias='_id')
    name: str


def format_length(delta: timedelta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


format_multiplied: Callable[[float, float], str] = lambda x, y: f'{round((x * y), 5)}'


def get_root_form(num):
    if 0 <= num <= 50000:
        root_form = num / 1000.0
        return round(root_form, 3)
    else:
        return "Input out of range (0 to 50,000)"


async def _costs_panel():
    async def get_costs() -> list[CostsInfo]:
        return await CostsInfo.find_all(ignore_cache=True).to_list()

    try:
        costs = await get_costs()
    except beanie.exceptions.CollectionWasNotInitialized as e:
        await init_db(mongodb_settings, [CostsInfo])
        costs = await get_costs()

    data = [
        {'Name': info.project_name, 'Duration': format_length(timedelta(seconds=info.video_duration)),
         'Total Cost': round(info.calculate_total_cost(), 3),
         'Language': info.target_language.value, 'Model Version': info.engine_version.value}
        for info in costs
    ]

    df = pd.DataFrame(data)
    st.data_editor(df, disabled=True, use_container_width=True)
    st.divider()
    st.subheader('Detailed Breakdown')

    detailed_data = [
        {
            'Name': info.project_name, 'Duration': format_length(timedelta(seconds=info.video_duration)),
            'GPT 4 Input Tokens Cost': info.calculate_cost_by_field_name('gpt4_input_token'),
            'GPT 4 Completion Tokens Cost': info.calculate_cost_by_field_name('gpt4_completion_token'),
            'Deepgram Cost': info.calculate_cost_by_field_name('deepgram_minute'),
            'Gender Recognition Cost': info.calculate_cost_by_field_name('presentid_rapidapi'),
            'GPT 3 Input Tokens Cost': info.calculate_cost_by_field_name('gpt3_input_token'),
            'GPT 3 Completion Tokens Cost': info.calculate_cost_by_field_name('gpt3_completion_token')
        }
        for info in costs
    ]
    detailed_df = pd.DataFrame(detailed_data)
    st.data_editor(detailed_df, disabled=True, use_container_width=True, key='detailedview')


def costs_panel():
    return asyncio.run(_costs_panel())
