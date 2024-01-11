import asyncio

import beanie.exceptions
import pandas as pd
from beanie import PydanticObjectId
from pydantic import BaseModel, Field

import streamlit as st

from common.config import mongodb_settings
from common.db import init_db
from common.models.core import Project, Client
from common.models.translation import CostRecord, Translation


class TranslationProjection(BaseModel):
    id: PydanticObjectId = Field(..., alias='_id')
    name: str


async def _costs_panel():
    async def get_costs():
        return await CostRecord.find_all().to_list()

    try:
        costs = await get_costs()
    except beanie.exceptions.CollectionWasNotInitialized as e:
        await init_db(mongodb_settings, [CostRecord, Translation, Project, Client])
        costs = await get_costs()

    costs: list[CostRecord]

    names = {proj.id: proj.name for proj in await Project.find_all().to_list()}
    tr_to_proj = {tr.id: tr.project.ref.id for tr in await Translation.find_all().to_list()}
    rows_counts_q = await Translation.find_all().aggregate(
        [{"$project": {"amountSubtitles": {"$size": "$subtitles"}}}]
    ).to_list()
    rows_counts = dict()
    for tr in rows_counts_q:
        rows_counts[tr['_id']] = tr['amountSubtitles']

    df = pd.DataFrame(
        [
            {
                'Name': names[tr_to_proj[record.translation.ref.id]],
                'Num Rows': rows_counts[record.translation.ref.id],
                'Total Cost': record.total_cost(),
                'Deepgram Minutes': record.costs.deepgram_minutes,
                'OpenAI Input Tokens': record.costs.openai_input_tokens,
                'OpenAI Output Tokens': record.costs.openai_completion_token,
            }
            for record in costs if record.translation.ref.id in tr_to_proj
        ]
    )
    st.data_editor(df, disabled=True)


def costs_panel():
    st.warning('Not ready yet bitch')
    return asyncio.run(_costs_panel())
