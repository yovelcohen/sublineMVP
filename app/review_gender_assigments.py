import pandas as pd

from app.common.models.translation import Translation
from app.common.utils import download_azure_blob


async def review(t: Translation):
    rows = t.subtitles
    path = f'{t.project_id}/misc/gender/{t.id}.wav'
    samples = await download_azure_blob('projects', path)
    df = pd.DataFrame([row.model_dump(include={'index', 'content', 'start', 'end'}) for row in rows])
    df['start'] = df['start'].apply(lambda x: str(x) if x else x)
    df['end'] = df['end'].apply(lambda x: str(x) if x else x)
    return df, samples
