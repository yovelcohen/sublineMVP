import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.common.models.translation import TranslationFeedbackV2, MarkedRow


async def load_from_file(file: UploadedFile, feedback: TranslationFeedbackV2 | None = None):
    df = pd.read_csv(file)
    rows = df.to_dict(orient='records')
    en_key = [col for col in df.columns if 'english' in col.lower()][0]

    if not feedback:
        feedback = TranslationFeedbackV2(
            marked_rows=[
                MarkedRow(original=rows[i][en_key], translation=rows[i]['en_key'])
                for i in range(len(rows))
            ]
        )
