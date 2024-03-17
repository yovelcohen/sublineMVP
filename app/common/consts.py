from enum import Enum
from typing import Final, Literal, TypeVar, TypeAlias, Annotated

from annotated_types import Predicate

XMLString = TypeVar('XMLString', bound=str | bytes)
SrtString = TypeVar('SrtString', bound=str)
JsonStr = TypeVar('JsonStr', bound=str)
YamlStr = TypeVar('YamlStr', bound=str)
VTTString = TypeVar('VTTString', bound=str)

MALE: Final[str] = 'male'
FEMALE: Final[str] = 'female'
Gender = Literal[MALE, FEMALE]  # type: ignore

RowIndex: TypeAlias = int
RowIndexJson: TypeAlias = str  # represents the index when it returns from a json
VSymbolType: TypeAlias = str
SentenceType: TypeAlias = str
SimilarityScore: TypeAlias = Annotated[float, Predicate(lambda x: -1 <= x <= 1.1)]
EnglishSentenceType = Annotated[SentenceType, "English"]


class AllowedSourceLanguages(str, Enum):
    """
    Supported languages for Transcribing Audio
    """

    de = "de"
    "German"

    en = "en"
    "Global English"

    en_au = "en_au"
    "Australian English"

    en_uk = "en_GB"
    "British English"

    en_us = "en_us"
    "English (US)"

    es = "es"
    "Spanish"

    fi = "fi"
    "Finnish"

    fr = "fr"
    "French"

    hi = "hi"
    "Hindi"

    it = "it"
    "Italian"

    ja = "ja"
    "Japanese"

    ko = "ko"
    "Korean"

    nl = "nl"
    "Dutch"

    pl = "pl"
    "Polish"

    pt = "pt"
    "Portuguese"

    ru = "ru"
    "Russian"

    tr = "tr"
    "Turkish"

    uk = "uk"
    "Ukrainian"

    vi = "vi"
    "Vietnamese"

    zh = "zh"
    "Chinese"

    he = "he"
    "Hebrew"

    HEBREW = he


LanguageCode = AllowedSourceLanguages

TEN_K, MILLION, BILLION = 10_000, 1_000_000, 1_000_000_000
TWO_HOURS = 60 * 60 * 2
THREE_MINUTES = 60 * 3


class AllowedLanguagesForTranslation(str, Enum):
    fr = 'fr'
    es = 'es'
    ru = 'ru'
    he = 'he'
    en = 'en_us'
