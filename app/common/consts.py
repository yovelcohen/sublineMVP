from typing import NewType, Final, Literal

XMLString = str | bytes
LanguageCode = NewType('LanguageCode', str)
SrtString = NewType('SrtString', str)
JsonStr = NewType('JsonStr', str)
VTTString = NewType('VTTString', str)

MALE: Final[str] = 'male'
FEMALE: Final[str] = 'female'
Gender = Literal[MALE, FEMALE]  # type: ignore
