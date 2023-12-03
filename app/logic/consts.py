from typing import NewType

XMLString = str | bytes
SrtString = NewType('SrtString', str)
JsonStr = NewType('JsonStr', str)
LanguageCode = NewType('LanguageCode', str)
Version = NewType('Version', int)
