from typing import NewType

XMLString = str | bytes
LanguageCode = NewType('LanguageCode', str)
SrtString = NewType('SrtString', str)
JsonStr = NewType('JsonStr', str)
VTTString = NewType('VTTString', str)