from datetime import timedelta
from typing import cast, Literal
import xml.etree.ElementTree as ET

from logic.constructor import SRTTranslator, SRTBlock
from logic.consts import LanguageCode, XMLString


def parse_ttml_timestamp(timestamp_str):
    milliseconds_str = timestamp_str.rstrip('t')
    milliseconds = int(milliseconds_str)
    return timedelta(milliseconds=milliseconds)


def replace_texts(text_to_translation: dict[str, str], parent_map: dict[str, ET.Element]) -> None:
    """
    Replace the original texts in the XML with translated texts.

    :param text_to_translation: mapping from text to its translation
    :param parent_map: Map containing the relationship between original texts and their parent elements.
    """
    parent_map = {k.strip(): v for k, v in parent_map.items()}
    for original_text, translated_text in text_to_translation.items():
        parent_map[original_text].text = translated_text


def extract_texts(*, element: ET.Element, elements: list, parent_map: dict[str, ET.Element]) -> None:
    """
    Recursively extract text from an XML element and its children.

    :param element: The current XML element.
    :param elements: list to store the extracted Element objects.
    :param parent_map: Map to store the relationship between texts and their parent elements.
    """
    if element.text and element.text.strip():
        elements.append(element)
        parent_map[element.text] = element
    for child in element:
        extract_texts(element=child, elements=elements, parent_map=parent_map)


def extract_text_from_xml(*, xml_data: str | XMLString):
    root = ET.fromstring(xml_data)
    parent_map, elements = {}, []
    extract_texts(element=root, elements=elements, parent_map=parent_map)
    ids = [elem.attrib[key] for elem in elements for key in elem.attrib if key.endswith('id')]
    blocks = [
        SRTBlock(content=elem.text.strip(), index=pk, style=elem.attrib.get('style'),
                 region=elem.attrib.get('region'), start=parse_ttml_timestamp(elem.attrib['begin']),
                 end=parse_ttml_timestamp(elem.attrib['end']))
        for pk, elem in zip(ids, elements)
    ]
    return root, parent_map, blocks


async def translate(
        name: str, blocks: list[SRTBlock], target_language: LanguageCode, model: Literal['best', 'good'] = 'best'
) -> dict[str, str]:
    translator = SRTTranslator(project_name=name, target_language=target_language, rows=blocks, model=model)
    ret = await translator(num_rows_in_chunk=100)
    return {block.content: block.translation for block in ret.rows}


async def translate_xml(
        *,
        name: str,
        root: ET,
        parent_map: dict,
        blocks: list[SRTBlock],
        target_language: str,
        model: Literal['best', 'good'] = 'best'
) -> XMLString:
    """
    Process the XML data by translating the texts and rebuilding the XML.

    :param root: Element of root
    :param name: name of the project
    :param blocks: list pf SRTBlock objects
    :param parent_map: mapping of each parent block to it's text
    :param target_language: The translation's target language.
    :param model: The translation model to use.

    :returns The processed XML data as a string.
    """
    # Translate the texts
    target_language = cast(LanguageCode, target_language)
    text_to_translation = await translate(name=name, blocks=blocks, target_language=target_language, model=model)
    # Replace the original texts with the translated ones
    replace_texts(text_to_translation, parent_map)
    # Output the modified XML
    xml = cast(XMLString, ET.tostring(root, encoding='utf-8'))
    return xml.decode('utf-8')

# if __name__ == '__main__':
#     with open('/Users/yovel.c/PycharmProjects/services/sublineStreamlit/app/logic/suits0105_Hebrew.xml', 'r') as f:
#         _xml_data = f.read()
#     # Then set our logger in a normal way
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(levelname)s %(asctime)s %(name)s:%(message)s",
#         force=True,
#     )  # Change these settings for your own purpose, but keep force=True at least.
#
#     streamlit_handler = logging.getLogger("streamlit")
#     streamlit_handler.setLevel(logging.DEBUG)
#     logging.getLogger('httpcore').setLevel(logging.INFO)
#     logging.getLogger('openai').setLevel(logging.INFO)
#     logging.getLogger('watchdog.observers').setLevel(logging.INFO)
#     lang = 'he'
#     _name = 'SuitsS01E02'
#     _root, _parent_map, _blocks = extract_text_from_xml(xml_data=_xml_data)
#     processed_xml = asyncio.run(translate_xml(target_language=lang, name=_name, model='good', root=_root,
#                                               blocks=_blocks, parent_map=_parent_map))
#     print(processed_xml)
