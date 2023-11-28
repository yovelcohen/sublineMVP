import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import timedelta
from typing import cast, Literal

from logic.constructor import SRTTranslator
from logic.function import SRTBlock

XMLString = str | bytes


def extract_texts(element: ET.Element, texts: list[str], parent_map: dict[str, ET.Element]) -> None:
    """
    Recursively extract text from an XML element and its children.

    :param element: The current XML element.
    :param texts: list to store the extracted texts.
    :param parent_map: Map to store the relationship between texts and their parent elements.
    """
    if element.text and element.text.strip():
        texts.append(element.text)
        parent_map[element.text] = element
    for child in element:
        extract_texts(child, texts, parent_map)


def replace_texts(text_to_translation: dict[str, str], parent_map: dict[str, ET.Element]) -> None:
    """
    Replace the original texts in the XML with translated texts.

    :param text_to_translation: mapping from text to its translation
    :param parent_map: Map containing the relationship between original texts and their parent elements.
    """
    for original_text, translated_text in text_to_translation.items():
        parent_map[original_text].text = translated_text


async def translate(
        name: str, blocks: list[SRTBlock], target_language: str, model: Literal['best', 'good'] = 'best'
) -> dict[str, str]:
    translator = SRTTranslator(project_name=name, target_language=target_language, rows=blocks, model=model)
    ret = await translator(100)
    return {block.content: block.translation for block in ret.rows}


def extract_text_from_xml(xml_data: str | XMLString):
    root = ET.fromstring(xml_data)
    parent_map, texts_to_translate = {}, []
    extract_texts(root, texts_to_translate, parent_map)
    blocks = [SRTBlock(content=text, index=i, start=timedelta(seconds=1), end=timedelta(seconds=1))
              for i, text in enumerate(texts_to_translate)]
    return root, parent_map, blocks


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
    text_to_translation = await translate(name=name, blocks=blocks, target_language=target_language, model=model)
    # Replace the original texts with the translated ones
    replace_texts(text_to_translation, parent_map)
    # Output the modified XML
    xml = cast(XMLString, ET.tostring(root, encoding='utf-8'))
    return xml.decode('utf-8')

# if __name__ == '__main__':
#     with open('/Users/yovel.c/PycharmProjects/services/sublineStreamlit/app/logic/episode_2_heb.nfs', 'r') as f:
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
#     processed_xml = asyncio.run(translate_xml(xml_data=_xml_data, target_language=lang, name=_name, model='good'))
#     print(processed_xml)
