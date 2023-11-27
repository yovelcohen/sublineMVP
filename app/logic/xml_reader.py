import xml.etree.ElementTree as ET
from typing import NewType, cast

from logic.function import translate_texts_list

XMLString = NewType('XMLString', str)


async def translate(texts: list[str], target_language: str):
    """
    Translate a list of texts.
    :param target_language: the translation's target language
    :param texts: A list of strings to be translated.
    :returns list[str]: The translated list of strings.
    """
    ret = await translate_texts_list(texts=texts, target_language=target_language)
    assert len(texts) == len(ret)
    return list(ret.values())


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


def replace_texts(texts_translated: list[str], parent_map: dict[str, ET.Element]) -> None:
    """
    Replace the original texts in the XML with translated texts.

    :param texts_translated: The list of translated texts.
    :param parent_map: Map containing the relationship between original texts and their parent elements.
    """
    for original_text, translated_text in zip(parent_map, texts_translated):
        parent_map[original_text].text = translated_text


async def translate_xml(xml_data: str, target_language: str) -> XMLString:
    """
    Process the XML data by translating the texts and rebuilding the XML.
    :param xml_data: XML data in string format.
    :param target_language: The translation's target language.

    :returns The processed XML data as a string.
    """
    # Parse the XML
    root = ET.fromstring(xml_data)

    # Extract texts and keep a map of their parent elements
    parent_map, texts_to_translate = {}, []
    extract_texts(root, texts_to_translate, parent_map)

    # Translate the texts
    translated_texts = await translate(texts_to_translate, target_language)

    # Replace the original texts with the translated ones
    replace_texts(translated_texts, parent_map)

    # Output the modified XML
    return cast(XMLString, ET.tostring(root, encoding='utf-8'))


if __name__ == '__main__':
    with open('/Users/yovel.c/PycharmProjects/services/sublineStreamlit/demo1.xml', 'r') as f:
        _xml_data = f.read()
    lang = 'Hebrew'
    processed_xml = translate_xml(_xml_data, target_language=lang)
    print(processed_xml)
