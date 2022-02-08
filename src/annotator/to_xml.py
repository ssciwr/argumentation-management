from xml.etree.ElementTree import Element, tostring
import xml.dom.minidom as mini


def list_to_xml(tag: str, idx: int, List: list) -> Element:
    """Convert a given list l of dictionaries d from stanza output into an xml element under
    given tag with idx."""

    # root Element is tag
    root = Element(tag)
    # use attribute to distinguish sentences
    root.attrib = {"Id": str(idx)}
    # iterate list
    for Dict in List:
        # iterate key/value pairs
        for key, val in Dict.items():
            # if key is id we start new node for token
            if key == "id":
                node = Element("Token")
                # attribute is token id
                node.attrib = {"Id": str(val)}
            elif key != "id":
                # if key is not id we are inside of token, start sub_node
                sub_node = Element(key)
                sub_node.text = str(val)
                # append sub_node with token attr to token node
                node.append(sub_node)
        # append the node to root
        root.append(node)

    return root


def dict_to_xml(tag: str, Dict: dict) -> Element:
    """Basic function to convert a given dictionary d into an xml element under
    given tag."""

    # initialize root
    root = Element(tag)

    # iterate key/value pairs
    for key, val in Dict.items():
        # every key is a sepparate node
        node = Element(key)
        # text of node is value
        node.text = str(val)
        # append node to root
        root.append(node)

    return root


def to_string(xml: Element) -> str:
    """Function to turn xml object to str."""

    return tostring(xml, "unicode")


def beautify(xml_str: str) -> str:
    """Function to beautify xml output."""

    return mini.parseString(xml_str).toprettyxml()


def start_xml(tag: str) -> Element:
    """Function to start xml object from given tag."""

    return Element(tag)
