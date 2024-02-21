#! /usr/bin/env python3
import argparse
import json


def escape_latex_characters(unformatted_string):
    """
    Escapes special LaTeX characters in a string to make it compatible with LaTeX text mode.
    Also attempts to replace certain Unicode characters with LaTeX representations.

    Args:
    - unformatted_string (str): The string to be formatted for LaTeX.

    Returns:
    - str: The LaTeX-compatible string with special characters escaped.
    """
    # Mapping of special characters and some Unicode characters to LaTeX
    special_chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "[": r"{[}",
        "]": r"{]}",
        '"': r"''",
        "\\": r"\textbackslash{}",
        "~": r"\textasciitilde{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
        "^": r"\textasciicircum{}",
        "`": r"\textasciigrave{}",
        "|": r"\textbar{}",
        # Unicode characters fixable within pdftex context
        "\u2039": r"\guilsinglleft{}",  # Single left-pointing angle quotation mark
        "\u203A": r"\guilsinglright{}",  # Single right-pointing angle quotation mark
        "\u2081": r"$_{1}$",  # Subscript one
        "\u21b5": r"\(\downarrow\)",  # Downwards arrow with corner leftwards
        "\u2219": r"\(\bullet\)",  # Bullet operator
        "№": r"\textnumero",
        "┃": r"$\Vert$",
    }

    # Escape each special character in the unformatted_string
    formatted_string = unformatted_string.translate(str.maketrans(special_chars))
    return formatted_string


plain_str = r""""\": '{...]\\u253c}:..}^{( \\\"${ `/'; ['================}] '';');\\r });\\r\\\";\\r\\\", `[{ <!-- [( \\\"$\\u22a5={\\\\}+)+???-%=-}+\ """
plain_str = json.loads(
    r""": '{...]\\u253c}:..}^{( \\\"${ `/'; ['================}] '';');\\r });\\r\\\";\\r\\\", `[{ <!-- [( \\\"$\\u22a5={\\\\}+)+???-%=-}+"""
)


def main():
    parser = argparse.ArgumentParser(description="Escape special characters in a string for LaTeX.")
    parser.add_argument("--string", type=str, default=None, help="The string to be escaped for LaTeX.")
    parser.add_argument("--ascii", action="store_true", default=False, help="Strip all non-ascii characters remaining after conversion.")
    parser.add_argument(
        "--alpha", action="store_true", default=False, help="Strip all non-alphabetic characters remaining after conversion."
    )

    args = parser.parse_args()
    if args.string is None:
        escaped_string = escape_latex_characters(plain_str)
    else:
        escaped_string = escape_latex_characters(args.string)

    if args.alpha:
        escaped_string = "".join(c for c in escaped_string if c.isalpha() or c.isascii())
    if args.ascii:
        escaped_string = "".join(c for c in escaped_string if c.isascii())
    print("\n")
    print(escaped_string)


if __name__ == "__main__":
    main()
