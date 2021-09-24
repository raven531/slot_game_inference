import re
import os

pattern = '[\.|\_]+\w*'


def single_trim(symbol) -> str:
    return re.sub(pattern, "", symbol)


def trim(symbols_path: str) -> list:
    symbol_classes = [re.sub(pattern, "", l) for l in os.listdir(symbols_path)]
    symbol_classes = list(dict.fromkeys(symbol_classes))  # remove duplicate from list

    return symbol_classes
