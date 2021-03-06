import sctokenizer
from sctokenizer.token import TokenType
import argparse
import glob
import os

C_VAR_DEC_SET = {
    'char',
    'double',
    'enum',
    'float',
    'for',
    'long',
    'signed',
    # 'struct',
    'unsigned',
    'void'
}


GENERIC_FUNCTION_NAME = "FUNCTION_NAME"
GENERIC_VAR_TEMPLATE = "VAR_{}"
GENERIC_STRING_VALUE = "STRING"


def process_tokens(tokens, same_var_name=False):
    variable_lines = set()
    generic_var_value = 0

    variable_map = dict()

    # find all lines with variable declaration (only for primitive variables)
    for token in tokens:
        if token.token_value in C_VAR_DEC_SET:
            if token.line == 1:
                continue
            variable_lines.add(token.line)

    # filter out comments
    tokens = list(filter(lambda x: x.token_type != TokenType.COMMENT_SYMBOL,tokens))

    # map all declared variables and strings to a generic name
    for token in tokens:
        if token.token_type == TokenType.STRING:
            token.token_value = GENERIC_STRING_VALUE

        if token.token_type == TokenType.IDENTIFIER:
            # replace function name with a generic name
            if token.line == 1:
                token.token_value = GENERIC_FUNCTION_NAME

            if token.line in variable_lines:
                old_name = token.token_value
                new_name = GENERIC_VAR_TEMPLATE.format(generic_var_value)

                token.token_value = new_name
                variable_map[old_name] = new_name

                if not same_var_name:
                    generic_var_value += 1

    # replace variable names with generic names
    for token in tokens:
        if token.token_value in variable_map:
            token.token_value = variable_map[token.token_value]

    return tokens


def convert_token_to_line(tokens, label, separator, delimeter):
    tok_list = list()
    for token in tokens:
        tok_list.append(token.token_value)

    data = separator.join(tok_list)
    data += delimeter + str(label)

    return data


def main(args):
    path = args.data_dir
    separator = args.separator
    delimeter = args.delimeter
    same_var_name = args.same_var_name
    print("Search for files in " + path)
    files = glob.glob(path + "**/*.c", recursive=True)

    data_list = []

    for file in files:
        if "goodB2G" in file or "goodG2B" in file:
            label = 0
        else:
            label = "1"

        tokens = sctokenizer.tokenize_file(filepath=file, lang='c')
        tokens = process_tokens(tokens, same_var_name)

        data_list.append(convert_token_to_line(tokens, label, separator, delimeter))

    with open(os.path.join(args.output_dir, args.output_filename), "w") as f:
        f.write("\n".join(data_list))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', default='data/',
                            help="The data directory where the processed CWE files are stored. Use absolute path.")
    arg_parser.add_argument("--output_dir", default="data_processor/C_output/")
    arg_parser.add_argument('--delimeter', default='\t', help="Separator used to separate tokens")
    arg_parser.add_argument('--separator', default=" ")
    arg_parser.add_argument("--output_filename", default="C_processed_data.csv")
    arg_parser.add_argument('--same_var_name',  action='store_false')
    args = arg_parser.parse_args()
    main(args)
