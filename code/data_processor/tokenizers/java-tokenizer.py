import sctokenizer
from sctokenizer.token import TokenType
import argparse
import glob

GENERIC_FUNCTION_NAME = "FUNCTION_NAME"

def process_tokens(tokens, distinct_var=True):
    # filter out comments
    tokens = list(filter(lambda x: x.token_type != TokenType.COMMENT_SYMBOL ,tokens))

    # map all functions to generic function names
    for token in tokens:
        if token.token_type == TokenType.IDENTIFIER:
            if token.line == 1:
                token.token_value = GENERIC_FUNCTION_NAME

    return tokens


def convert_token_to_line(tokens, label, delimiter):
    tok_list = list()
    for token in tokens:
        tok_list.append(token.token_value)

    data = delimiter.join(tok_list)
    data += "," + str(label)

    return data
    
def main(args):
    path = args.data_dir
    delimiter = args.delimiter
    distinct_var = args.distinct_var
    print("Search for files in " + path)
    files = glob.glob(path + "**/*.java", recursive=True)

    data_list = []

    for file in files:
        if "good" in file:
            label = 0
        else:
            label = "1"

        tokens = sctokenizer.tokenize_file(filepath=file, lang='java')
        tokens = process_tokens(tokens, distinct_var)

        data_list.append(convert_token_to_line(tokens, label, delimiter=delimiter))

    with open("processed_data.csv", "w") as f:
        f.write("\n".join(data_list))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', default='../Java_output/',
                            help="The data directory where the processed CWE files are stored. Use absolute path.")
    arg_parser.add_argument('--delimiter', default='\t', help="Separator uesd to separate tokens")
    arg_parser.add_argument('--distinct_var', type=bool, default=True)
    args = arg_parser.parse_args()
    main(args)
