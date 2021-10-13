import glob
import re
import os
import argparse
#TODO: refactor code for parser (combine java and C parser)

# NOTE: the following parser is a naive java code parser. It assumes the functions containing CWE from the input code are self-contained. I.E. does not depend on:
#       1) external classes
#       2) other functions
#       3) static variables from standard library
# Suggested next step: use a library that parses code and returns an AST. Traverse the AST to check for function calls, class intializations, private/static variable usage 

# Function signatures for code that contains CWE vs does not contain CWE
BAD_FUNC_NAMES = {
    "public void bad(": "bad"
}
GOOD_FUNC_NAMES = {
    "private void good1(":"good1",
    "private void good2(": "good2",
    "private void goodG2B(": "goodG2B",
    "private void goodB2G(": "goodB2G",
    "private void goodG2B1(": "goodG2B1",
    "private void goodG2B2(": "goodG2B2",
    "private void goodB2G1(": "goodB2G1",
    "private void goodB2G2(": "goodB2G2"
}
FUNC_NAMES = { ** BAD_FUNC_NAMES, ** GOOD_FUNC_NAMES}


def write_output_to_file(filename, data, func, output_path):
    """Writes given function to file 

    Args:
        filename (Path): file name of file to write function to.
        data (str): Code of function.
        func (str): function signature. 

    Returns:
        [None]
    """

    output_path += "BAD/" if func in BAD_FUNC_NAMES else "GOOD/"
    output_path += BAD_FUNC_NAMES[func] if func in BAD_FUNC_NAMES else GOOD_FUNC_NAMES[func]
    output_path += "_"
    output_path += filename
    
    with open(output_path, "w") as f:
        f.write(data)


def process_file(filename, line_num):
    """Processes source code starting from line_num (start of a function) to the end of the function

    Args:
        filename (Path): path to source code file to be processed. 
        line_num (int): line number that indicates the start of a function

    Returns:
        [str]: code of function
    """

    code = ""
    cnt_braket = 0
    found_start = False

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if(i >= (line_num - 1)):
                code += line

                if line.count("{") > 0:
                    found_start = True
                    cnt_braket += line.count("{")

                if line.count("}") > 0:
                    cnt_braket -= line.count("}")

                if cnt_braket == 0 and found_start == True:
                    return code


def get_line_numbers(f):
    """Parses source code to find functions that has function signature specified in FUNC_NAMES and the starting line number

    Args:
        f (Path): Path to source code 

    Returns:
        [Dict]: Dictionary in the form [line number]: function signature. Where function signature is one of FUNC_NAME
    """

    line_nums_dict = {}
    f = open(f, 'r')
    for line_num, line in enumerate(f):
        line = line.strip()
        for func_sig in FUNC_NAMES:
            if func_sig in line:
                line_nums_dict[func_sig] = line_num+1
    return line_nums_dict


def main(args):
    path = args.data_dir
    print(f"Parsing code in {path}")

    files = glob.glob(path + "*.java", recursive=True)

    for f in files:
        line_nums_dict = get_line_numbers(f)
        if len(line_nums_dict) < 3:
            print(f"Less than 3 functions in file {f}. "
                  f"Please inspect manually to check if any functions are not parsed.")

        for func_name, line_num in line_nums_dict.items():
                output = process_file(f, line_num)
                write_output_to_file(os.path.basename(f), output, func_name, args.output_dir)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir',
                            default='../../code/data_processor/sample_data/Java/CWE476_NULL_Pointer_Dereference/',
                            help="The data directory where the CWE files are stored")
    arg_parser.add_argument('--output_dir', default='../../code/data_processor/Java_Output/')
    args = arg_parser.parse_args()
    main(args)
