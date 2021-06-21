import subprocess
import glob
import re
import os
import argparse


CTAG_CMD = "ctags -x --c-kinds=fp {}"
BAD_func_names = ["CWE457.*bad"]
GOOD_func_names = ["goodG2B", "goodB2G"]

ALL_FUNC_NAME_FORMAT = BAD_func_names + GOOD_func_names


# Use tool ctag to get all functions in a .c file and get the line number in which these functions are
def get_line_numbers(filename):
    cmd = CTAG_CMD.format(filename)
    output = subprocess.getoutput(cmd)
    line_numbers = []
    function_names = []

    lines = output.splitlines()
    for line in lines:
        out = line.strip()
        split_lines = list(filter(None, out.split(" ")))
        line_num = split_lines[2]
        func_name = split_lines[0]

        for f_name in ALL_FUNC_NAME_FORMAT:
            found = re.search(f_name, func_name)

            if found:
                line_numbers.append(int(line_num))
                function_names.append(func_name)
                break
    return line_numbers, function_names

# Process the C file and grab the function source code and extract the function at line_num
# I copied + modified this code from stack overflow...
def process_file(filename, line_num):
    code = ""
    cnt_braket = 0
    found_start = False
    found_end = False

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
                    found_end = True
                    return code

# write function into its own file for easy use
def write_output_to_file(filename, data, func):
    output_path = "../../code/data_processor/C_output/"
    found = False
    filename = filename

    for f_name in BAD_func_names:
        found = re.search(f_name, func)

    if found:
        output_path += "BAD/"
    else:
        output_path += "GOOD/" + func + "_"

    output_path += filename

    with open(output_path, "w") as f:
        f.write(data)


def main(args):
    path = args.data_dir
    print(path)

    files = glob.glob(path + "*.c", recursive=True)

    for f in files:
        line_nums, func_names = get_line_numbers(f)
        if not line_nums:
            print("No functions found in file: {}".format(f))

        else:
            for i in range(len(line_nums)):
                    num = line_nums[i]
                    func = func_names[i]
                    output = process_file(f, num)
                    write_output_to_file(os.path.basename(f), output, func)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', default='data/',
                            help="The data directory where the CWE files are stored. Use absolute path.")
    args = arg_parser.parse_args()
    main(args)



