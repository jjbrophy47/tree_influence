"""
Utility functions for processing arguments and config files.
"""


def args_to_yaml(args, outfile_name, ignore_list=[]):
    """
    Writes the values to a file in YAML format. Some
        parameter values might not need to be saved, so
        you can pass a list of parameter names as the
        `ignore_list`, and the values for these parameter
        names will not be saved to the YAML file.

    Input
        cmd_args: ArgParser, parsed arguments.
        outfile_name: str, filename to write to.
        ignore_list: list, names of arguments to ignore.
    """
    args_dict = vars(args)

    with open(outfile_name, 'w') as yaml_outfile:
        for parameter, value in args_dict.items():

            # don't write the parameter value if parameter in the
            # ignore list or the value of the parameter is None
            if parameter in ignore_list or value is None:
                continue

            # write boolean values as 1's and 0's
            else:
                if isinstance(value, bool):
                    value = int(value)

                yaml_outfile.write(f'{parameter}: {value}\n')