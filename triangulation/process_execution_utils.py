import subprocess


def execute_process(shell_command, stdin_input="", verbose=False):
    result = None
    try:
        if verbose:
            print("Executing command: {}".format(shell_command))
        proc = subprocess.Popen(shell_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout_data, stderr = proc.communicate(stdin_input)
        result = stdout_data
        if verbose:
            print("Process response: {}".format(result))
        return result
    except:
        if verbose:
            print("Command execution error.")
        return result