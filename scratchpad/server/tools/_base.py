import re
from scratchpad.server.tools.core_functions import *


def render_function(text):
    # extract the function signature, which is wrapped like #calculator(3,5,'+')#
    functions = re.findall(r"#(.*?)#", text)
    for function in functions:
        # split the function signature into function name and arguments
        function_name, *args = function.split("(")
        args = args[0].split(",")
        args[-1] = args[-1].replace(")", "")
        # calculate the result of the function
        result = eval(f"{function_name}({','.join(args)})")
        # replace the function signature with the result
        text = text.replace(f"#{function}#", str(result))
    return text


if __name__ == "__main__":
    # text = "Since 1234 * 4321= #calculator(1234,4321,'*')#, then what is 1234*4321*2?"
    text = "With the context: #search('Switzerland')#, tell me something about Switzerland."
    result = render_function(text)
