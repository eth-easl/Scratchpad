import re
import json

TOOLS_TAG_LIST = ["<|plugin|>", "<function=", "<tool_call>", "<|python_tag|>"]


def parse_tool_response(text, tools, **kwargs):
    """Parse model response containing tool information.

    Args:
        text(str): model response in string format
        tools(List): tools from user request
    """

    if "<function=" in text:  # llama3.1
        action, _ = text.split("</function>")
        parameters = action[action.find("{") :]
        name = action.split("<function=")[1].split(">{")[0]
        call_info_list = [(name, parameters)]
    elif "<tool_call>" in text and "</tool_call>" in text:  # qwen2.5
        # get tool_call in text
        pattern = r"<tool_call>(.*?)</tool_call>"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        call_info_list = []
        for match_result in match_result_list:
            action = json.loads(match_result)
            call_info_list.append(
                (action["name"], json.dumps(action["arguments"], ensure_ascii=False))
            )
        # get text outside of tags
        if not text.startswith("<tool_call>"):
            text = text[: text.find("<tool_call>")]
        elif not text.endswith("</tool_call>"):
            text = text[text.rfind("</tool_call>") + len("</tool_call>") :]
        else:
            text = ""
    elif "<|python_tag|>" in text:  # llama3.2
        _, action = text.split("<|python_tag|>")
        action = json.loads(action)
        name, parameters = action["name"], json.dumps(
            action.get("parameters", action.get("arguments", {})), ensure_ascii=False
        )
        call_info_list = [(name, parameters)]
    else:
        raise RuntimeError(f"Unexpected model response: {text}")

    call_info_list = [
        (
            [tool.function.name for tool in tools].index(call_info[0]),
            call_info[0],
            call_info[1],
        )
        for call_info in call_info_list
    ]
    return text, call_info_list
