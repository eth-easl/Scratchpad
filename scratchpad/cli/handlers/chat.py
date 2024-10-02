# adapted from https://github.com/qnixsynapse/rich-chat/blob/main/source/rich-chat.py
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
import argparse
import json
import os
import openai
import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from scratchpad.server.tools._base import render_function


def remove_lines_console(num_lines):
    for _ in range(num_lines):
        print("\x1b[A", end="\r", flush=True)


def estimate_lines(text):
    columns, _ = os.get_terminal_size()
    line_count = 1
    text_lines = text.split("\n")
    for text_line in text_lines:
        lines_needed = (len(text_line) // columns) + 1

        line_count += lines_needed

    return line_count


def handle_console_input(session: PromptSession) -> str:
    return session.prompt("(Prompt: ⌥ + ⏎) | (Exit: ⌘ + c): ", multiline=True).strip()


class ChatHandler:
    def __init__(
        self,
        server_addr,
        model_name: str,
        min_p: float = 0.5,
        repeat_penalty: float = 1.0,
        seed: int = 0,
        top_k=10,
        top_p=0.95,
        temperature=0.12,
        n_predict=-1,
        stream: bool = True,
        cache_prompt: bool = True,
        model_frame_color: str = "red",
    ) -> None:
        self.model_frame_color = model_frame_color
        self.model_name = model_name
        self.serveraddr = server_addr
        self.temperature = temperature
        self.stream = stream
        self.cache_prompt = cache_prompt
        self.headers = {"Content-Type": "application/json"}
        self.chat_history = []
        self.model_name = ""
        self.console = Console()
        self.client = openai.OpenAI(api_key="test", base_url=self.serveraddr + "/v1")
        # TODO: Gracefully handle user input history file.
        self.session = PromptSession(history=FileHistory(".rich-chat.history"))
        self.seed = seed

    def chat_generator(self, prompt):
        prompt = render_function(prompt)
        print(f"Prompt: {prompt}")
        print(f"--------")
        self.chat_history.append({"role": "user", "content": prompt})

        payload = {
            "messages": self.chat_history,
            "temperature": self.temperature,
            "stream": self.stream,
            "seed": self.seed,
            "model": self.model_name,
        }
        try:
            response = self.client.chat.completions.create(**payload)
            for chunk in response:
                if chunk:
                    # if "content" in chunk:
                    yield chunk
        except Exception as e:
            print(f"GeneratorError: {e}")

    def health_checker(self):
        try:
            endpoint = self.serveraddr + "/health"
            response = requests.get(url=endpoint, headers=self.headers)
            assert (
                response.status_code == 200
            ), "Unable to reach server! Please check if server is running or your Internet connection is working or not."
            status = response.content.decode("utf-8")
            return status.lower()
        except Exception as e:
            print(f"HealthError: {e}")

    def get_model_name(self):
        try:
            endpoint = self.serveraddr + "/slots"
            response = requests.get(url=endpoint)
            assert response.status_code == 200, "Server not reachable!"
            data = json.loads(response.content.decode("utf-8"))[0]["model"]
            return data
        except Exception as e:
            print(f"SlotsError: {e}")

    def handle_streaming(self, prompt):
        self.console.print(Markdown("**>**"), end=" ")
        text = ""
        block = "█ "
        with Live(
            console=self.console,
        ) as live:
            for token in self.chat_generator(prompt=prompt):
                if token.choices[0].delta.content:
                    text = text + token.choices[0].delta.content
                if token.choices[0].finish_reason is not None:
                    block = ""
                markdown = Markdown(text + block)
                live.update(
                    markdown,
                    refresh=True,
                )
        self.chat_history.append({"role": "assistant", "content": text})

    def chat(self):
        status = self.health_checker()
        assert status == "ok", "Server not ready or error!"
        while True:
            try:
                user_m = handle_console_input(self.session)
                self.handle_streaming(prompt=user_m)

            # NOTE: Ctrl + c (keyboard) or Ctrl + d (eof) to exit
            # Adding EOFError prevents an exception and gracefully exits.
            except (KeyboardInterrupt, EOFError):
                exit()
