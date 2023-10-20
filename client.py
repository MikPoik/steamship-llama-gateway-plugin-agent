from steamship import Steamship

#from plugin_handle import PLUGIN_HANDLE
PLUGIN_HANDLE = "llamagateway-plugin"

text = """{input}"""


def main(prompt: str):
  with Steamship.temporary_workspace() as client:
    print(f'Running in workspace: {client.config.workspace_handle}')
    llm = client.use_plugin(
        PLUGIN_HANDLE,
        config={
            "api_key": "",
            "model": "NousResearch/Nous-Hermes-Llama2-13b",
            "temperature": 0.6,
            "max_tokens": 256,
            "top_p": 0.9,
            "presence_penalty": 1.15,
            #"stop": "stopstring"
        })
    msg = text.format(input=prompt)
    task = llm.generate(text=msg, )
    task.wait()

    output_blocks = task.output.blocks
    for block in output_blocks:
      print(block.text)


if __name__ == "__main__":

  main((input("text:")))
