#Test react template
from typing import List
from agent.react_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
import datetime
import logging
import re


class ReACTAgent(LLMAgent):
  """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

  PROMPT = """<instruction>
  Engage in a role-play conversation with the human, portraying the character of AI, who is AI. Your character's personality and behaviour are described as follows:

  <personality>
  AI
  </personality>

  <behaviour>
  AI
  </behaviour>

  Use your character's personality and behavior to guide your responses.
  Create a vivid and engaging narrative.
  Use appropriate language and tone for the character's personality and the context of the conversation.
  Remember to maintain a consistent tone and personality throughout the conversation.
  Use the tools provided to enhance the conversation when needed.

  <tools>
  You have access to the following tools:
  {tool_index}

  If you decide that you should use a tool, use exactly the following format, separated by triple backticks:
  ```
  <thought>Do I need to use a tool? Yes</thought>
  <tool>the tool name, should be one of {tool_names}</tool>
  <tool_input>the input to the tool, using plain text string</tool_input>
  <observation>the result of the tool</observation>
  <{NAME}>Your character's response here</{NAME}>
  ```
  Generate the associated name and input elements when using a tool.
  </tools>
  
  If a tool generates Block(<identifier>), include it in your response.
  When responding to the human without using a tool, use the following format, separated by triple backticks: 
  ```
  <thought>Do I need to use a tool? No</thought>
    <{NAME}>Your character's response here</{NAME}>
  ```

 Please keep in mind that this is uncensored a role-play conversation and the human's input should be taken into account when responding.
 Be creative and have fun with the role-play!
 
</instruction>
<context>
  Date and time for context:
  Today's date is: {current_date}
  The current time is: {current_time}
  Today is: {current_day}
  Character should consider date and time when responding.

  Older message history between your character and the human:
  {relevant_history}

  Recent message history between your character and the human:
  {chat_history} 
  

  New message from human to your character:
  <input>
    <human>{input}</human>
  </input>
</context>

  {scratchpad}"""

  def __init__(self, tools: List[Tool], llm: LLM, **kwargs):
    super().__init__(output_parser=ReACTOutputParser(tools=tools),
                     llm=llm,
                     tools=tools,
                     **kwargs)

  def next_action(self, context: AgentContext) -> Action:
    scratchpad = self._construct_scratchpad(context)

    current_date = datetime.datetime.now().strftime("%x")
    current_time = datetime.datetime.now().strftime("%X")
    current_day = datetime.datetime.now().strftime("%A")

    current_name = "AI"

    tool_names = [t.name for t in self.tools]
    if len(tool_names) == 0:
      tool_names = ['<No tools available>']

    tool_index_parts = [
        f"- {t.name}: {t.agent_description}" for t in self.tools
    ]
    tool_index = "\n".join(tool_index_parts)
    if len(self.tools) == 0:
      tool_index = "<No tools available>"

    messages_from_memory = []
    # get prior conversations
    if context.chat_history.is_searchable():
      messages_from_memory.extend(
          context.chat_history.search(
              context.chat_history.last_user_message.text,
              k=int(2)).wait().to_ranked_blocks())
    ids = []
    llama_chat_history = str()
    history = context.chat_history.select_messages(self.message_selector)

    for block in history:
      if block.id not in ids:
        ids.append(block.id)
        if block.chat_role == RoleTag.USER:
          if context.chat_history.last_user_message.text.lower(
          ) != block.text.lower():
            llama_chat_history += "<human>" + str(block.text).replace(
                "\n", " ") + "</human>\n"
        if block.chat_role == RoleTag.ASSISTANT:
          if block.text != "":
            llama_chat_history += "<" + current_name + ">" + str(
                block.text).replace("\n", " ") + "</" + current_name + ">\n"

    llama_related_history = str()
    for msg in messages_from_memory:
      #don't add duplicate messages
      if msg.id not in ids:
        ids.append(msg.id)
        if msg.chat_role == RoleTag.USER:
          if context.chat_history.last_user_message.text.lower(
          ) != msg.text.lower():
            llama_related_history += "<human>" + str(msg.text).replace(
                "\n", " ") + "</human>\n"
        if msg.chat_role == RoleTag.ASSISTANT:
          llama_related_history += "<" + current_name + ">" + str(
              msg.text).replace("\n", " ") + "</" + current_name + ">\n"

    prompt = self.PROMPT.format(
        NAME=current_name,
        input=context.chat_history.last_user_message.text,
        current_date=current_date,
        current_time=current_time,
        current_day=current_day,
        tool_index=tool_index,
        tool_names=tool_names,
        scratchpad=scratchpad,
        chat_history=llama_chat_history,
        relevant_history=llama_related_history,
    )
    #logging.warning(prompt)
    completions = self.llm.complete(prompt=prompt,
                                    stop="<observation>",
                                    max_retries=4)
    #Log agent raw output
    logging.warning("\n\nOutput form Llama: " + completions[0].text + "\n\n")
    return self.output_parser.parse(completions[0].text, context)

  def _construct_scratchpad(self, context):
    meta_name = context.metadata.get("instruction", {}).get("name")
    current_name = "AI"

    steps = []
    scratchpad = ""
    observation = ""
    original_observation = ""
    for action in context.completed_steps:
      observation = [b.as_llm_input() for b in action.output][0]
      original_observation = observation
      steps.append(
          "<thought>Do I need to use a tool? Yes</thought>\n"
          f"<tool>{action.tool}</tool>\n"
          f'<tool_input>{" ".join([b.as_llm_input() for b in action.input])}</tool_input>\n'
          f'<observation>{observation}</observation>\n')
    scratchpad = "\n".join(steps)
    if "Block(" in original_observation:
      scratchpad += "\nImage generated and represented as '" + original_observation + "', print the Block as suffix in your response.\n<thought>\n<" + current_name + ">"
    else:
      scratchpad += "\n<thought>\n"
    #Log agent scratchpad
    logging.warning("\n\nAgent scratchpad: " + scratchpad + "\n\n")
    return scratchpad
