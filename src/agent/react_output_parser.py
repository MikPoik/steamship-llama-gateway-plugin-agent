import logging
import re
from typing import Dict, List, Optional
from steamship import Block, Steamship  #upm package(steamship)
from steamship.agents.schema import Action, AgentContext, FinishAction, OutputParser, Tool  #upm package(steamship)


class ReACTOutputParser(OutputParser):
  'Parse LLM output expecting structure matching ReACTAgent default prompt'

  tools_lookup_dict: Optional[Dict[str, Tool]] = None

  def __init__(self, **kwargs):
    tools_lookup_dict = {tool.name: tool for tool in kwargs.pop("tools", [])}
    super().__init__(tools_lookup_dict=tools_lookup_dict, **kwargs)

  def parse(self, text: str, context: AgentContext) -> Action:
    text = text.replace('`', "")  # no backticks
    text = text.replace('"', "'")  # use single quotes in text
    text = text.strip()  #remove extra spaces

    current_name = "AI"

    #logging.warning(text)

    if "<" + current_name + ">" in text or "</" + current_name + ">" in text:
      if not "<tool>" in text:
        return FinishAction(output=ReACTOutputParser._blocks_from_text(
            context.client, text, context),
                            context=context)

    regex = r"<tool>(.*?)<\/tool>\s*<tool_input>(.*?)<\/tool_input>"
    match = re.search(regex, text, re.DOTALL)

    if not match:
      logging.warning(f"Prefix missing, {text} send output to user..")
      text = text.replace(current_name + ":", "").strip()
      return FinishAction(output=ReACTOutputParser._blocks_from_text(
          context.client, text, context),
                          context=context)
    action = match.group(1)
    action_input = match.group(2).strip()
    tool = action.strip()
    if tool is None:
      raise RuntimeError(
          f"Could not find tool from action: `{action}`. Known tools: {self.tools_lookup_dict.keys()}"
      )
    return Action(
        tool=tool,
        input=[Block(text=action_input)],
        context=context,
    )

  @staticmethod
  def _blocks_from_text(client: Steamship, text: str,
                        context: AgentContext) -> List[Block]:
    current_name = "AI"

    message = text
    if "<" + current_name + ">" in message:
      message = message.split("<" + current_name + ">", 1)[-1].strip()
    if "</" + current_name + ">" in message:
      message = message.split("</" + current_name + ">")[0].strip()

    result_blocks: List[Block] = []

    block_id_regex = r"(?:(?:\[|\(|<)?Block)?\(?([A-F0-9]{8}\-[A-F0-9]{4}\-[A-F0-9]{4}\-[A-F0-9]{4}\-[A-F0-9]{12})\)?(?:(\]|\)|>)?)"
    remaining_text = message
    while remaining_text is not None and len(remaining_text) > 0:
      match = re.search(block_id_regex, remaining_text)
      if match:
        pre_block_text = ReACTOutputParser._remove_block_prefix(
            candidate=remaining_text[0:match.start()])
        if len(pre_block_text) > 0:
          result_blocks.append(Block(text=pre_block_text))
        result_blocks.append(Block.get(client, _id=match.group(1)))
        remaining_text = ReACTOutputParser._remove_block_suffix(
            remaining_text[match.end():])
      else:
        result_blocks.append(
            Block(text=remaining_text.replace("</message>", "")))
        remaining_text = ""
    return result_blocks

  @staticmethod
  def _remove_block_prefix(candidate: str) -> str:
    removed = candidate
    if removed.endswith("(Block") or removed.endswith(
        "[Block") or removed.endswith("<Block"):
      removed = removed[len("Block") + 1:]
    elif removed.endswith("Block"):
      removed = removed[len("Block"):]
    return removed

  @staticmethod
  def _remove_block_suffix(candidate: str) -> str:
    removed = candidate
    if removed.startswith(")") or removed.endswith("]") or removed.endswith(
        ">"):
      removed = removed[1:]
    return removed
