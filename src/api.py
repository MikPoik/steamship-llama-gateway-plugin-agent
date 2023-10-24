from agent.llama_react import ReACTAgent
from agent.gwllama_llm import LlamaLLM
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from steamship.agents.service.agent_service import AgentService
from steamship.agents.tools.image_generation import DalleTool
from steamship.agents.tools.search import SearchTool
from steamship.utils.repl import AgentREPL


class MyAssistant(AgentService):
  """MyAssistant is an example for using llamagateway-plugin"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.set_default_agent(
        ReACTAgent(
            tools=[SearchTool(), DalleTool()],
            llm=LlamaLLM(
                self.client,
                api_key="",  #Get your apikey at llamagateway.com
                temperature=0.6,
                top_p=0.9,
                max_tokens=300,
                model_name="teknium/OpenHermes-2-Mistral-7B"),
            message_selector=MessageWindowMessageSelector(k=2),
        ))


if __name__ == "__main__":
  # AgentREPL provides a mechanism for local execution of an AgentService method.
  # This is used for simplified debugging as agents and tools are developed and
  # added.
  AgentREPL(MyAssistant, agent_package_config={}).run()
