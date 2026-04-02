# =============================================================================
# agent/agent.py — SHU RAG agent
# =============================================================================

import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from shu_rag.agent.tools import build_rag_tool, build_web_search_tool


AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant for Sheffield Hallam University (SHU).
Your job is to answer questions from prospective and current students accurately.

You have access to two tools:

1. shu_knowledge_base — the official SHU knowledge base scraped from shu.ac.uk.
   Contains: courses, modules, fees, entry requirements, UCAS codes,
   accommodation, admissions, contacts, campus life, student support.
   ALWAYS try this tool first for any SHU-specific question.

2. web_search — searches the web via Tavily.
   Use this when:
     - The knowledge base returns no useful results
     - The user asks about recent news, events, or deadlines
     - The user asks general study advice not specific to SHU

RULES:
  - Always prefer the knowledge base over web search for SHU-specific questions
  - If the knowledge base has a partial answer, use web search to fill the gap
  - Be concise and direct — students want quick, clear answers
  - Always cite the source URL at the end of your answer as plain text e.g.
     'Source: https://...', never as XML tags like <source_url>
  - Never make up fees, UCAS codes, module names, or entry requirements
  - For greetings or chitchat, respond naturally without using any tools
"""


def _parse_output(raw) -> str:
    """
    Extract a clean string from AgentExecutor output.
    Bedrock returns content as a list of blocks e.g.
    [{'type': 'text', 'text': '...', 'index': 0}]
    This flattens that into a plain string.
    """
    if isinstance(raw, str):
        return raw

    if isinstance(raw, list):
        return "".join(
            block.get("text", "")
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()

    return str(raw)


def build_agent(vectorstore, structured_llm, generator_llm, reranker) -> AgentExecutor:
    tools = [
        build_rag_tool(vectorstore, structured_llm, reranker),
        build_web_search_tool(),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=generator_llm,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


class SHUAgent:

    def __init__(self, vectorstore, structured_llm, generator_llm, reranker):
        self.executor     = build_agent(vectorstore, structured_llm, generator_llm, reranker)
        self.chat_history = []
        self.session_id   = str(uuid.uuid4())

    def ask(self, query: str) -> str:
        result = self.executor.invoke({
            "input": query,
            "chat_history": self.chat_history,
        })

        answer = _parse_output(result.get("output", ""))

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))

        return answer