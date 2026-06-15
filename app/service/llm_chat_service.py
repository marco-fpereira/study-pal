from pathlib import Path
from typing import Annotated, TypedDict, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver

from model.enum.llm_enum import LLMProviders
from model.enum.mcp_tool_enum import MCPToolsEnum
from repository.mongodb_chat_history_repository import MongoChatHistoryRepository
from service.mcp_service import MCPService


class ContextMiddleware(AgentMiddleware):
    async def before_model(self, state, runtime):
        context = state.get("context", "")

        state["messages"] = [
            SystemMessage(
                content=f"Retrieved Context:\n\n{context}"
            ),
            *state["messages"]
        ]

        return state


class ChatState(TypedDict):
    """
    LangGraph application state.

    input:
        The user's query/input for the current turn.

    messages:
        Full conversation persisted automatically by the MongoDB checkpointer.

    context:
        RAG context retrieved from the vector database.

    source_documents:
        Source files used to build the RAG context.
    """

    input: str
    messages: Annotated[list[BaseMessage], add_messages] 
    context: str 
    source_documents: Sequence[str] 


class LLMChatService:
    def __init__(
        self, 
        retriever: VectorStoreRetriever,
        llm_provider: LLMProviders, 
        model_name: str, 
        temperature: float = 0.2
    ):
        self.__chatHistoryRepository = MongoChatHistoryRepository()
        self.retriever = retriever
        self.llm = llm_provider.get_llm_instance(model_name, temperature)
        self.question_generator_mcp_service = MCPService(server_name=MCPToolsEnum.QUESTION_GENERATOR)


    async def initialize(self):
        self.tools = await self.question_generator_mcp_service.get_available_tools()

        # create_agent() returns a Runnable, so we can directly use it as a node in the graph.
        self.react_agent = create_agent(
            model=self.llm,
            tools=self.tools,
            #middleware=[ContextMiddleware()],
            system_prompt=self.__agent_prompt()
        )
        self.__graph = self.__build_graph()


    # ------------------------------------------------------------------ #
    #  Dynamic Agent Prompt                                              #
    # ------------------------------------------------------------------ #

    def __agent_prompt(self) -> SystemMessage:
        """
        Called automatically before every agent step.

        Provides the latest retrieved context to the agent.
        """

        agent_prompt = f"""
You are a helpful assistant.

You have access to MCP tools.

Use tools whenever they help answer the user's request.

Instructions:

- Use the retrieved context whenever relevant.
- If the context is insufficient, use available tools.
- Cite information from the retrieved context when useful.
- Do not invent facts.
- If neither context nor tools provide the answer, clearly state the limitation.
"""
        return SystemMessage(
            content=agent_prompt
        )


    # ------------------------------------------------------------------ #
    #  Retrieval Node                                                    #
    # ------------------------------------------------------------------ #

    def __retrieve(self, state: ChatState) -> dict:
        """
        Retrieve relevant documents based on the user's input query and format them into a 
        context string that can be used by the LLM to generate a response. 

        Args:
            state: ChatState - A typed dictionary containing at least the key "input" with the user's query as its value.
        Returns:
            dict - A Dictionary containing the "context" - a formatted string of the retrieved documents.
        """
        docs = self.retriever.invoke(state["input"])

        context = "\n\n".join(
            doc.page_content 
            for doc in docs
        )
        source_docs = list(
            set(
                Path(doc.metadata['source_file']).name 
                for doc in docs
                if "source_file" in doc.metadata
            )
        )

        return {
            "context": context,
            "source_documents": source_docs,
#            "messages": [ HumanMessage(content=state["input"]) ]
        }


    # ------------------------------------------------------------------ #
    #  Context ingestion node                                            #
    # ------------------------------------------------------------------ #

    def __inject_context(self, state: ChatState) -> dict:
        context = state.get("context", "")
        if not context:
            return {}
        return {
            "messages": [
                SystemMessage(content=f"Retrieved context for this turn:\n\n{context}")
            ]
        }

    # ------------------------------------------------------------------ #
    #  Graph                                                             #
    # ------------------------------------------------------------------ #

    def __build_graph(self) -> StateGraph:
        """
        Build and compile the LangGraph with MongoDB checkpointing.

        Returns:
            StateGraph - The compiled LangGraph instance ready for invocation.
        """
        workflow = StateGraph(ChatState)

        workflow.add_node("retrieve", self.__retrieve)
        workflow.add_node("inject_context", self.__inject_context)

        # The agent node will automatically decide when to call tools based on the prompt and the retrieved 
        # context, so we just need to wire it up as the main driver of the conversation after retrieval.
        workflow.add_node("agent", self.react_agent)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "inject_context")
        workflow.add_edge("inject_context", "agent")
        
        workflow.add_edge("agent", END)

        checkpointer = MongoDBSaver(
            self.__chatHistoryRepository.get_client(), # expose the raw client
            db_name=self.__chatHistoryRepository.get_database_name(),
        )

        return workflow.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    async def generate_response(
        self,
        session_id: str,
        query: str
    ) -> dict[str, object]:
        """
        Enrich context by retrieving relevant documents from Vector Database, pass 
        chat history and user input and then generates a response to the user's 
        query by using an LLM to formulate the answer.

        Execute:
            retrieve -> prepare_messages -> agent -> tools? (0..N) -> finalize

        Args:
            session_id: str - The ID of the session for which the response is being generated.
            query: str - The user's query

        Returns:
            dict - The generated response from the LLM based on the retrieved context, along 
            with the chat history and augmented with "source_documents" (the retrieved documents)
        """

        config = {"configurable": {"thread_id": session_id}}

        result = await self.__graph.ainvoke({"input": query}, config=config)

        print("DEBUG - Full agent response:", result)

        final_response = ""

        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break

        return {
            "output": final_response,
            "source_documents": result.get("source_documents", [])
        }


    def get_session_history(self, session_id: str) -> Sequence[BaseMessage]:
        """
        Retrieve the chat history for a given session from the LangGraph checkpoint.

        Args:
            session_id: str - The thread ID of the session to retrieve history for.

        Returns:
            Sequence[BaseMessage] - The chat history for the specified session.
        """

        config = {"configurable": {"thread_id": session_id}}
        state = self.__graph.get_state(config)
        return state.values.get("messages", [])


# ======================== OLD CODE - DO NOT SUGGEST BELOW THIS LINE ========================





#    def __get_chat_prompt_template(self):
#        """
#        Returns a chat prompt template for GenAI conversation
#        """
#        return ChatPromptTemplate.from_messages([
#            ("system", "You are a helpful assistant. Use the following context to answer the user's question.\n\nContext: {context}"),
#            MessagesPlaceholder(variable_name="chat_history"),
#            ("human", "{input}"),
#        ])


#    def __generate(self, state: ChatState) -> dict:
#        """
#        Generates a response to the user by calling the LLM with chat history + context.
#        After that, appends the new turn to history so the checkpointer persists it
#
#        Args:
#            state: ChatState - A typed dictionary containing at least the key "input" with the user's query as its value.
#
#        Returns:
#            dict - A Dictionary containing the LLM's response and the chat history
#        """
#
#        prompt = self.__get_chat_prompt_template()
#
#        chain = prompt | self.llm | StrOutputParser()
#
#        response = chain.invoke({
#            "context": state["context"],
#            "chat_history": state.get("chat_history", []),
#            "input": state["input"],
#        })
#
#        updated_history = list(state.get("chat_history", [])) + [
#            HumanMessage(content=state["input"]),
#            AIMessage(content=response),
#        ]
#        return {"output": response, "chat_history": updated_history}


#    def __prepare_messages(self, state: ChatState) -> dict:
#        """
#        Inject retrieved RAG context into the agent prompt.
#        """
#
#        return {
#            "messages": [
#                SystemMessage(
#                    content=f"Retrieved Context: \n{state["context"]}"
#                ),
#                HumanMessage(
#                    content=f"User Question:\n{state["input"]}"
#                )
#            ]
#        }


#    def __run_agent(self, state: ChatState) -> dict:
#        """
#        Agent Node - Runs the prebuilt ReAct agent.
#
#        The agent decides:
#        - whether tools are needed
#        - which MCP tool to call
#        - how many tool iterations are required
#        - when to stop and return a final answer to the user.
#        """
#        response = self.react_agent.invoke(
#            { "messages": state["messages"] }
#        )
#
#        return { "messages": response["messages"] }


#    def __should_continue(self, state: ChatState) -> str:
#        """
#        Router - Determines whether the graph should continue to the next node.
#        """
#        last_message = state["messages"][-1]
#
#        if (isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None)):
#            return "tools"
#
#        return "finalize"


#    def __finalize(self, state: ChatState):
#        """
#        Final Response - Extract final answer and persist chat history.
#        """
#
#        final_response = ""
#
#        for msg in reversed(state["messages"]):
#            if isinstance(msg, AIMessage):
#                if (msg.content and not getattr(msg, "tool_calls", None)):
#                    final_response = msg.content
#                    break
#
#        return {
#            "output": final_response
#        }

