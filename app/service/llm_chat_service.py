import json
from pathlib import Path
from typing import Annotated, TypedDict, Sequence, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver

from model.enum.llm_enum import LLMProviders
from model.enum.mcp_tool_enum import MCPToolsEnum
from repository.mongodb_chat_history_repository import MongoChatHistoryRepository
from repository.vector_db_repository import VectorDBRepository
from service.mcp_service import MCPService

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
        vector_db_repository: VectorDBRepository,
        llm_provider: LLMProviders, 
        model_name: str, 
        temperature: float = 0.2
    ):
        self.__chatHistoryRepository = MongoChatHistoryRepository()
        self.vector_db_repository = vector_db_repository
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.llm = llm_provider.get_llm_instance(model_name, temperature)
        self.question_generator_mcp_service = MCPService(server_name=MCPToolsEnum.QUESTION_GENERATOR)
        self.__graph_exam = None
        self.__graph_regular = None


    async def get_graph_exam(self) -> StateGraph:
        if self.__graph_exam is None:
            tools = await self.question_generator_mcp_service.get_available_tools()

            for tool in tools:
                if tool.name == "generate_exam_html":
                    tool.return_direct = True

            tools_agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=self.__agent_prompt(enable_exam_mode=True)
            )
            self.__graph_exam = self.__build_graph(tools_agent)
        return self.__graph_exam


    def get_graph_regular(self) -> StateGraph:
        if self.__graph_regular is None:
            # create_agent() returns a Runnable, so we can directly use it as a node in the graph.
            regular_agent = create_agent(
                model=self.llm,
                tools=[],
                system_prompt=self.__agent_prompt(enable_exam_mode=False)
            )
            self.__graph_regular = self.__build_graph(regular_agent)
        return self.__graph_regular


    # ------------------------------------------------------------------ #
    #  Dynamic Agent Prompt                                              #
    # ------------------------------------------------------------------ #

    def __agent_prompt(
        self,
        enable_exam_mode: bool
    ) -> SystemMessage:
        """
        Called automatically before every agent step.

        Provides the latest retrieved context to the agent.
        """

        if enable_exam_mode:
            tool_instruction = "You MUST use the available MCP tools to generate exam questions in HTML format as requested."
        else:
            tool_instruction = "Respond using the retrieved context and your knowledge."

        agent_prompt = f"""
You are a helpful assistant.

{tool_instruction}

Instructions:

- Use the retrieved context whenever relevant.
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
        docs = self.vector_db_repository.retrieve(query=state["input"])

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
        }


    # ------------------------------------------------------------------ #
    #  Context ingestion node                                            #
    # ------------------------------------------------------------------ #

    def __inject_context(self, state: ChatState) -> dict:
        human_message = HumanMessage(content=state["input"])

        context = state.get("context", "")

        system_message = SystemMessage(
            content=("No context retrieved for this turn" if not context 
                   else f"Retrieved context for this turn:\n\n{context}")
        )
            
        return {
            "messages": [ system_message, human_message ]
        }


    # ------------------------------------------------------------------ #
    #  Graph                                                             #
    # ------------------------------------------------------------------ #

    def __build_graph(
        self, 
        agent: CompiledStateGraph
    ) -> StateGraph:
        """
        Build and compile the LangGraph with MongoDB checkpointing.

        Params:
            agent: CompiledStateGraph - Final executable instance of the agent workflow in LangGraph

        Returns:
            StateGraph - The compiled LangGraph instance ready for invocation.
        """
        workflow = StateGraph(ChatState)

        workflow.add_node("retrieve", self.__retrieve)
        workflow.add_node("inject_context", self.__inject_context)

        # The agent node will automatically decide when to call tools based on the prompt and the retrieved 
        # context, so we just need to wire it up as the main driver of the conversation after retrieval.
        workflow.add_node("agent", agent)

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
        query: str,
        enable_exam_mode: bool
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
            enable_exam_mode: bool - Toggle to activate tools to generate exam questions in html format

        Returns:
            dict - The generated response from the LLM based on the retrieved context, along 
            with the chat history and augmented with "source_documents" (the retrieved documents)
        """

        config = {"configurable": {"thread_id": session_id}}

        graph_exam = await self.get_graph_exam()
        graph_regular = self.get_graph_regular()
        graph = graph_exam if enable_exam_mode else graph_regular

        result = await graph.ainvoke({"input": query}, config=config)

        self.__log_graph_state(graph, config, query)

        print("DEBUG - Full agent response:", result)

        final_response = ""

        messages = result.get("messages", [])
        for msg in reversed(messages):
            isValidInstance = isinstance(msg, AIMessage) or (isinstance(msg, ToolMessage) and enable_exam_mode)
            if isValidInstance and msg.content:
                final_response = msg.content
                break

        return {
            "output": final_response,
            "exam_mode": enable_exam_mode,
            "source_documents": result.get("source_documents", [])
        }


    def generate_exam_context_for_responses(
        self, 
        session_id: str,
    ) -> list[str]: 
        """
        Extract overall context of the Vector Database and summarizes it in a bulleted list

        Args:
            session_id: str - The ID of the session for which the response is being generated.

        Returns:
            list[str] - The retrieved and summarized context formatted as a list of topics.
        """
        context_docs = self.vector_db_repository.get_collection_topics(
            session_id=session_id
        )
        context_text = "\n---\n".join([doc.page_content for doc in context_docs])

        chat_prompt_template = """
You are given a list of text fragments extracted from a database vector store.
Analyze the texts and summarize their main topics or themes.
Return them as a json list with a 1-sentence explanation for each **WITHOUT ANY EXPLANATION OF THE CONTENT, JUST THE CRUDE JSON**.
Example response:
[
    "topic lorem - simply dummy text of the printing and typesetting industry",
    "Survival - It has survived not the leap into electronic typesetting"
]
Texts:
{context}
"""

        prompt = ChatPromptTemplate.from_template(chat_prompt_template)

        chain = prompt | self.llm

        response = chain.invoke({"context": context_text})

        try:
            response_content = response.content

            response_body = response_content[response_content.find('['):response_content.find(']')+1].replace("\n", "")

            db_context_topics = json.loads(response_body)

            return db_context_topics
        except Exception as e:
            print(f"Error getting source document context. Details: {e}")


    def get_session_history(self, session_id: str) -> Sequence[BaseMessage]:
        """
        Retrieve the chat history for a given session from the LangGraph checkpoint.

        Args:
            session_id: str - The thread ID of the session to retrieve history for.

        Returns:
            Sequence[BaseMessage] - The chat history for the specified session.
        """

        config = {"configurable": {"thread_id": session_id}}
        state = self.get_graph_regular().get_state(config)
        messages = state.values.get("messages", [])

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({
                    "role": "user",
                    "content": msg.content
                })
            elif isinstance(msg, AIMessage) and msg.content:
                exam_mode = msg.additional_kwargs.get("exam_mode", False)
                history.append({
                    "role": "assistant",
                    "content": msg.content,
                    "exam_mode": exam_mode
                })
        
        return {"messages": history}


    def __log_graph_state(self, graph: StateGraph, config: dict, input_state: str):
        state = graph.get_state(config)
        print(f'State: {state}')

        history = list(graph.get_state_history(config))

        for i, snapshot in enumerate(history):
            print(f'Snapshot {i} - {snapshot}')
