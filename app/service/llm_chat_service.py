from model.enum.llm_enum import LLMProviders
from repository.mongodb_chat_history_repository import MongoChatHistoryRepository
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from typing import TypedDict, Sequence
from pathlib import Path

class ChatState(TypedDict):
    input: str
    chat_history: Sequence[BaseMessage]
    context: str
    output: str
    source_documents: Sequence[Document]

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
        self.__graph = self.__build_graph()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
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
        context = "\n\n".join(doc.page_content for doc in docs)
        source_docs = list(set([Path(doc.metadata['source_file']).name for doc in docs]))
        return {
            "context": context,
            "source_documents": source_docs
        }


    def __get_chat_prompt_template(self):
        """
        Returns a chat prompt template for GenAI conversation
        """
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the user's question.\n\nContext: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])


    def __generate(self, state: ChatState) -> dict:
        """
        Generates a response to the user by calling the LLM with chat history + context.
        After that, appends the new turn to history so the checkpointer persists it

        Args:
            state: ChatState - A typed dictionary containing at least the key "input" with the user's query as its value.

        Returns:
            dict - A Dictionary containing the LLM's response and the chat history
        """

        prompt = self.__get_chat_prompt_template()

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": state["context"],
            "chat_history": state.get("chat_history", []),
            "input": state["input"],
        })

        updated_history = list(state.get("chat_history", [])) + [
            HumanMessage(content=state["input"]),
            AIMessage(content=response),
        ]
        return {"output": response, "chat_history": updated_history}


    def __build_graph(self) -> StateGraph:
        """
        Build and compile the LangGraph with MongoDB checkpointing.

        Returns:
            StateGraph - The compiled LangGraph instance ready for invocation.
        """
        workflow = StateGraph(ChatState)

        workflow.add_node("retrieve", self.__retrieve)
        workflow.add_node("generate", self.__generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # MongoDB checkpointer replaces RunnableWithMessageHistory
        checkpointer = MongoDBSaver(
            self.__chatHistoryRepository.get_client(),  # expose the raw client
            db_name=self.__chatHistoryRepository.get_database_name(),
        )
        return workflow.compile(checkpointer=checkpointer)


    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def generate_response(
        self,
        session_id: str,
        query: str
    ) -> dict[str]:
        """
        Enrich context by retrieving relevant documents from Vector Database, pass 
        chat history and user input and then generates a response to the user's 
        query by using an LLM to formulate the answer.

        Args:
            session_id: str - The ID of the session for which the response is being generated.
            query: str - The user's query

        Returns:
            dict - The generated response from the LLM based on the retrieved context, along 
            with the chat history and augmented with "source_documents" (the retrieved documents)
        """

        config = {"configurable": {"thread_id": session_id}}
        result = self.__graph.invoke({"input": query}, config=config)
        return {
            "output": result["output"],
            "source_documents": result["source_documents"]
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
        return state.values.get("chat_history", [])
