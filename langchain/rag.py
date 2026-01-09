import bs4
from aiohttp.web_middlewares import middleware
from langchain_classic.chains.question_answering.map_reduce_prompt import messages

import model as m
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    #query = "standard method for Task Decomposition"
    retrieved_docs = vector_store.similarity_search(last_query, k=2)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

def load_webpage_and_store(url: str):
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )

    return loader.load()

def load_pdf_and_store(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

model = m.get_gemini_model()

embedding = m.get_gemini_embeddings()

vector_store = InMemoryVectorStore(embedding)


#docs = load_webpage_and_store("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = load_pdf_and_store("../data/rag/vmware-vsphere-metro-storage-cluster-recommended-practices-white-paper.pdf")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)

tools = [retrieve_context]
middlewares = [prompt_with_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, system_prompt=prompt, tools=tools)

#user_query = "What is the standard method for Task Decomposition?"
user_query = "What's different about uniform to non-uniform metro storage clusters?"

response_output = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
for msg in response_output["messages"]:
    msg.pretty_print()


# for event in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode="values",
# ):
#     event["messages"][-1].pretty_print()