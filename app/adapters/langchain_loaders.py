from langchain_community.document_loaders import WebBaseLoader 




def load_url_text(url: str) -> str:
    """
    Load and return raw text from a URL using LangChain.
    Returns plain text ONLY.
    """

    loader = WebBaseLoader(url)
    documents = loader.load()

    
    text = "\n\n".join(doc.page_content for doc in documents)

    return text
