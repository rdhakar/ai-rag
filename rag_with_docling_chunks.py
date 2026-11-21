#uv add qdrant-client sentence-transformers docling groq docling-hierarchical-pdf

import os
from xmlrpc import client

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from groq import Groq
from hierarchical.postprocessor import ResultPostprocessor
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from semchunk import Chunker
from sentence_transformers import SentenceTransformer
from sympy.physics.mechanics import system

from dotenv import load_dotenv
load_dotenv()



SOURCE = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/AtliqAI_HR_Policies.pdf"
EMBEDDING_MODEL="all-MiniLM-L6-v2"

SYSTEM_PROMPT = """
Your are a helpful HR assistant.
Answer the your question using ONLY the context provided below.
if the context does not contain enough information,  say so  - do not make things up.
Always cite the section name with referencing specific inromation.  
"""

def load_document(source: str):
    """
    Parse a PDF using Docling.
    Returns a DoclingDocument object — not a plain string.
    """
    converter = DocumentConverter()
    result = converter.convert(source)
    ResultPostprocessor(result).process()
    return result.document
def convert_chunk(chunk) -> dict:
    """
    convert Docling chunck to a plain dist
    headings -> Headings list of the chunk
    content -> Content of the chunk
    chunk_text -> breadcrumb (heading 1 > heading 2 >heading 3 ) +\n\n+ content (what  get embedded)
    """
    headings = chunk.meta.headings
    breadcrumb = " > ".join(headings)
    content = chunk.text.strip()
    chunk_text = f"{breadcrumb}\n\n{content}"
    return {
        "headings": headings,
        "content": content,
        "chunk_text": chunk_text
    }
def retrieve(
        query : str,
        top_k:int =5
) -> list[dict]:

    query_vector = embedder.encode(query).tolist()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True)

    return [{**hit.payload,"score": round(hit.score,4)}   for hit in hits.points]

def build_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for i in enumerate(retrieved_chunks):
        parts.append(f"[Source {i}]\n{chunk['content']}")
    return "\n\n-\n\n".join(parts)

def rag(query: str, top_k: int = 5) :

    #Step 1 - retrieve chunks
    chunks = retrieve(query, top_k=top_k)

    #Step 2 - build context
    context1 = build_context(chunks)
    user_message = f"Context:\n{context1}\n\nQuestion: {query}"
    #Step 3 - Build LLM
    groq_client = Groq()  # Reads GROQ_API_KEY from environment automatically
    GROQ_MODEL = "openai/gpt-oss-20b"
    result = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system","content": SYSTEM_PROMPT},
            {"role":"user", "content":user_message}
        ],
        temperature=0.2
    )
    return result.choices[0].message.content,context1


    #Step 4 - Generate


if __name__ == "__main__":
    print("Hi there!")
    doc = load_document(SOURCE)
    print(f"Document loaded: {doc.name}")

    print(f"Document markdown: {doc.export_to_markdown()}")
    #print(f"Document converted: {doc.export_to_html()}")
    print("*" * 80)

    chunker = HierarchicalChunker()
    chunkList = list(chunker.chunk(doc))
    print(f"Chunks loaded: {len(chunkList)}")
    print(f"2nd element from Chunk list: {chunkList[2]}")
   #print(f"Heading: {chunkList[2].meta.headings}")
    print(f"text: {chunkList[2].text}")

    convertedChunks = [convert_chunk(chunk) for  chunk in chunkList]

    for  chunk in convertedChunks[:3]:
        print("─" * 60)
        #print(f"Index {index}")
        print(chunk)
        #print(f"headings: {chunk["headings"]}")
        #print(f"content: {chunk['content']}")
        #print(f"chunk_text: {chunk['chunk_text']}")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    Only_chunk_text = [chunk["chunk_text"]  for chunk in convertedChunks]
    embeddings = embedder.encode(Only_chunk_text,show_progress_bar=True)
    print("embeddings shape:", embeddings.shape)
    client = QdrantClient(path="/tmp/my_qdrant_for_docling")
    COLLECTION_NAME ="docling_docs"
    DIM = embedder.get_embedding_dimension()
    print("DIM1:", DIM)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=DIM,
            distance=Distance.COSINE
        )
    )
    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "headings": chunk["headings"],  # stored as a JSON array
                "content": chunk["content"],
                "chunk_text": chunk["chunk_text"],
            },
            #payload=chunk  We can directly use chunk , as chunk also have same format
        )
        for idx, (chunk, embedding) in enumerate(zip(convertedChunks, embeddings))
    ]

    result = client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )
    print(f"Indexed {len(points)} points — status: {result.status}")

    info = client.get_collection(COLLECTION_NAME)
    print(f"Points     : {info.points_count}")
    print(f"Dimensions : {info.config.params.vectors.size}")

    print("_"*80)

    results = retrieve("What is the leave policy?", top_k=3)
    for r in results:
        print(f"[{r['score']}]  {r['headings']}")
        print(f"  {r['content'][:200]}…\n")
    print("-"*80)
    print("************** Build RAG PIPE Line Now and execute  ***************")

    answer, context = rag("How many casual leaves am I entitled to?")
    print(answer)
    print(f"{250 * '='}")
    print(f"\n\nSOURCES:\n {context}")











    print("Program ends")

    client.close()



