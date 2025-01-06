"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging

import wandb
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from prompts import load_chat_prompt
from src.config import default_config

logger = logging.getLogger(__name__)

class ChromaDB:

    def __init__(self, wandb_run: wandb.run, openai_api_key: str):
       
        """
        Recupera o banco de vetores do wandb
        Args:
            run (wandb.run): An active Weights & Biases run
            openai_api_key (str): The OpenAI API key to use for embedding
        """
        vector_store_artifact_dir = wandb_run.use_artifact(
            wandb_run.config.vector_store_artifact, type="search_index"
        ).download()
        embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key, model=default_config.modelo_embed)
        self.vector_store = Chroma(
            embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
        )

        # self.logar_fontes_recuperadas(wandb_run)

    
    def logar_fontes_recuperadas(self, wandb_run: wandb.run):
        """
        Loga no wandb os nomes dos arquivos fonte do banco de vetores recuperado
        """
        source = set()
        for metadata in self.vector_store.get()["metadatas"]:
            if "source" in metadata:
                    source.add(metadata['source'])

        arquivos_artifact = wandb.Artifact(name="fontes_recuperadas", type="dataset")
        with arquivos_artifact.new_file("fontes_recuperadas.txt") as f:
            for arquivo in source:
                f.write(arquivo + "\n")

        wandb_run.log_artifact(arquivos_artifact)
         
    
    def recuperar_fragmentos_relevantes(self, pergunta: str):
        """
        Realiza consulta no banco de vetores para identificar fragmentos com similaridade com a pergunta
        """
        retriever = self.vector_store.as_retriever(search_kwargs=dict(k=3))
        retrieved_docs = retriever.get_relevant_documents(pergunta)
        return retrieved_docs



    # def load_vector_store(wandb_run: wandb.run, openai_api_key: str) -> Chroma:
    #     """Load a vector store from a Weights & Biases artifact
    #     Args:
    #         run (wandb.run): An active Weights & Biases run
    #         openai_api_key (str): The OpenAI API key to use for embedding
    #     Returns:
    #         Chroma: A chroma vector store object
    #     """
    #     # load vector store artifact
    #     vector_store_artifact_dir = wandb_run.use_artifact(
    #         wandb_run.config.vector_store_artifact, type="search_index"
    #     ).download()
    #     embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    #     # load vector store
    #     vector_store = Chroma(
    #         embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    #     )
    #     i = 0
    #     print("\n\n EXIBIR TODOS OS DOCUMENTOS\n\n ================================================")
    #     for doc in vector_store.get()["documents"]:
    #         i = i+1
    #         if i < 30:
    #             print(f"========\n{doc}\n======")
    #         else:
    #             break

    #     return vector_store

    # def recuperar_fragmentos(pergunta: str):



# def load_chain(wandb_run: wandb.run, vector_store: Chroma, openai_api_key: str):
#     """Load a ConversationalQA chain from a config and a vector store
#     Args:
#         wandb_run (wandb.run): An active Weights & Biases run
#         vector_store (Chroma): A Chroma vector store object
#         openai_api_key (str): The OpenAI API key to use for embedding
#     Returns:
#         ConversationalRetrievalChain: A ConversationalRetrievalChain object
#     """
#     retriever = vector_store.as_retriever()
#     llm = ChatOpenAI(
#         openai_api_key=openai_api_key,
#         model_name=wandb_run.config.model_name,
#         temperature=wandb_run.config.chat_temperature,
#         max_retries=wandb_run.config.max_fallback_retries,
#     )
#     chat_prompt_dir = wandb_run.use_artifact(
#         wandb_run.config.chat_prompt_artifact, type="prompt"
#     ).download()
#     qa_prompt = load_chat_prompt(f"{chat_prompt_dir}/prompt.json")
#     print(f"\n QA PROMPT ==================================================== {qa_prompt}\n")
#     adjusted_prompt = qa_prompt.format(question="{question}", context="{context}")
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         combine_docs_chain_kwargs={"prompt": qa_prompt},
#         return_source_documents=True,
#     )
#     print(f"\n QA chain ==================================================== {qa_chain}\n")
#     return qa_chain


