# from langchain.chains import ConversationalRetrievalChain
# from langchain_openai import ChatOpenAI
# from prompts import load_chat_prompt

import logging
import wandb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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
        retriever = self.vector_store.as_retriever(search_kwargs=dict(k=2))
        retrieved_docs = retriever.get_relevant_documents(pergunta)
        return retrieved_docs

