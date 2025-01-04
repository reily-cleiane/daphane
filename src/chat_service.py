"""A Simple chatbot that uses the LangChain and Gradio UI to answer questions about wandb documentation."""
import json
import os
from types import SimpleNamespace
import re

import wandb
from src.chromadb_service import ChromaDB
from src.ollama import Ollama
from src.config import default_config
import logging

logger = logging.getLogger(__name__)

class ChatService:
    """A chatbot interface that persists the vectorstore and chain between calls."""

    def __init__(self):
        """Initialize the chatbot.
        Args:
            config (SimpleNamespace): The configuration.
        """
        self.config = default_config
        self.wandb_run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            job_type=self.config.job_type,
            config=self.config,
        )
        self.chain = None
        self.tabela_requisicao = wandb.Table(columns=["pergunta", "fragmentos_recuperados", "resposta_modelo"])

        if os.environ["OPENAI_API_KEY"]:
            openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("\nCHAVE OPENAI NÃO ENCONTRADA NA VARIÁVE DE AMBIENTE OPENAI_API_KEY\n")
        
        self.chroma_service = ChromaDB(wandb_run=self.wandb_run, openai_api_key=openai_key)
        self.ollama_service = Ollama(wandb_run=self.wandb_run)

    
    def __call__(
        self,
        pergunta: str,
        historico: list[tuple[str, str]] | None = None,
    ):
        """Respode uma pergunta sobre violência doméstica usando a lei maria da penha.
        Args:
            pergunta (str): Pergunta a ser respondida.
            history (list[tuple[str, str]] | None, optional): Histórico do chat. Defaults to None.
            openai_api_key (str, optional): Chave da OPENAI. Defaults to None.
        Returns:
            list[tuple[str, str]], list[tuple[str, str]]: The chat history before and after the question is answered.
        """

        historico = historico or []
        pergunta = pergunta.lower()

        retrieved_docs = self.chroma_service.recuperar_fragmentos_relevantes(pergunta)

        # for doc in retrieved_docs:
        #     print(doc.metadata["source"])
        #     print(f"\n\nInício do contexto: =======\n {doc.page_content}\nFim do contexto ======\n\n") 

        contexto = self.recuperar_conteudo_arquivos(retrieved_docs)

        resposta = self.ollama_service.responder(pergunta, contexto, historico)
        # print(f"\n\n resposta ============================== \n\n{resposta}")

        self.tabela_requisicao.add_data(pergunta, (doc.metadata["documents"] for doc in retrieved_docs) , resposta)
        wandb.log({"Tabela_Requisicao": self.tabela_requisicao})
        historico.append((pergunta, resposta))
        print("\n\n ============ resposta ============ \n\n", resposta)
        
        msg = {
            "tipo": "dados",
            "descricao": "Mensagem do LLM",
            "dados":{
                "resposta": resposta
            }
        }
        return json.dumps(msg)


        # # Obtenha a resposta, passando os documentos e o contexto
        # response = get_answer(
        #     chain=self.chain,
        #     question=question,
        #     chat_history=history,
        #     retrieved_docs=retrieved_docs,
        #     context=combined_context,
        # )
        # print(f"\n\nCHAT HISTORY =================================== {history}\n\n")
        # history.append((question, response))
        # return history, history

    @staticmethod
    def recuperar_conteudo_arquivos(retrieved_docs):
        unique_sources = set(doc.metadata["source"] for doc in retrieved_docs)
        conteudo = []
        for source_file in unique_sources:
            with open("./src/"+source_file, "r") as f:
                conteudo.append(ChatService.md_to_plain_text(f.read()))
        conteudo_combinano = "\n".join(conteudo)
        return conteudo_combinano
    
    @staticmethod
    def md_to_plain_text(md_text):
        """
        Converte texto Markdown para texto plano, preservando níveis de cabeçalho até o terceiro nível.
        
        Args:
            md_text (str): Texto em Markdown.

        Returns:
            str: Texto convertido em texto plano.
        """
        # Substituir cabeçalhos de nível 1
        plain_text = re.sub(r'^# (.+)', r'\1\n', md_text, flags=re.M)
        # Substituir cabeçalhos de nível 2
        plain_text = re.sub(r'^## (.+)', r'  \1\n', plain_text, flags=re.M)
        # Substituir cabeçalhos de nível 3
        plain_text = re.sub(r'^### (.+)', r'    \1\n', plain_text, flags=re.M)
        # Substituir múltiplas quebras de linha por uma única
        plain_text = re.sub(r'\n+', '\n', plain_text)

        return plain_text.strip()
