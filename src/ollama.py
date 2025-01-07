# import os, random
# from pathlib import Path
# import tiktoken
# from getpass import getpass
# from rich.markdown import Markdown
# from langchain_community.vectorstores import Chroma
# from chromadb import Client
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM
# from langchain.schema import Document

import wandb
import requests
import json

from src.config import default_config

OLLAMA_SERVER_URL = "http://localhost:11434"
TAMANHO_MAXIMO_HISTORICO = 8

class Ollama:
    def __init__(self, wandb_run: wandb.run):
        chat_prompt_dir = wandb_run.use_artifact(
        wandb_run.config.chat_prompt_artifact, type="prompt"
        ).download()

        with open(f"{chat_prompt_dir}/prompt_mensagem_sistema.txt", "r") as file:
            self.system_prompt = file.read()

        self.payload = self.gerar_payload()
        
        # prompt_artifact = wandb.Artifact(name="prompt_mensagem_sistema_recuperado", type="prompt")
        # with prompt_artifact.new_file("prompt_mensagem_sistema_recuperado.txt") as f:
        #     f.write(self.system_prompt)

        # wandb_run.log_artifact(prompt_artifact)



    def responder(self, pergunta: str, contexto: str, historico: list[str]):

        if len(historico) > TAMANHO_MAXIMO_HISTORICO:
            historico.pop()
            historico.pop()

        self.payload["messages"] = self.formatar_historico_mensagens(pergunta, contexto, historico)
        print(f"\n============= Requisição ao ollama ================\nPergunta: {pergunta}\nPayload: {self.payload}\n\n")

        try:

            response = requests.post(
                f"{OLLAMA_SERVER_URL}/api/chat", 
                json=self.payload, 
                timeout=100
            )

            if response.status_code != 200:
                print("Erro na requisição:", response.text)
                return ""
            
            resposta_json, conteudo_resposta = self.tratar_resposta(response)
            
            return resposta_json, conteudo_resposta
        
        except Exception as e:
            print(f"Erro ao realizar a requisição: {e}")
            return ""
        
    
    @staticmethod
    def tratar_resposta(response):
        conteudo_resposta = ""
        for line in response.iter_lines():
            if line:  # Ignorar linhas vazias
                data = line.decode('utf-8')
                print(f"\n================= Resposta: ========================\n{data}\n\n")
                try:
                    json_data = json.loads(data)
                    # Verifica e adiciona o conteúdo
                    if "message" in json_data and "content" in json_data["message"]:
                        conteudo_resposta += json_data["message"]["content"]
                except Exception as e:
                    print(f"Erro ao processar linha: {data}, Erro: {e}")
                    continue
        # resposta da requisição com todos os atributos (in)
        return json_data, conteudo_resposta.strip()

    
    def gerar_payload(self):
        payload = {
            "model": default_config.modelo_resposta,
            # "system": mensagem_sistema,
            # "prompt":  mensagem_usuario,
            "options": {
                "temperature": default_config.options["temperature"],
                "top_k": default_config.options["top_k"],
                "top_p": default_config.options["top_p"]
            },
            "stream": False, # Se for usar tool precisa ser false
        }
        return payload
    
    
    def formatar_historico_mensagens(self, pergunta, contexto, historico):
        # mensagens = [{"role": "system", "content": self.system_prompt + 
        # "Você pode dar outras informações genéricas humanizadas, mas use obrigatoriamente esse fragmento da lei maria da penha para responder a pergunta da usuária:: " + 
        # contexto + ". Se esse fragmento da lei apresentado anteriormente não contiver a resposta para a pergunta da usuária, informe a usuária que você não tem dados suficientes para responder a pergunta."}]
        mensagens = [{"role": "system", "content": self.system_prompt }]

        for indice, mensagem in enumerate(historico):
            if indice % 2 == 0:
                mensagens.append({"role": "user", "content": mensagem})
            else:
                mensagens.append({"role": "assistant", "content": mensagem})
        
        # mensagens.append({"role": "user", "content": pergunta})
        mensagens.append({"role": "user", "content": 
            "Use esse texto referencial para embasar sua resposta: "+ contexto +
            "Não mencione que houve um texto referencial na sua resposta. "+
            "Utilize linguagem simples e evite o uso de palavras difíceis ou rebuscadas, jargões ou termos técnicos. "+
            "Demonstre empatia e sensibilidade. "+
            "Responda essa pergunta: " + pergunta})
        return mensagens
