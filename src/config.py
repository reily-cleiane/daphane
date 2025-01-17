"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = "daphane"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="cleiane-projetos/daphane/vector_store:v39",
    chat_prompt_artifact="cleiane-projetos/daphane/prompt_mensagem_sistema:latest",
    # chat_temperature=1.6,
    options = {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 0.0
    },
    max_fallback_retries=1,
    numero_fragmentos_recuperados = 4,
    modelo_embed="text-embedding-ada-002",
    # modelo_embed="text-embedding-3-small",
    # eval_model="gpt-4o-mini",
    # eval_artifact="cleiane-projetos/daphane/generated_examples:latest",
    modelo_resposta="llama3.1"

)