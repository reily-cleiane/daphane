"""Ingest a directory of documentation files into a vector store and store the relevant artifacts in Weights & Biases"""
import argparse
import logging
import time
import os
import pathlib
from typing import List, Tuple

import langchain
import wandb
from langchain_community.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
import shutil
from config import default_config
import re

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (EasyOcrOptions, PdfPipelineOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)

def converter_pdf_para_markdown(caminho_documentos_base):
    docs_dir = pathlib.Path(caminho_documentos_base)

    if not docs_dir.is_dir():
        logger.error(f"Diretório {docs_dir} não encontrado.")
        return
    
    # Docling Parse with EasyOCR
    # ----------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=["pt"])
    pipeline_options.ocr_options = ocr_options
    # pipeline_options.accelerator_options = AcceleratorOptions(
    #     num_threads=4, device=Device.AUTO
    # )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    ## Export results
    output_dir = docs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in docs_dir.glob("*.pdf"):
        try:
            logger.info(f"Convertendo {pdf_file.name}...")
            start_time = time.time()
            
            # Converter o PDF
            conv_result = doc_converter.convert(pdf_file)
            end_time = time.time() - start_time
            
            # Salvar resultado em Markdown
            doc_filename = conv_result.input.file.stem
            output_file = output_dir / f"{doc_filename}.md"
            with output_file.open("w", encoding="utf-8") as fp:
                fp.write(conv_result.document.export_to_markdown())

            logger.info(f"{pdf_file.name} convertido em {output_file} em {end_time:.2f} segundos.")
        
        except Exception as e:
            logger.error(f"Erro ao converter {pdf_file.name}: {e}")


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


def load_documents(data_dir: str) -> List[Document]:
    """Load documents from a directory of markdown files

    Args:
        data_dir (str): The directory containing the markdown files

    Returns:
        list[str], List[Document]: Nomes dos arquivos recuperados .md recuperados. A list of documents
    """
    md_files = list(map(str, pathlib.Path(data_dir).glob("*.md")))
    documents = [
        UnstructuredMarkdownLoader(file_path=file_path).load()[0]
        for file_path in md_files
    ]
    for doc in documents:
        doc.page_content = md_to_plain_text(doc.page_content)

    # print("\n\n ======== documentos ============== \n\n", documents)     
    return md_files, documents

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
from docling.chunking import HierarchicalChunker

def load_documents_com_docling(data_dir: str) -> Tuple[List[str], List[DoclingDocument]]:
    """Carrega documentos markdown em um diretório específico

    Args:
        data_dir (str): O diretório em que estão os documentos

    Returns:
        List[str]: uma lista de urls/caminhos para os arquivos .md.
        List[DoclingDocument]: Uma lista de objetos docling_core.types.doc.document.DoclingDocument
        gerados a partir de arquivos .md.
    """

    md_files = list(map(str, pathlib.Path(data_dir).glob("*.md")))

    conversor = DocumentConverter()
    documents = []
    for url_arquivo_md in md_files:
        result = conversor.convert(url_arquivo_md)
        documents.append(result.document)
    return md_files, documents

def chunk_documents_com_docling(
    urls_md_files: List[str],
    documents: List[DoclingDocument]
) -> List[Document]:
    """Divide os arquivos em fragmentos de forma hierárquica e converte
       em langchain.docstore.document.Document

    Args:
        List[str]: uma lista de urls/caminhos para os arquivos .md.
        documents List[DoclingDocument]: Uma lista de objetos docling_core.types.doc.document.DoclingDocument
        gerados a partir de arquivos .md.

    Returns:
        List[Document]: Uma lista de documento langchain.docstore.document.Document, sendo cada um
        um fragmento gerado por particionamento hierárquico de um arquivo .md.
    """
    
    # Tem que ver se tem alguma configuração de hierarquia aqui,
    # pra ficar de um jeito que você goste.
    particionador_hierarquico = HierarchicalChunker()
    
    documentos_langchain = []
    
    for idx in range(len(urls_md_files)):
        path = urls_md_files[idx]
        doc = documents[idx]
        chunks = particionador_hierarquico.chunk(dl_doc=doc)
        
        for chunk in chunks:
            # Usando direto o método HierarchicalChunker.serialize
            # >> Ele mantém o cabeçalho da seção em fragmento criado. Cada fragmento
            #    é correspondente, pelo que entendi, a uma linha da seção. Acho que
            #    deve ter como configurar            
            texto_formatado = particionador_hierarquico.serialize(chunk=chunk)
            
            # Aqui eu acho que consegui fazer o negócio que você faz de formatar os documentos
            # em plain text no método original
            plain_text = md_to_plain_text(texto_formatado)
            
            doc_langchain = Document(
                page_content=plain_text,
                metadata={'source': path}
            )
            
            documentos_langchain.append(doc_langchain)
            
    return documentos_langchain

def chunk_documents(
    documents: List[Document], chunk_size: int = 500, chunk_overlap=0
) -> List[Document]:
    """Split documents into chunks

    Args:
        documents (List[Document]): A list of documents to split into chunks
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 0.

    Returns:
        List[Document]: A list of chunked documents.
    """
    markdown_text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = markdown_text_splitter.split_documents(documents)
    return split_documents


def create_vector_store(
    documents,
    vector_store_path: str = "./vector_store",
) -> Chroma:
    """Create a ChromaDB vector store from a list of documents

    Args:
        documents (_type_): A list of documents to add to the vector store
        vector_store_path (str, optional): The path to the vector store. Defaults to "./vector_store".

    Returns:
        Chroma: A ChromaDB vector store containing the documents.
    """
    
    # Limpar o diretório de persistência
    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)

    api_key = os.environ.get("OPENAI_API_KEY", None)
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key, model=default_config.modelo_embed)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=vector_store_path,
    )
    # LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
    # vector_store.persist()
    return vector_store


def log_dataset(documents: List[Document], run: "wandb.run"):
    """Log a dataset to wandb

    Args:
        documents (List[Document]): A list of documents to log to a wandb artifact
        run (wandb.run): The wandb run to log the artifact to.
    """
    document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
    with document_artifact.new_file("documents.json") as f:
        for document in documents:
            f.write(document.model_dump_json() + "\n")

    run.log_artifact(document_artifact)


def log_index(vector_store_dir: str, run: "wandb.run"):
    """Log a vector store to wandb

    Args:
        vector_store_dir (str): The directory containing the vector store to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)

def log_arquivos_base(arquivos: list[str], run: "wandb.run"):
    """Loga os arquivos base do RAG para o wandb

    Args:
        arquivos (list[str]): nomes dos arquivos
        run (wandb.run): The wandb run to log the artifact to.
    """
    arquivos_artifact = wandb.Artifact(name="arquivos_base", type="dataset")
    with arquivos_artifact.new_file("arquivos_base.txt") as f:
        for arquivo in arquivos:
            f.write(arquivo + "\n")

    run.log_artifact(arquivos_artifact)

def log_prompt_mensagem_sistema(run: "wandb.run"):
    """Loga o prompt utilizado como mensagem do sistema no wandb

    Args:
        run (wandb.run): The wandb run to log the artifact to.
    """
    artefato = wandb.Artifact(name="prompt_mensagem_sistema", type="prompt")
    artefato.add_file("prompt_mensagem_sistema.txt")
    run.log_artifact(artefato)


# def log_prompt(prompt: dict, run: "wandb.run"):
#     """Log a prompt to wandb

#     Args:
#         prompt (str): The prompt to log
#         run (wandb.run): The wandb run to log the artifact to.
#     """
#     prompt_artifact = wandb.Artifact(name="chat_prompt", type="prompt")
#     with prompt_artifact.new_file("prompt.json") as f:
#         f.write(json.dumps(prompt))
#     run.log_artifact(prompt_artifact)


def ingest_data(
    docs_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    vector_store_path: str,
) -> Tuple[List[Document], Chroma]:
    """Ingest a directory of markdown files into a vector store

    Args:
        docs_dir (str):
        chunk_size (int):
        chunk_overlap (int):
        vector_store_path (str):

    """

    # converter_pdf_para_markdown(docs_dir)
    # load the documents
    # md_files, documents = load_documents(docs_dir)
    md_files, documents = load_documents_com_docling(docs_dir)
    # split the documents into chunks
    # split_documents = chunk_documents(documents, chunk_size, chunk_overlap)
    split_documents = chunk_documents_com_docling(md_files, documents)
    # create document embeddings and store them in a vector store
    vector_store = create_vector_store(split_documents, vector_store_path)
    return split_documents, vector_store, md_files


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        # required=True,
        default="../documentos_base",
        help="Diretório contendo os documentos base em formato md",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=450,
        help="The number of tokens to include in each document chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="The number of tokens to overlap between document chunks",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_store",
        help="The directory to save or load the Chroma db to/from",
    )
    # parser.add_argument(
    #     "--prompt_file",
    #     type=pathlib.Path,
    #     default="./chat_prompt.json",
    #     help="The path to the chat prompt to use",
    # )
    parser.add_argument(
        "--wandb_project",
        default="daphane",
        type=str,
        help="The wandb project to use for storing artifacts",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.job_type = "ingest"
    args.chunking = "langchain"
    # args.chunking = "docling"
    args.modelo_embed = default_config.modelo_embed
    run = wandb.init(project=args.wandb_project, job_type="ingest", config=args)
    
    documents, vector_store, md_files = ingest_data(
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.vector_store,
    )

    log_dataset(documents, run)
    log_index(args.vector_store, run)
    # log_prompt(json.load(args.prompt_file.open("r")), run) # esse aqui é pra permanecer removido
    log_prompt_mensagem_sistema(run)
    log_arquivos_base(md_files, run)
    run.finish()


if __name__ == "__main__":
    main()
