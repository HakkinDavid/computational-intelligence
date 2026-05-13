from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from uuid import uuid4
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class DocumentoInput(BaseModel):
    text: str
    category: str
    location: str


class BusquedaInput(BaseModel):
    query: str
    top_k: int = 5
    metadata_filter: dict[str, str] | None = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

class Document:
    def __init__(self, text: str, metadata: dict[str, str]):
        self.text = text
        self.metadata = metadata


class SearchResult:
    def __init__(self, score: float, document: Document):
        self.score = score
        self.document = document
        
class FilteredVectorStore:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.documents: list[Document] = []
        self.embeddings = None

    def add_documents(self, documents: list[Document]):
        self.documents.extend(documents)

        embebido = self.embedding_model.encode(
            [str(document.text) for document in documents], # ya mejor ni me quejo
            convert_to_tensor=True
        )

        self.embeddings = embebido if self.embeddings is None else torch.cat((self.embeddings, embebido), dim=0) # otro gato diría el chavo

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        if len(self.documents) == 0:
            return []

        filtros_aplicados = [
            indicini
            for indicini, documentini in enumerate(self.documents) if (metadata_filter is None or all(documentini.metadata.get(llave) == valor for llave, valor in metadata_filter.items()))
        ]

        if not filtros_aplicados:
            return []

        valores, ubicaciones = torch.topk(
            torch.nn.functional.normalize(self.embeddings[filtros_aplicados], p=2) # type: ignore
            @
            torch.nn.functional.normalize(self.embedding_model.encode(query,convert_to_tensor=True),p=2,dim=0),
            min(top_k, len(filtros_aplicados))
        )

        return [
            SearchResult(float(v), self.documents[filtros_aplicados[int(u)]]) # ya ni me quejo 2
            # yo cuando le sacan secuela a mi queja
            for v, u in zip(valores, ubicaciones)
        ]

tienda_vector = FilteredVectorStore(embedding_model=SentenceTransformer("all-MiniLM-L6-v2", device=device.type))

documentos_originales = {}

def dataframe_a_doc_lista(input: pd.DataFrame):
    return [Document(text=str(row.text), metadata={'category': str(row.category), 'location': str(row.location)}) for row in input.itertuples(index=False)]


def fragmentar(texto: str):
    return [texto] if len(texto) <= 500 else [texto[i:i+400] for i in range(0, len(texto), 400)]


@app.post("/documents")
def crear_documento(documento: DocumentoInput):
    document_id = str(uuid4())
    fragmentos = fragmentar(documento.text)

    documentos_originales[document_id] = {
        'id': document_id,
        'text': documento.text,
        'metadata': {
            'category': documento.category,
            'location': documento.location
        }
    }

    tienda_vector.add_documents([
        Document(
            text=fragmento,
            metadata={
                'document_id': document_id,
                'category': documento.category,
                'location': documento.location
            }
        )
        for fragmento in fragmentos
    ])

    return {
        'id': document_id,
        'chunks': len(fragmentos)
    }


@app.get("/documents/{document_id}")
def obtener_documento(document_id: str):
    documento = documentos_originales.get(document_id)

    if documento is None:
        raise HTTPException(status_code=404, detail='Documento no encontrado')

    return documento


@app.post("/documents/search")
def buscar_documentos(busqueda: BusquedaInput):
    return [
        {
            'score': resultado.score,
            'text': resultado.document.text,
            'metadata': resultado.document.metadata
        }
        for resultado in tienda_vector.search(
            query=busqueda.query,
            top_k=busqueda.top_k,
            metadata_filter=busqueda.metadata_filter
        )
    ]


@app.get("/", response_class=HTMLResponse)
def abrir_local():
    with open("static/index.html") as html:
        return html.read()
    
@app.get("/documents/all")
def todos():
    return list(documentos_originales.values())