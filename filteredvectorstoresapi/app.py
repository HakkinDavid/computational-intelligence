from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from fastapi import FastAPI

app = FastAPI()

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

def dataframe_a_doc_lista(input: pd.DataFrame):
    return [Document(text=str(row.text), metadata={'category': str(row.category), 'location': str(row.location)}) for row in input.itertuples(index=False)]

@app.get("/", response_class=HTMLResponse)
def abrir_local():
    return """<html>
<h1>local</h1>
</html>"""