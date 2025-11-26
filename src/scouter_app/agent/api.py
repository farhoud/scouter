from fastapi import APIRouter
from typing import List
from ..shared.domain_models import SearchResult
from .service import SearchService

router = APIRouter()

service = SearchService()


@router.get("/v1/search", response_model=List[SearchResult])
async def search_documents(query: str, limit: int = 10):
    """
    Search documents in the knowledge graph.
    """
    return service.search(query, limit)
