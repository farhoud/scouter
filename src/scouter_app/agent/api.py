from fastapi import APIRouter

from scouter_app.shared.domain_models import SearchResult
from scouter_app.agent.service import SearchService

router = APIRouter()

service = SearchService()


@router.get("/v1/search", response_model=list[SearchResult])
async def search_documents(query: str, limit: int = 10):
    """
    Search documents in the knowledge graph.
    """
    return service.search(query, limit)
