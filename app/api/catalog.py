"""GET /catalog: the companies, report types and periods covered by the uploaded documents."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.catalog import catalog_response

router = APIRouter(tags=["catalog"])


@router.get("/catalog")
def catalog(request: Request):
    return catalog_response(request.app.state.store.list())
