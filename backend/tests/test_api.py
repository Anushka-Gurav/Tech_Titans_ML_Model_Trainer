import pytest
from httpx import AsyncClient, ASGITransport
import sys
import os

# Add backend path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app


# ───── Root API ─────
@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/")
    assert response.status_code == 200


# ───── Models List ─────
@pytest.mark.asyncio
async def test_models_list():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/models/list")
    assert response.status_code == 200


# ───── Train Endpoint Exists ─────
@pytest.mark.asyncio
async def test_train_endpoint_exists():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/model/train", json={})
    assert response.status_code != 404


# ───── Compare Endpoint Exists ─────
@pytest.mark.asyncio
async def test_compare_endpoint_exists():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/model/compare", json={})
    assert response.status_code != 404


# ───── Invalid Progress Check ─────
@pytest.mark.asyncio
async def test_invalid_progress():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/model/progress/invalid_id")
    assert response.status_code in [200, 404]
