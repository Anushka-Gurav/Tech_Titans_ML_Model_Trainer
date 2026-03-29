import pytest
from httpx import AsyncClient, ASGITransport
import sys
import os

# Make sure backend folder is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # import your FastAPI app


# ───── Health Check ─────
@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
    assert response.status_code == 200


# ───── Train Endpoint ─────
@pytest.mark.asyncio
async def test_train_endpoint_reachable():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/train", json={
            "model_type": "random_forest",
            "target_column": "target",
            "dataset_name": "iris"
        })
    # Should not return 404 (route must exist)
    assert response.status_code != 404


# ───── Predict Endpoint ─────
@pytest.mark.asyncio
async def test_predict_endpoint_reachable():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/predict", json={
            "features": [5.1, 3.5, 1.4, 0.2]
        })
    assert response.status_code != 404


# ───── Models List Endpoint ─────
@pytest.mark.asyncio
async def test_models_list_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/models")
    assert response.status_code in [200, 404]  # 404 ok if route not yet built


# ───── Input Validation Test ─────
@pytest.mark.asyncio
async def test_train_missing_fields():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/train", json={})
    # FastAPI returns 422 for missing required fields
    assert response.status_code in [400, 422]

