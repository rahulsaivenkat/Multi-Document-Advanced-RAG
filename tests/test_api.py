"""Tests for FastAPI endpoints using TestClient."""
import io
from pathlib import Path
from fastapi.testclient import TestClient


def test_api_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "embedding_model" in data
    # Ensure no secrets exposed
    assert "api_key" not in str(data).lower()


def test_api_upload_and_documents_flow(client: TestClient, sample_txt: Path, sample_csv: Path):
    txt_bytes = sample_txt.read_bytes()
    csv_bytes = sample_csv.read_bytes()

    # Upload two files
    files = [
        ("files", ("test_txt.txt", txt_bytes, "text/plain")),
        ("files", ("test_csv.csv", csv_bytes, "text/csv")),
    ]
    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 200
    data = upload_res.json()
    assert len(data["uploaded"]) == 2
    assert data["uploaded"][0]["status"] == "success"
    doc_id_1 = data["uploaded"][0]["document_id"]
    doc_id_2 = data["uploaded"][1]["document_id"]

    # List documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total_documents"] >= 2
    doc_ids = [d["document_id"] for d in list_data["documents"]]
    assert doc_id_1 in doc_ids
    assert doc_id_2 in doc_ids

    # Query with chat
    chat_payload = {
        "question": "What are the architecture principles for RAG?",
        "document_ids": [doc_id_1]
    }
    chat_res = client.post("/chat", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "answer" in chat_data
    assert "citations" in chat_data
    assert chat_data["diagnostics"] is not None
    assert chat_data["diagnostics"]["semantic_results_count"] >= 1

    # Delete doc 1
    del_res = client.delete(f"/documents/{doc_id_1}")
    assert del_res.status_code == 200
    assert "successfully purged" in del_res.json()["message"]

    # Delete non-existent doc -> 404
    del_nonexistent = client.delete("/documents/non_existent_doc_id")
    assert del_nonexistent.status_code == 404


def test_api_chat_empty_question_validation(client: TestClient):
    res = client.post("/chat", json={"question": "   "})
    assert res.status_code == 422 or res.status_code == 400
