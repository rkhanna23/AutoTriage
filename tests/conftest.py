"""
Shared test configuration for AutoTriage.
Ensures test_intake and test_pipeline_integration use the same DB and override,
so app.dependency_overrides doesn't conflict and reset_db works correctly.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.intake.database import Base, get_db
from services.intake.main import app

TEST_DB_URL = "sqlite:///./test_autotriage.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Set override once for all intake tests
app.dependency_overrides[get_db] = override_get_db
Base.metadata.create_all(bind=test_engine)

from fastapi.testclient import TestClient
client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db_for_intake_tests(request):
    """Reset DB before each test that uses the intake app (test_intake, test_pipeline_integration)."""
    if "test_intake" in request.module.__name__ or "test_pipeline" in request.module.__name__:
        Base.metadata.drop_all(bind=test_engine)
        Base.metadata.create_all(bind=test_engine)
    yield
