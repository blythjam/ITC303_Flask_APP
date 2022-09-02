from logging import makeLogRecord
from typing_extensions import assert_type
from urllib import response
from .LicensePlateDetection import app

def test_paid_page():
    flask_app = app
    
    with flask_app.test_client() as test_client:
        response = test_client.get('/home')       
        assert response.status_code == 200
        assert b"Car Park Software Beta" in response.data
        