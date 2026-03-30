import os
import pickle
from typing import Iterable, Optional

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def create_service(
    client_secret_file: str,
    api_name: str,
    api_version: str,
    scopes: Iterable[str],
) -> Optional[object]:
    """Authenticate and build a Google API service client."""
    credentials = None
    pickle_file = f"token_{api_name}_{api_version}.pickle"

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, list(scopes))
            credentials = flow.run_local_server()

        with open(pickle_file, "wb") as token:
            pickle.dump(credentials, token)

    try:
        return build(api_name, api_version, credentials=credentials)
    except Exception:
        return None
