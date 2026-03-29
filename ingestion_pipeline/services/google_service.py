import pickle  # used to save/load credentials (so you don't log in every time)
import os      # used to check if files exist
from google_auth_oauthlib.flow import Flow, InstalledAppFlow  # handles OAuth login flow
from googleapiclient.discovery import build  # creates the API service object
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload  # for uploading/downloading files
from google.auth.transport.requests import Request  # used to refresh expired tokens
import datetime  # needed for the datetime function below


def Create_Service(client_secret_file, api_name, api_version, *scopes):
    """
    This function:
    - Authenticates with Google
    - Saves credentials locally
    - Returns a service object to interact with the API
    """

    # Print inputs (just for debugging)
    print(client_secret_file, api_name, api_version, scopes, sep='-')

    CLIENT_SECRET_FILE = client_secret_file   # path to your credentials.json
    API_SERVICE_NAME = api_name               # e.g. 'drive'
    API_VERSION = api_version                 # e.g. 'v3'

    # scopes define what permissions your app has (e.g. read-only drive access)
    SCOPES = [scope for scope in scopes[0]]
    print(SCOPES)

    cred = None  # this will hold your credentials

    # File where credentials will be saved locally
    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'

    #  STEP 1: Load existing credentials if they exist
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    # STEP 2: If no valid credentials, log in
    if not cred or not cred.valid:

        # If credentials exist but expired → refresh them automatically
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())

        else:
            # Otherwise → open browser for login
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES
            )
            cred = flow.run_local_server()  # opens browser to sign in

        #  STEP 3: Save credentials so you don't log in again next time
        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    #  STEP 4: Build the API service (this is what you actually use)
    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, 'service created successfully')
        return service

    except Exception as e:
        print('Unable to connect.')
        print(e)
        return None


def convert_to_RFC_datetime(year=1900, month=1, day=1, hour=0, minute=0):
    """
    Converts a normal date into RFC3339 format (required by Google APIs)

    Example output:
    '2026-03-27T15:30:00Z'
    """

    dt = datetime.datetime(year, month, day, hour, minute, 0).isoformat() + 'Z'
    return dt