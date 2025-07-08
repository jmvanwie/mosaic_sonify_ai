import os
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

def run_test():
    """
    A simple, standalone script to test the direct connection to Firestore.
    """
    print("--- Starting Firebase Connection Test ---")
    try:
        # 1. Construct an absolute path to the credentials file
        basedir = os.path.abspath(os.path.dirname(__file__))
        key_path = os.path.join(basedir, "serviceAccountKey.json")
        print(f"Attempting to load credentials from: {key_path}")

        # 2. Check if the file exists before trying to use it
        if not os.path.exists(key_path):
            print("\nERROR: The file 'serviceAccountKey.json' was not found at the specified path.")
            print("Please ensure the key file is in the same folder as this script and is named correctly.")
            return

        # 3. Initialize the Firebase app
        cred = credentials.Certificate(key_path)
        # Check if an app is already initialized to avoid errors
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'projectId': 'mosaic-tess-1',
            })
        
        print("Firebase app initialized successfully.")

        # 4. Get a client and try to write data
        db = firestore.client()
        test_collection = db.collection('test_connection')
        test_doc_id = f"test-{uuid.uuid4()}"
        doc_ref = test_collection.document(test_doc_id)

        print(f"Attempting to write to document: {test_doc_id}")
        doc_ref.set({
            'message': 'Connection successful!',
            'status': 'OK'
        })

        # 5. Report success
        print("\n--- TEST SUCCESSFUL! ---")
        print("Successfully wrote data to your Firestore database.")
        print("This confirms your credentials and network connection are working.")
        print("The issue likely lies within the main application's configuration.")

    except Exception as e:
        # 6. Report failure
        print(f"\n--- TEST FAILED ---")
        print(f"An error occurred: {e}")
        print("\nThis likely confirms a network issue between your computer and Google Cloud,")
        print("such as a firewall, proxy, DNS problem, or ISP blocking.")

if __name__ == '__main__':
    run_test()
