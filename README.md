### V. Test the API

- Open `tp_4/app.py` and fill in the TODOs.

### VI. Test Locally

- Launch the API: `uvicorn api:app --host 0.0.0.0 --port 8181`
- Set the HOST in `app.py`
- Launch the app: `streamlit run app.py`

Here we just display the relevant documents from a user query. We don't ask the LLM to answer from these documents.

### VII. Deploy the API

- Deploy the FastAPI app:
```bash
# May change depending on your platform
# Replace <my-docker-image-name> and <my-app-name> with your initials + _api
# Example: Florian Bastin -> <my-docker-image-name>fb_api
# Replace docker buildx build --platform linux/amd64 with docker build -t if it does not work
docker buildx build --platform linux/amd64 --push -t europe-west1-docker.pkg.dev/dauphine-437611/dauphine-ar/hibkab4_api:latest -f Dockerfile_api .

# Be careful, the default port is 8080 for Cloud Run.
# If you encounter an error, edit the default Cloud Run port on the interface or via command line
gcloud run deploy <my-app-name> \
    --image=<my-region>-docker.pkg.dev/<my-project-id>/<my-registry-name>/<my-docker-name>:latest \
    --platform=managed \
    --region=<my-region> \
    --allow-unauthenticated \
    --set-env-vars GOOGLE_API_KEY=[INSERT_GOOGLE_API_KEY],DB_PASSWORD=[INSERT_DB_PASSWORD] \
    --port 8181



# Note that a SECRET KEY like this should be provided by GOOGLE SECRET MANAGER for more safety.
# For simplicity, we will use the env variable here.
```

- Change the HOST in your Streamlit `app.py` to the URL of the FastAPI:
Example: `HOST = "https://fb-1021317796643.europe-west1.run.app/answer"`

- Deploy the Streamlit app:
```bash
# May change depending on your platform
# Replace <my-docker-image-name> and <my-app-name> with your initials + _streamlit
# Example: Florian Bastin -> <my-docker-image-name>fb_streamlit
# Replace docker buildx build --platform linux/amd64 with docker build -t if it does not work
docker buildx build --platform linux/amd64 --push -t europe-west1-docker.pkg.dev/dauphine-437611/dauphine-ar/hibkab4-streamlit:latest -f Dockerfile .

gcloud run deploy <initials>-streamlit \
    --image=europe-west1-docker.pkg.dev/dauphine-437611/dauphine-ar/<initials>-streamlit:latest \
    --platform=managed \
    --region=europe-west1 \
    --allow-unauthenticated \
    --port 8080
```
