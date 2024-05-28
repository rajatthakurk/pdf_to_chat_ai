This FastAPI application leverages the power of Google Generative AI and FAISS to provide a robust solution for PDF content processing and question answering. Follow the installation and usage instructions to set up and run the application locally.
## Clone the repository:

```
git clone <repository-url>

cd <repository-directory>
```
## Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
## Install dependencies:
```
pip install -r requirements.txt
```
## Set up environment variables:
Create a .env file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```


## Run the application:

```
uvicorn main:app --reload
```
## Upload a PDF:

- Endpoint: /upload-pdf/
- Method: POST
- Form-Data:
  * Key: pdf_file
  * Value: Select a PDF file to upload
## Ask a Question:

- Endpoint: /chat/
- Method: POST
- JSON Body:
```
{
    "user_question": "Your question here"
}
```
