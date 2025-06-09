Building a Retrieval Augmented Generation (RAG) Chatbot
Using Gemini, LangChain, and ChromaDB

This notebook will guide you through implementing the backend components of a RAG chatbot system.

Setup and Prerequisites
First, let's install the necessary libraries.


[ ]
# Install required packages
!pip install langchain langchain-google-genai langchain_community pypdf chromadb sentence-transformers -q
!pip install google-generativeai pdfplumber -q
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.3/67.3 kB 3.6 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.8/44.8 kB 3.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 65.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 304.2/304.2 kB 21.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.3/19.3 MB 95.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 94.9/94.9 kB 6.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 284.2/284.2 kB 20.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 15.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 70.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.6/101.6 kB 7.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 MB 99.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.8/65.8 kB 4.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.7/55.7 kB 4.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.2/196.2 kB 14.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 118.4/118.4 kB 8.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.9/101.9 kB 7.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.4/44.4 kB 2.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 363.4/363.4 MB 2.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.8/13.8 MB 102.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24.6/24.6 MB 82.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 883.7/883.7 kB 48.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 664.8/664.8 MB 2.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 211.5/211.5 MB 5.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.3/56.3 MB 18.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 127.9/127.9 MB 6.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.5/207.5 MB 6.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.1/21.1 MB 90.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 459.8/459.8 kB 32.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.9/50.9 kB 4.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.5/71.5 kB 5.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.0/4.0 MB 89.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 454.8/454.8 kB 29.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 3.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 7.1 MB/s eta 0:00:00
  Building wheel for pypika (pyproject.toml) ... done
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
google-generativeai 0.8.5 requires google-ai-generativelanguage==0.6.15, but you have google-ai-generativelanguage 0.6.18 which is incompatible.
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.8/42.8 kB 2.7 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.2/48.2 kB 3.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 43.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.2/60.2 kB 4.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.6/5.6 MB 105.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 81.2 MB/s eta 0:00:00
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
langchain-google-genai 2.1.5 requires google-ai-generativelanguage<0.7.0,>=0.6.18, but you have google-ai-generativelanguage 0.6.15 which is incompatible.
Next, let's import all required libraries:


[ ]
import os
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

[ ]
from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get("GEMINIAI")
Section 1: Uploading PDF
In this section, we'll implement the functionality to upload PDF files. For this notebook demonstration, we'll assume the PDF is in a local path.


[ ]
def upload_pdf(pdf_path):
    """
    Function to handle PDF uploads.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: PDF file path if successful
    """
    try:
        # In a real application with Streamlit, you would use:
        # uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        # But for this notebook, we'll just verify the file exists

        if os.path.exists(pdf_path):
            print(f"PDF file found at: {pdf_path}")
            return pdf_path
        else:
            print(f"Error: File not found at {pdf_path}")
            return None
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        return None

[ ]
attention_paper_path = "/content/attention_paper.pdf"

[ ]
upload_pdf(attention_paper_path)
Error: File not found at /content/attention_paper.pdf
Section 2: Parsing the PDF and Creating Text Files
Now we'll extract the text content from the uploaded PDFs.


[ ]
def parse_pdf(pdf_path):
    """
    Function to extract text from PDF files.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""

        # Using pdfplumber to extract text
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        # Save the extracted text to a file (optional)
        text_file_path = pdf_path.replace('.pdf', '.txt')
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"PDF parsed successfully, extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None

[ ]
text_file = parse_pdf(attention_paper_path)
Error parsing PDF: [Errno 2] No such file or directory: '/content/attention_paper.pdf'
Section 3: Creating Document Chunks
To effectively process and retrieve information, we need to break down our document into manageable chunks.


[ ]
def create_document_chunks(text):
    """
    Function to split the document text into smaller chunks for processing.

    Args:
        text (str): The full text from the PDF

    Returns:
        list: List of text chunks
    """
    try:
        # Initialize the text splitter
        # We can tune these parameters based on our needs and model constraints
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # Size of each chunk in characters
            chunk_overlap=100,      # Overlap between chunks to maintain context
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Hierarchy of separators to use when splitting
        )

        # Split the text into chunks
        chunks = text_splitter.split_text(text)

        print(f"Document split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error creating document chunks: {e}")
        return []

[ ]
text_chunks = create_document_chunks(text_file)
Error creating document chunks: expected string or bytes-like object, got 'NoneType'
Section 4: Embedding the Documents
Now we'll create vector embeddings for each text chunk using Gemini's embedding model.


[ ]
def embed_and_view(text_chunks):
    """
    Embed document chunks and display their numeric embeddings.

    Args:
        text_chunks (list): List of text chunks from the document
    """
    try:
        # Initialize the Gemini embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"  # Specify the Gemini Embedding model
        )

        print("Embedding model initialized successfully")

        # Generate and display embeddings for all chunks
        for i, chunk in enumerate(text_chunks):
            embedding = embedding_model.embed_query(chunk)
            print(f"Chunk {i} Embedding:\n{embedding}\n")

    except Exception as e:
        print(f"Error embedding documents: {e}")

# Example usage
sample_chunks = ["This is the first chunk.", "This is the second chunk.", "And this is the third chunk."]
embed_and_view(sample_chunks)
Embedding model initialized successfully
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab255b68e90>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab255b68e90>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254ebf5d0>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254ebf5d0>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e82050>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e82050>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab2551919d0>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab2551919d0>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e71fd0>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e71fd0>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254df7a10>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254df7a10>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e83890>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e83890>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254df7790>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254df7790>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e72610>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e72610>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e5b190>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e5b190>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e58310>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e58310>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e070d0>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e070d0>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e83bd0>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e83bd0>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e83dd0>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e83dd0>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e311d0>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e311d0>)
ERROR:grpc._plugin_wrapping:AuthMetadataPluginCallback "<google.auth.transport.grpc.AuthMetadataPlugin object at 0x7ab254fad090>" raised exception!
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 126, in refresh
    self._retrieve_info(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 99, in _retrieve_info
    info = _metadata.get_service_account_info(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 338, in get_service_account_info
    return get(request, path, params={"recursive": "true"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/_metadata.py", line 263, in get
    raise exceptions.TransportError(
google.auth.exceptions.TransportError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e47090>)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/grpc/_plugin_wrapping.py", line 105, in __call__
    self._metadata_plugin(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 95, in __call__
    callback(self._get_authorization_headers(context), None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/google/auth/transport/grpc.py", line 81, in _get_authorization_headers
    self._credentials.before_request(
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 239, in before_request
    self._blocking_refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/credentials.py", line 202, in _blocking_refresh
    self.refresh(request)
  File "/usr/local/lib/python3.11/dist-packages/google/auth/compute_engine/credentials.py", line 132, in refresh
    raise new_exc from caught_exc
google.auth.exceptions.RefreshError: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e47090>)
Error embedding documents: Error embedding content: Timeout of 60.0s exceeded, last exception: 503 Getting metadata from plugin failed with error: ("Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true from the Google Compute Engine metadata service. Status: 404 Response:\nb''", <google.auth.transport.requests._Response object at 0x7ab254e47090>)

[ ]
def embed_documents(text_chunks):
    """
    Function to generate embeddings for the text chunks.

    Args:
        text_chunks (list): List of text chunks from the document

    Returns:
        object: Embedding model for further use
    """
    try:
        # Initialize the Gemini embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"  # Specify the Gemini Embedding model
        )

        print("Embedding model initialized successfully")
        return embedding_model, text_chunks
    except Exception as e:
        print(f"Error embedding documents: {e}")
        return None, None

[ ]
embedded_documents = embed_documents(text_chunks)
Embedding model initialized successfully
Section 5: Storing in Vector Database (ChromaDB)
In this section, we'll store the embedded document chunks in a vector database for efficient semantic search.


[ ]
def store_embeddings(embedding_model, text_chunks):
    """
    Function to store document embeddings in ChromaDB.

    Args:
        embedding_model: The embedding model to use
        text_chunks (list): List of text chunks to embed and store

    Returns:
        object: Vector store for retrieval
    """
    try:
        # Create a vector store from the documents
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"  # Directory to persist the database
        )

        # Persist the vector store to disk
        vectorstore.persist()

        print(f"Successfully stored {len(text_chunks)} document chunks in ChromaDB")
        return vectorstore
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None

[ ]
chroma_store = store_embeddings(embedded_documents[0],embedded_documents[1])
Error storing embeddings: Expected Embeddings to be non-empty list or numpy array, got [] in upsert.
Section 6: Embedding User Queries
When a user submits a query, we need to embed it using the same embedding model to find semantically similar chunks.


[ ]
def embed_query(query, embedding_model):
    """
    Function to embed the user's query.

    Args:
        query (str): User's question
        embedding_model: The embedding model to use

    Returns:
        list: Embedded query vector
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(query)

        print("Query embedded successfully")
        return query_embedding
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

[ ]
user_query = "Who are the authors of the Attention paper?"

[ ]
embedded_query = embed_query(user_query, embedded_documents[0])
print(embedded_query)
Query embedded successfully
[0.027056798338890076, 0.013028842397034168, -0.03088494762778282, 0.01905638352036476, -0.021859562024474144, 0.0568273700773716, 0.025342877954244614, 0.07066012918949127, 0.030687717720866203, -0.010601082816720009, -0.02538389153778553, 0.04057272896170616, 0.015543883666396141, -0.017788197845220566, -0.009052356705069542, -0.03872228413820267, 0.04835465922951698, 0.010206279344856739, 0.008470979519188404, 0.042000047862529755, -0.01562256459146738, -0.003287555417045951, -0.001992535777390003, -0.02069873735308647, 0.017305027693510056, -0.05144113674759865, 0.024168048053979874, -0.05869678035378456, -0.018119435757398605, -0.08787237107753754, -0.05156655237078667, 0.009036662988364697, -0.00023783252981957048, 0.018655333667993546, 0.0076125930063426495, 0.07520867139101028, -0.024185165762901306, 0.05185379832983017, 0.04049595445394516, -0.0520133376121521, -0.030061930418014526, -0.06000596284866333, 0.007626835256814957, 0.0522470586001873, -0.05023351311683655, 0.01707395166158676, 0.06736037880182266, 0.06700339168310165, -0.012275859713554382, 0.06897923350334167, -0.027558790519833565, -0.017224766314029694, -0.06208447366952896, 0.007423860486596823, -0.052715495228767395, -0.03518657013773918, 0.004004145506769419, 0.014924619346857071, 0.08062388002872467, -0.010238182730972767, -0.037857141345739365, -0.012602722272276878, -0.011011693626642227, 0.0021808387245982885, 0.016280638054013252, -0.050913985818624496, -0.003099051769822836, 0.03796398639678955, -0.05163642019033432, 0.03208162263035774, 0.010089470073580742, -0.021980086341500282, -0.009681150317192078, 0.04341153800487518, 0.02646706812083721, -0.03272674232721329, -0.02587565965950489, -0.008515512570738792, 0.028500497341156006, 0.056854601949453354, -0.011556314304471016, 0.06665099412202835, 0.09386670589447021, 0.0062550827860832214, 0.03788004815578461, -0.00704280287027359, 0.04517560824751854, -0.07224764674901962, -0.03305954858660698, -0.03917593136429787, 0.10215091705322266, -0.004317968152463436, -0.019895751029253006, -0.0003577613679226488, 0.0649157464504242, -0.06720425188541412, -0.016738971695303917, 0.020158978179097176, 0.11948147416114807, 0.03890072554349899, -0.027657130733132362, 0.005508420988917351, -0.021039223298430443, -0.04685721918940544, 0.04603811353445053, 0.027200382202863693, 0.03587302565574646, -0.03505025431513786, 0.0005727695534005761, -0.0028554860036820173, -0.012292537838220596, -0.03389742970466614, -0.04845762252807617, 0.024275073781609535, 0.03192492201924324, -0.02811160683631897, -0.06611601263284683, -0.017772415652871132, 0.005005171988159418, -0.035390812903642654, -0.028677502647042274, 0.03273867070674896, -0.017406577244400978, 0.0132669135928154, 0.02174987457692623, -0.032861240208148956, 0.02584965154528618, -0.04615968465805054, -0.0696442574262619, 0.04740836098790169, 0.054417483508586884, -0.04919328913092613, -0.043972309678792953, -0.03778444975614548, -0.006465267855674028, -0.04817504435777664, 0.04015014320611954, -0.002916093450039625, 0.03929739072918892, 0.040631260722875595, 0.03742046654224396, -0.0312221497297287, -0.0006980002508498728, -0.025316620245575905, 0.06505259871482849, -0.002680195029824972, 0.009827302768826485, 0.054740726947784424, 0.06490886956453323, 0.021396901458501816, -0.030771199613809586, 0.0247588399797678, -0.01227739080786705, 0.007821994833648205, -0.06724900007247925, -0.04863180220127106, 0.04487880319356918, -0.021745147183537483, 0.0762147381901741, -0.030360523611307144, 0.0027198295574635267, -0.009714937768876553, -0.02109050750732422, -0.015488684177398682, -0.01910707913339138, 0.008586128242313862, -0.01096358522772789, -0.014213073998689651, 0.0020438863430172205, 0.03666289523243904, -0.013043932616710663, -0.022581402212381363, 0.03280806913971901, -0.013755262829363346, 0.042210958898067474, 0.006791478488594294, -0.011389672756195068, 0.011339187622070312, 0.018695088103413582, -0.015202879905700684, 0.029560532420873642, 0.0007130727753974497, -0.04476958513259888, -0.0015566516667604446, -0.04095449671149254, -0.04641121253371239, 0.05526680126786232, -0.022218072786927223, 0.04961418733000755, -0.006577093154191971, 0.007852059789001942, 0.0460611954331398, 0.0629272386431694, 0.010615117847919464, -0.0004968802677467465, -0.061384834349155426, -0.06083855405449867, -0.02920924872159958, 0.04676654934883118, -0.010863893665373325, -0.0009162421338260174, 0.03621770069003105, -0.004412543494254351, -0.036075178533792496, -0.0282826516777277, -0.007955791428685188, -0.04946114122867584, -0.0530550479888916, 0.0743950828909874, 0.024487657472491264, -0.025849025696516037, -0.05212590470910072, 0.023163193836808205, -0.06668369472026825, 0.02228635922074318, -0.02288162149488926, 0.08526717871427536, 0.04825305566191673, 0.05341605842113495, -0.023463640362024307, -0.031375687569379807, -0.005977738182991743, -0.026654217392206192, -0.03466835245490074, 0.01499415934085846, 0.056748539209365845, -0.029048284515738487, -0.013475487940013409, -0.017591755837202072, -0.017878970131278038, -0.0003084926283918321, -0.053554557263851166, -0.013328657485544682, 0.028934208676218987, -0.006757066585123539, 0.0065501718781888485, 0.005693657323718071, 0.012492068111896515, -0.03562813997268677, -0.0005738681647926569, -0.0024373293854296207, 0.09911144524812698, -0.0020863416139036417, -0.05118155479431152, -0.04048857465386391, 0.06842041015625, 0.037642013281583786, -0.01081090234220028, -0.043456144630908966, -0.04128151386976242, -0.014363166876137257, 0.01575690507888794, -0.04402603581547737, 0.03317530080676079, -0.09030172228813171, -0.04900049790740013, -0.021862637251615524, -0.045858535915613174, -0.045021459460258484, -0.017863962799310684, -0.024072173982858658, -0.040067583322525024, -0.05983398109674454, -0.07798957079648972, -0.04486028850078583, -0.04588911309838295, 0.020949961617588997, -0.037785835564136505, 0.0688636377453804, 0.03845107927918434, -0.012593179941177368, 0.010664977133274078, -0.04338192194700241, 0.0038524912670254707, -0.06180635467171669, 0.01958296447992325, 0.019146695733070374, -0.009711100719869137, -0.025420593097805977, 0.032708704471588135, 0.10061681270599365, 0.05855494737625122, -0.060735005885362625, -0.04393663629889488, 0.039425287395715714, 0.002424242440611124, 0.013974928297102451, -0.019085880368947983, -0.04633297026157379, -0.03974442929029465, 0.056474871933460236, 0.04494019225239754, -0.03988734260201454, -0.024350587278604507, -0.007551808841526508, 0.0253042820841074, -0.009891289286315441, 0.03750637173652649, 0.004739376250654459, 0.03862914443016052, -0.013125010766088963, 0.05611294135451317, -0.03247331455349922, 0.019798096269369125, 0.0045926193706691265, -0.05569339543581009, -0.00016662015696056187, -0.003817003220319748, -0.08318678289651871, -0.01307790819555521, 0.057084813714027405, -0.006819719914346933, 0.01888830028474331, -0.015018953010439873, -0.025668537244200706, -0.040670424699783325, -0.07710039615631104, 0.016585834324359894, 0.01034157071262598, 0.001207715948112309, -0.025897204875946045, -0.020472247153520584, -0.049018047749996185, 0.00019006516959052533, 0.04428023472428322, 0.010722305625677109, -0.001215027761645615, 0.025884604081511497, -0.01817246526479721, 0.05531671643257141, 0.05204758420586586, -0.034707408398389816, 0.0027579881716519594, -0.07292283326387405, -0.0581023246049881, 9.447227785130963e-05, 0.01376680564135313, 0.060303401201963425, 0.06361903995275497, 0.03333170339465141, 0.06340261548757553, 0.012230444699525833, 0.07563317567110062, -0.00028118304908275604, 0.014682811684906483, -0.036760516464710236, -0.07328906655311584, -0.007551171351224184, -0.005059943068772554, -0.008726079016923904, 0.023438069969415665, 0.06385339796543121, -0.012908266857266426, -0.05374028533697128, -0.007753116078674793, -0.01239883340895176, 0.07219171524047852, -0.01807067170739174, -0.01640145853161812, -0.02459082007408142, -0.005521808285266161, 0.03379567340016365, 0.026738939806818962, 0.023219235241413116, -0.015280107967555523, -0.012389139272272587, 0.03699155151844025, 0.07451077550649643, -0.009572777897119522, -0.03679874539375305, 0.06100859493017197, 0.08037964999675751, 0.01363856066018343, 0.005829701665788889, -0.009693888016045094, 0.005892232060432434, 0.017951223999261856, 0.07903250306844711, -0.023400625213980675, -0.03494594246149063, -0.008991961367428303, -0.04180792346596718, 0.002466596895828843, 0.03334503248333931, 0.01388531643897295, 0.040323950350284576, -0.05775150656700134, 0.017544956877827644, -0.008461043238639832, 0.0008852377068251371, 0.013134758919477463, 0.04135144501924515, -0.029893245548009872, 0.06233776733279228, 0.009875018149614334, -0.003757375292479992, -0.005296635907143354, 0.03805331140756607, 0.006471718195825815, -0.006293427664786577, 0.05084584653377533, -0.014536016620695591, 0.059546440839767456, -0.007059171330183744, -0.010050760582089424, 0.051188793033361435, 0.014697556383907795, 0.009118244983255863, 0.025695523247122765, -0.007497760001569986, -0.035661716014146805, 0.011181454174220562, -0.015584150329232216, -0.04494433104991913, -0.04340972378849983, 0.0016681802226230502, 0.019349584355950356, -0.009341854602098465, 0.019635656848549843, 0.007011977955698967, 0.02435382641851902, 0.04334881156682968, 0.05935797840356827, -0.013676738366484642, -0.01039219368249178, -0.06063016131520271, 0.047385502606630325, -0.04577513784170151, 0.016790200024843216, 0.0285476241260767, -0.0024787713773548603, 0.05183528736233711, -0.01882815733551979, -0.017441948875784874, 0.031243374571204185, 0.0310496985912323, 0.003197680227458477, -0.02010757476091385, 0.012403905391693115, 0.0232075285166502, 0.020932013168931007, -0.019381342455744743, 0.06042161583900452, 0.05589019134640694, 0.004431882873177528, 0.002789766527712345, 0.015233821235597134, -0.025232963263988495, 0.027861909940838814, -0.08492778241634369, 0.01730511337518692, -0.040370069444179535, -0.02716822922229767, -0.03332364559173584, -0.05655458942055702, -0.04227067902684212, 0.03235946223139763, 0.006281700916588306, -0.06668910384178162, -0.0027970345690846443, 0.005055297631770372, -0.005018434952944517, 0.014309565536677837, 0.031137662008404732, 0.0374414324760437, -0.03491346165537834, 0.04264402016997337, -0.03469197079539299, -0.01687474362552166, 0.004312417004257441, 0.04885876178741455, 0.009582786820828915, 0.01871534250676632, 0.010223152115941048, -0.09601184725761414, -0.028421776369214058, 0.026789596304297447, 0.048824701458215714, 0.012751777656376362, -0.06224643811583519, 0.04299085959792137, 0.0041052973829209805, -0.04865912348031998, -0.010809146799147129, 0.007023450452834368, -0.01252589002251625, -0.10111475735902786, 0.003340316703543067, 0.08818694949150085, -0.044184986501932144, -0.031091559678316116, -0.00022302987053990364, -0.007264542393386364, -0.0028041531331837177, -0.00228125206194818, 0.030427273362874985, 0.02193382941186428, -0.030587488785386086, 0.007158436346799135, 0.04210421442985535, 0.026713192462921143, -0.009920971468091011, 0.03005985915660858, -0.05731340870261192, -0.028110457584261894, 0.023549335077404976, -0.024589238688349724, 0.0019656564109027386, -0.02085157111287117, 0.018475497141480446, 0.06273774802684784, 0.0009622873621992767, -0.005094081163406372, 0.0013053631410002708, 0.013419306837022305, 0.015437648631632328, -0.019063489511609077, 0.02072647400200367, -0.03488403186202049, 0.0043463618494570255, -0.0016328641213476658, 0.030608540400862694, 0.0781119093298912, 0.014164404012262821, 0.04300729185342789, -0.03784314543008804, -0.041223954409360886, -0.04710252210497856, 0.03652598336338997, -0.018503079190850258, 0.017634455114603043, 0.09182088077068329, -0.0049885534681379795, -0.01512760017067194, 0.012348098680377007, 0.031864065676927567, 0.0002546844771131873, 0.031814370304346085, -0.0025790988001972437, 0.018832923844456673, -0.00027975751436315477, 0.025461558252573013, 0.03970201686024666, 0.040224913507699966, 0.05007651075720787, -0.035234879702329636, -0.038176462054252625, 0.035227175801992416, 0.014646796509623528, 0.06502144038677216, 0.002913439879193902, -0.07018497586250305, -0.007290653884410858, -0.01576479710638523, -0.01730455458164215, -0.019880007952451706, 0.06606663763523102, 0.008643570356070995, -0.017811136320233345, -0.01458506379276514, -0.04700269177556038, -0.07843448966741562, 0.026759760454297066, 0.0269780233502388, -0.023422999307513237, 0.03385433927178383, -0.012957431375980377, -0.0012952168472111225, -0.0024927726481109858, -0.038485389202833176, 0.05372394621372223, -0.0066300807520747185, 0.007462918758392334, -0.005251738708466291, -0.039149701595306396, -0.021178383380174637, -0.02066340111196041, 0.04866326227784157, 0.008303558453917503, -0.03702864050865173, 4.409779648995027e-05, -0.05354355648159981, 0.03137488290667534, 0.036677464842796326, -0.032132312655448914, 0.01837744750082493, 0.00043261790415272117, 0.05256858468055725, 0.0259469673037529, -0.013993004336953163, -0.009776536375284195, 0.031881485134363174, 0.04857863485813141, -0.014436556957662106, 0.0033893180079758167, -0.014729500748217106, 0.017473379150032997, -0.044643599539995193, -0.002748092170804739, -0.008461983874440193, 0.034596290439367294, 0.006917495746165514, -0.02576654963195324, 0.01935059204697609, 0.03181585296988487, -0.008259042166173458, -0.038375090807676315, -0.030253898352384567, -0.037422869354486465, -0.03495887666940689, -0.02541322074830532, 0.07275908440351486, 0.036188531666994095, 0.013098877854645252, -0.05706150457262993, -0.0452863946557045, 0.029498722404241562, -0.06339889019727707, 0.003367767436429858, -0.014478571712970734, 0.0007643370190635324, 0.04627791419625282, -0.015285227447748184, 0.03654450923204422, 0.021023225039243698, 0.11642158031463623, -0.00732293538749218, 0.002697851974517107, 0.01742793619632721, -0.006751567125320435, -0.009737885557115078, 0.0046380688436329365, -0.004227094352245331, -0.040604133158922195, 0.03555960953235626, 0.00805797427892685, 0.04709852114319801, -0.026446988806128502, -0.02701067179441452, -0.008691255003213882, 0.05967441573739052, 0.014780007302761078, 0.03801611065864563, 0.05297614634037018, -0.026406390592455864, 0.0398748442530632, -0.018069583922624588, -0.025769809260964394, 0.014196556992828846, -0.01008087582886219, 0.028896424919366837, 0.043265920132398605, -0.04217049479484558, 0.017482029274106026, 0.038229573518037796, -0.02371508628129959, -0.040305666625499725, -0.0010339365107938647, -0.0015947693027555943, -0.028146682307124138, 0.0472334660589695, -0.05242297425866127, 0.01241529081016779, -0.014296655543148518, 0.04151812568306923, -0.06612320244312286, -0.03755486384034157, 0.049204032868146896, 0.07133234292268753, -0.013996964320540428, -0.010526553727686405, -0.011237273924052715, 0.014070918783545494, 0.05819070339202881, 0.0029924612026661634, 0.0031778637785464525, 0.015303161926567554, -0.010955316945910454, 0.014810062944889069, -0.011141634546220303, -0.0007889915141277015, -0.06472942978143692, 0.04977940768003464, 0.055740516632795334, 0.02685951068997383, -0.027941318228840828, 0.06893368065357208, 0.03904823586344719, 0.03463110327720642, 0.03656685724854469, -0.05368680879473686, -0.028715016320347786, -0.03470007702708244, 0.014101661741733551, 0.001422745524905622, 0.011345193721354008, 0.015566479414701462, 0.023012978956103325, -0.007521944120526314, -0.01387458574026823, 0.013431807048618793, -0.014018741436302662, -0.055712345987558365, 0.029201390221714973, 0.0018155323341488838, -0.03211665153503418, 0.03886754810810089, -0.011953870765864849, -0.041249535977840424, 0.02157711610198021, -0.02286698669195175, 0.017701569944620132, 0.05187929421663284, -0.034149300307035446, -0.008053621277213097, 0.010202358476817608, 0.024125445634126663, -0.007057040464133024, 0.008152717724442482, -0.031375087797641754, -0.02012910507619381, -0.028377389535307884, 0.025751013308763504, 0.015014395117759705, 0.004309793468564749, -0.016434337943792343, -0.04018353298306465, -0.0227042268961668, -0.020538751035928726, 0.030581125989556313, 0.07253724336624146, 0.021406713873147964, 0.01181443128734827, 0.05042819306254387, -0.03253919631242752, -0.006194444373250008, -0.0069805411621928215, -0.03411298990249634, -0.014659912325441837, 0.010722828097641468, -0.023517031222581863, 0.011619973927736282, 0.024254057556390762, 0.012862409465014935, -0.007729734294116497, 0.025650659576058388, -0.01301048044115305, -0.01953227072954178, -0.020458778366446495, -0.02204384282231331, -0.05271754041314125, 0.019964251667261124, 0.02931988425552845, -0.03906792402267456, 0.006437600124627352, -0.011958910152316093, 0.021044518798589706, -0.020620161667466164, -0.0038918338250368834, 0.03876076638698578, 0.03424949198961258, 0.03696994110941887, -0.0013987816637381911, 0.047700345516204834, 0.04027770087122917, 0.00487911980599165, 0.01690875180065632, -0.006427663378417492]
Section 7: Retrieval Process
Now we'll implement the retrieval component that finds the most relevant document chunks based on the user's query.


[ ]
def retrieve_relevant_chunks(vectorstore, query, embedding_model, k=3):
    """
    Function to retrieve the most relevant document chunks for a query.

    Args:
        vectorstore: The ChromaDB vector store
        query (str): User's question
        embedding_model: The embedding model
        k (int): Number of chunks to retrieve

    Returns:
        list: List of relevant document chunks
    """
    try:
        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # Can also use "mmr" for Maximum Marginal Relevance
            search_kwargs={"k": k}     # Number of documents to retrieve
        )

        # Retrieve relevant chunks
        relevant_chunks = retriever.get_relevant_documents(query)

        print(f"Retrieved {len(relevant_chunks)} relevant document chunks")
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []

[ ]
relevant_chunks = retrieve_relevant_chunks(chroma_store, user_query, embedded_documents[0])
Error retrieving chunks: 'NoneType' object has no attribute 'as_retriever'

[ ]
relevant_chunks
[]

[ ]
def get_context_from_chunks(relevant_chunks, splitter="\n\n---\n\n"):
    """
    Extract page_content from document chunks and join them with a splitter.

    Args:
        relevant_chunks (list): List of document chunks from retriever
        splitter (str): String to use as separator between chunk contents

    Returns:
        str: Combined context from all chunks
    """
    # Extract page_content from each chunk
    chunk_contents = []

    for i, chunk in enumerate(relevant_chunks):
        if hasattr(chunk, 'page_content'):
            # Add a chunk identifier to help with tracing which chunk provided what information
            chunk_text = f"[Chunk {i+1}]: {chunk.page_content}"
            chunk_contents.append(chunk_text)

    # Join all contents with the splitter
    combined_context = splitter.join(chunk_contents)

    return combined_context

[ ]
context = get_context_from_chunks(relevant_chunks)

[ ]
context


[ ]
 final_prompt = f"""You are a helpful assistant answering questions based on provided context.

The context is taken from academic papers, and might have formatting issues like spaces missing between words.
Please interpret the content intelligently, separating words properly when they appear joined together.

Use ONLY the following context to answer the question.
If the answer cannot be determined from the context, respond with "I cannot answer this based on the provided context."

Context:
{context}

Question: {user_query}

Answer:"""

[ ]
final_prompt


[ ]
def generate_response(prompt, model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3, top_p=0.95):
    """
    Function to generate a response using the Gemini model.

    Args:
        prompt (str): The prompt for the model

    Returns:
        str: Model's response
    """

