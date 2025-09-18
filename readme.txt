python generate_embeddingsSQL.py
python app.py
pip install pyodbc
 python -m pip show sentence-transformers

python generate_embeddings.py <--JSON
venv\Scripts\activate 
pip install sentence-transformers numpy torch scikit-learn


pip install Flask




CREATE TABLE DocumentEmbeddings (
    DocumentID NVARCHAR(255) PRIMARY KEY,
    DocumentText NVARCHAR(MAX),
    Embedding VARBINARY(MAX)
);

