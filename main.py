import pandas as pd
import openai as OpenAI
import asyncio
import backoff
from chromadb.api.types import Documents, Images, Embeddings
from typing import Union, TypeVar
import chromadb.utils.embedding_functions as ef
import datetime
import chromadb

Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable)
MODEL = "text-embedding-3-large"

# OpenAI client initialization
openai_client = OpenAI.Client(api_key="your_open_ai_key")

class EmbeddingHandler:
    def __call__(self, input_data: D) -> Embeddings:
        embeds = []
        batch_size = 200
        for i in range(0, len(input_data), batch_size):
            lines_batch = input_data[i: i + batch_size]
            assert all(isinstance(line, str) for line in lines_batch), f"Invalid input: {lines_batch}"
            res = openai_client.embeddings.create(input=lines_batch, model=MODEL)
            batch_embeds = [record['embedding'] for record in res.data]
            embeds.extend(batch_embeds)
        return embeds

def build_chroma_db(docs):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    db_name = f"mydb_{current_time}"
    chroma_client = chromadb.Client()
    openai_ef = ef.OpenAIEmbeddingFunction(
        api_key="your_open_ai_key",  # Use your actual API key
        model_name="text-embedding-3-large"
    )
    db = chroma_client.create_collection(name=db_name, embedding_function=openai_ef)
    for i, d in enumerate(docs):
        db.add(documents=d, ids=str(i))
    return db

@backoff.on_exception(backoff.expo, Exception, max_tries=10)
async def async_query_processor(queries, db):
    results = []
    for query in queries:
        passages = db.query(query_texts=[query], n_results=5)['documents'][0]
        prompt = f"Here is the expense I need to categorize '{query}', and here are the potential categories: \n\n"
        for i, passage in enumerate(passages):
            prompt += f"{passage}\n"
        prompt += "\nWhich is the most accurate category for our accounting department to select for this expense? Output only the category name (including subfolders if they were provided) and nothing else."
        completion = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "This is an Accounts Payable Tool to Categorize Expenses"},
                {"role": "user", "content": prompt}
            ],
            stop=["#"],
            top_p=1,
            temperature=0.1,
            max_tokens=224
        )
        output = completion.choices[0].message.content
        results.append(output)
    return results

def process_input_files(financial_categories, items_to_categorize_csv):
    df_docs = pd.read_csv(financial_categories, header=None)
    docs = df_docs[0].tolist()
    db = build_chroma_db(docs)
    df_queries = pd.read_csv(items_to_categorize_csv, header=None)
    queries = df_queries[0].tolist()
    results = asyncio.run(async_query_processor(queries, db))
    df_results = pd.DataFrame({
        'Input Item': queries,
        'Matched Category': results
    })
    df_results.to_csv('path_to_output.csv', index=False)

if __name__ == "__main__":
    process_input_files('path_to_categories.csv', 'path_to_items_to_categorize.csv')
