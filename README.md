# AI Categorization Tool for Accounts Payable Departments

This tool uses AI to categorize financial expenses for an Accounts Payable department. It uses embeddings and a prompt for each individual categorization task, allowing it to mimic a human manual categorization process much closer than other methods.

## Dependencies

This tool requires the following Python libraries:

- pandas
- openai
- asyncio
- backoff
- chromadb

## How to use

1. Prepare your data: You need two CSV files. One for the financial categories and another for the items to categorize. Each file should contain one item per line.

2. Set your OpenAI API key: Replace "your_open_ai_key" with your actual OpenAI API key in the code.

3. Run the tool: Call the `process_input_files` function with the paths to your CSV files as arguments.

```python
if __name__ == "__main__":
    process_input_files('path_to_categories.csv', 'path_to_items_to_categorize.csv')
```

## How it works

The tool first builds an embedding database using the financial categories. It then reads the items to categorize and for each item, it queries the database to find the most similar categories. It then uses OpenAI's GPT-4 model to decide which category is the most accurate for each item.

The tool uses backoff to handle exceptions and retries the process up to 10 times in case of failure.

The results are saved in a CSV file with two columns: 'Input Item' and 'Matched Category'.

## Important Note

Current speed is relatively slow but the quality of output may be higher than other methods due to the addition of a GPT-4 prompt for each individual categorization. Current categories are just examples, replace them with your own. You can also add few-shot examples to the prompt in order to improve performance for your specific use case.
