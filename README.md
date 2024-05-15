# AI Categorization Tool with Embeddings

This tool uses AI to categorize financial expenses for an Accounts Payable department. It uses embeddings and a prompt for each individual categorization task, allowing it to mimic a human manual categorization process much closer than other methods.

##Important Note

This is a demo intended to showcase the potential of using embeddings for categorization. The current implementation is not optimized for speed and requires updates for scalability in production environments.

## Why This Method vs. Fine Tuning?

There are several advantages to using embeddings with a prompt vs. using a fine tuned model.

### Flexibility

Since you aren't fine-tuning on a specific dataset, the program can handle a wide vareity of input data. So if you fine-tuned a model to categorize expenses based on one type of input data it might be unable to categorize expenses for another type of input data. This method, on the other hand, is agnostic of input data type.

### Accuracy

Since we are bringing in a Chat Completion call into the categorization process, we're making the process closer to human-level thought. Instead of just suggesting a likely category based on the fine-tuning data, the LLM call resemebles more of the process of 'thinking through' the various options before deciding, similar to how a human would.

### Training Data Not Required

Unlike fine-tuning this method requires no training data. It only requires your list of target categories and the items you want to categorize. This can be an especially difficult process when you are dealing with thousands of categories, which can require millions of examples in your training dataset in order to properly train.

### Other Thoughts

There will be cases where this method is preferable and others where fine-tuning is a better option. I suggest this method will likely be preferable where accuracy or flexibility is key or if the training data for fine-tuning would be difficult to aquire. This method can also be enchanced by providing few-shot examples in the prompt tailored to your specific use-case.


## Dependencies

This tool requires the following Python libraries:

- pandas
- openai
- asyncio
- backoff
- chromadb

## How to use

1. Prepare your data: You need two CSV files. One for the financial categories and another for the items to categorize. Each file should contain one item per line (feel free to use the example files in this repo).

2. Set your OpenAI API key: Replace "your_open_ai_key" with your actual OpenAI API key in the code.

3. Run the tool: Call the `process_input_files` function with the paths to your CSV files as arguments by running the main.py file.

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
