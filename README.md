# College Q&A Chatbot

This project is a Q&A chatbot designed to answer questions related to the **Global Academy of Technology (GAT)** using a combination of large language models (LLMs), Text-Embeddings, Retrieval-Augmented Generation (RAG), and Prompt Engineering Techniques. The chatbot can process both text and audio inputs, providing relevant answers based on the conversation history and preloaded documents.

## Features

- **Text and Audio Input**: Accepts user queries via text input or voice recording.
- **Retrieval-Augmented Generation (RAG)**: Enhances responses using relevant information retrieved from preloaded documents.
- **Context-Aware Responses**: Utilizes conversation history to provide coherent and contextually appropriate answers.
- **Streamlit Interface**: User-friendly interface built with Streamlit, featuring options for follow-up mode and search depth customization.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/mahadev0811/CollegeChatbot.git
    cd CollegeChatbot
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure API Key**

    - Create a `config.json` file in the root directory with your Google API key:

    ```json
    {
      "google_api_key": "YOUR_GOOGLE_API_KEY"
    }
    ```

## Usage

1. **Run the Application**

    ```bash
    streamlit run st_app.py
    ```

2. **Interact with the Chatbot**

    - Use the text input box or the audio recorder to submit your queries.
    - Adjust settings using the sidebar options:
        - **Follow-up Questions Mode**: Toggle to use conversation history for responses.
        - **Search Depth**: Adjust the number of paragraphs to search in the documents for relevant information.

## Project Structure

- **st_app.py**: Main application script.
- **embedding_generator.py**: Script for generating embeddings from the data file.
- **webscrapper.ipynb**: Jupyter notebook for scraping text data from URLs to generate the raw data file.
- **config.json**: Configuration file for API keys.
- **requirements.txt**: List of required Python packages.
- **data_generation/gat_raw.txt**: Raw scraped data containing information about GAT.
- **data_generation/gat_refined.txt**: Human-supervised and edited version of the raw data.
- **gat_embeddings.pkl**: Precomputed embeddings for the preloaded document.

## Generating Data

To generate the initial raw data file (`gat_raw.txt`), use the `webscrapper.ipynb` notebook. This notebook scrapes text content from given URLs and formats it appropriately.

## Generating Embeddings

To generate embeddings from your data file, use the **embedding_generator.py** script. This script reads a text file containing the data, generates embeddings using the **FlagEmbedding model**, and saves the embeddings as a pickle file.

### Running the Embedding Generator

#### Prepare Your Data File
- Ensure your data file (e.g., data_generation/gat_refined.txt) is in the correct format, with paragraphs separated by double newlines (\n\n).

#### Run the Script

- Run the embedding_generator.py script with the path to your data file as an argument:

    ```bash
    python embedding_generator.py --data_file data_generation/gat_refined.txt
    ```

- The script will generate embeddings for the paragraphs in the data file and save them as a pickle file (`gat_embeddings.pkl`).

## Screenshots

- This video shows the chatbot in action, answering questions about GAT:

https://github.com/mahadev0811/CollegeChatbot/assets/124283478/2845e5c3-c243-4793-a40f-b2653c23233f

## Credits

- FlagEmbedding: Custom embedding model used for encoding queries.
- Streamlit: Open-source app framework for ML and data science projects.
- Hugging Face Transformers: Library for state-of-the-art NLP models.
- Google Cloud Speech-to-Text API: Service for converting speech into text.
- Google Generative AI:  Used for generating responses.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

