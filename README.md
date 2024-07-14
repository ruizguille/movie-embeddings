# Vector Embeddings in Action: Visualizing IMDB's Top 250 Movies

Explore the semantic relationships between movies with this Python project. The code scrapes IMDB's Top 250 movies, transforms the plot summaries into vector embeddings, and creates 2D and 3D visualizations of the embeddings to explore the connections between them.

For a detailed explanation of the code and the concepts, check out [this blog post](https://codeawake.com/blog/movie-embeddings).

This project was developed by [CodeAwake](https://codeawake.com).

## Structure

The code is organized into the following files:

- `scrape_embed_movies.py`: Scrapes IMDB's Top 250 movies and creates embeddings for their plot summaries.
- `plot_movies.py`: Plots the movie embeddings in 2D and 3D.
- `requirements.txt`: Python dependencies for the project.
- `data/`: The data folder contains the output `movies.csv` and `movies_embeddings.csv` with the scraped movie data and the generated vector embeddings ready for plotting.

## Installation

1. Create and activate a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Install the dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. If you want to generate the vector embeddings, create a `.env` file copying the `.env.example` provided file and set the `OPENAI_API_KEY` environment variable with your own OpenAI API key. You can follow [this guide](https://platform.openai.com/docs/api-reference/introduction) to get started with the OpenAI API.
  
## Running the Application

If you want to scrape IMDB's Top 250 movies and create the vector embeddings:

```bash
python scrape_embed_movies.py
```

This will create two CSV files in the `data/` directory: `movies.csv` with the scraped data and `movies_embeddings.csv` which also includes the embeddings. But these files are already included in the repository, so you can plot the embeddings directly.

To plot the movie embeddings:

```bash
python plot_movies.py
```

By default, this will generate the 3D plot. To generate the 2D plot instead, call the function `plot_movies_2d()` instead of `plot_movies_3d()` in the main block.