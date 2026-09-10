.. BigQuery DataFrames documentation main file

Scalable Python Data Analysis with BigQuery DataFrames (BigFrames)
==================================================================

.. meta::
   :description: BigQuery DataFrames (BigFrames) provides a scalable, pandas-compatible Python API for data analysis and machine learning on petabyte-scale datasets using the BigQuery engine.

**BigQuery DataFrames** (``bigframes``) is an open-source Python library that brings the power of **distributed computing** to your data science workflow. By providing a familiar **pandas** and **scikit-learn** compatible API, BigFrames allows you to analyze and model massive datasets where they live—directly in **BigQuery**.

Why Choose BigQuery DataFrames?
-------------------------------

BigFrames eliminates the "data movement bottleneck." Instead of downloading large datasets to a local environment, BigFrames translates your Python code into optimized SQL, executing complex transformations across the BigQuery fleet.

*   **Petabyte-Scale Scalability:** Effortlessly process datasets that far exceed local memory limits.
*   **Familiar Python Ecosystem:** Use the same ``read_gbq``, ``groupby``, ``merge``, and ``pivot_table`` functions you already know from pandas.
*   **Generative AI and Machine Learning:** Seamlessly leverage Gemini models, AI functions, and vector search with :mod:`bigframes.bigquery.ai`, alongside BigQuery ML's powerful algorithms via a scikit-learn-compatible interface (``bigframes.ml``).
*   **Enterprise-Grade Security:** Maintain data governance and security by keeping your data within the BigQuery perimeter.
*   **Hybrid Flexibility:** Easily move between distributed BigQuery processing and local pandas analysis with ``to_pandas()``.

Core Components of BigFrames
----------------------------

BigQuery DataFrames is organized into specialized modules designed for the modern data stack:

1.  :mod:`bigframes.pandas`: A high-performance, pandas-compatible API for scalable data exploration, cleaning, and transformation.
2.  :mod:`bigframes.bigquery`: Specialized utilities for direct BigQuery resource management, including integrations with Gemini and other AI models in the :mod:`bigframes.bigquery.ai` submodule.


Quickstart: Scalable Data Analysis in Seconds
---------------------------------------------

Install BigQuery DataFrames via pip:

.. code-block:: bash

    pip install --upgrade bigframes

The following example demonstrates how to perform a distributed aggregation on a public dataset with millions of rows using just a few lines of Python:

.. code-block:: python

    import bigframes.pandas as bpd

    # If running in your local environment or Colab, uncomment these lines and add your GCP project ID
    # PROJECT_ID = "bigframes-dev"
    # bpd.options.bigquery.project = PROJECT_ID

    # Initialize BigFrames and load a public dataset
    df = bpd.read_gbq("bigquery-public-data.usa_names.usa_1910_2013")

    # Perform familiar pandas operations that execute in the cloud
    top_names = (
        df.groupby("name")
        .agg({"number": "sum"})
        .sort_values("number", ascending=False)
        .head(10)
    )

    # Bring the final, aggregated results back to local memory if needed
    print(top_names.to_pandas())


Sample Notebooks and Interactive Demos
--------------------------------------

Explore sample notebooks demonstrating end-to-end workflows across analytics, GenAI, and machine learning. Each notebook can be launched directly in **BigQuery Studio** or **Consumer Colab**:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Notebook
     - BigQuery Studio
     - Consumer Colab
   * - **Getting started with BigFrames**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/getting_started/bq_dataframes_template.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/getting_started/bq_dataframes_template.ipynb>`__
   * - **AI functions (Gemini & GenAI)**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/generative_ai/ai_functions.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/generative_ai/ai_functions.ipynb>`__
   * - **Data visualization**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/visualization/tutorial.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/visualization/tutorial.ipynb>`__
   * - **Analyzing posters with AI functions**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/generative_ai/ai_movie_poster.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/generative_ai/ai_movie_poster.ipynb>`__
   * - **DataFrame operations**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/dataframes/dataframe.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/dataframes/dataframe.ipynb>`__
   * - **Multimodal DataFrames**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/multimodal/multimodal_dataframe.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/multimodal/multimodal_dataframe.ipynb>`__
   * - **SQL interoperability with bqsql magic**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/dataframes/magics_with_local_data.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/dataframes/magics_with_local_data.ipynb>`__
   * - **Timedelta operations**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/data_types/timedelta.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/data_types/timedelta.ipynb>`__
   * - **Remote Functions**
     - `Open in BQ Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/remote_functions/remote_function.ipynb>`__
     - `Run in Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/remote_functions/remote_function.ipynb>`__

Browse the `complete notebooks catalog on GitHub <https://github.com/google/bigframes/tree/main/notebooks>`_ for additional tutorials and community examples.


Articles and Guides
-------------------

To learn more about BigQuery DataFrames architecture, features, and best practices, check out the following articles and documentation guides:

*   `BigQuery DataFrames: Generally Available <https://cloud.google.com/blog/products/data-analytics/bigquery-dataframes-generally-available>`_: Official announcement detailing petabyte-scale pandas and scikit-learn on BigQuery.
*   `Generative AI in BigQuery with DataFrames <https://cloud.google.com/blog/products/data-analytics/generative-ai-in-bigquery-with-dataframes>`_: How to leverage Gemini models and AI functions directly in Python.
*   `Analyze Multimodal Data with BigQuery DataFrames <https://cloud.google.com/blog/products/data-analytics/analyze-multimodal-data-with-bigquery-dataframes>`_: End-to-end unstructured data analysis across images, audio, and PDF documents.
*   `Scalable Data Science with BigQuery DataFrames <https://cloud.google.com/blog/products/data-analytics/data-science-with-bigquery-dataframes>`_: Best practices for scaling analytics and machine learning workflows in the cloud.
*   `BigQuery DataFrames Introduction (Cloud Docs) <https://cloud.google.com/bigquery/docs/bigquery-dataframes-introduction>`_: Official documentation covering data manipulation, sessions, and deployment.
*   `Try BigQuery DataFrames (Quickstart Guide) <https://cloud.google.com/bigquery/docs/dataframes-quickstart>`_: Step-by-step tutorial in BigQuery Studio and Colab.


Explore the Documentation
-------------------------

.. toctree::
    :maxdepth: 2
    :caption: User Documentation

    user_guide/index

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    reference/index
    supported_pandas_apis

.. toctree::
    :maxdepth: 1
    :caption: Community & Updates

    CHANGELOG
