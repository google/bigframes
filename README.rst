BigQuery DataFrames (BigFrames)
===============================


|GA| |pypi| |versions|

BigQuery DataFrames (also known as BigFrames) provides a Pythonic DataFrame
and machine learning (ML) API powered by the BigQuery engine. It provides modules
for many use cases, including:

* `bigframes.pandas <https://dataframes.bigquery.dev/reference/api/bigframes.pandas.html>`_
  is a pandas API for analytics. Many workloads can be
  migrated from pandas to bigframes by just changing a few imports.
* `bigframes.bigquery.ai <https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ai.html>`_
  provides powerful AI functions, text generation, vector embeddings, and semantic search powered by Gemini.
* `bigframes.ml <https://dataframes.bigquery.dev/reference/index.html#ml-apis>`_
  is a scikit-learn-like API for ML.

BigQuery DataFrames is an `open-source package <https://github.com/google/bigframes>`_.

.. |GA| image:: https://img.shields.io/badge/support-GA-gold.svg
   :target: https://github.com/googleapis/google-cloud-python/blob/main/README.rst#general-availability
.. |pypi| image:: https://img.shields.io/pypi/v/bigframes.svg
   :target: https://pypi.org/project/bigframes/
.. |versions| image:: https://img.shields.io/pypi/pyversions/bigframes.svg
   :target: https://pypi.org/project/bigframes/

Why Use BigQuery DataFrames?
----------------------------

BigQuery DataFrames eliminates the "data movement bottleneck." Instead of downloading large datasets to a local environment, BigFrames translates your Python code into optimized SQL executed across BigQuery's distributed compute engine.

* **Petabyte-Scale Scalability:** Effortlessly process datasets exceeding local memory constraints without provisioning or managing Spark or Ray clusters.
* **Familiar Python Ecosystem:** Use the pandas operations you already know (``read_gbq``, ``groupby``, ``merge``, ``pivot_table``) directly on BigQuery tables.
* **Generative AI & Multimodality:** Seamlessly leverage Gemini models, AI functions, multimodal processing (audio, video, PDF), vector embeddings, and vector search with ``bigframes.bigquery.ai``.
* **Optimized Execution:** Take advantage of BigFrames partial ordering mode (``bpd.options.bigquery.ordering_mode = "partial"``) for considerable performance gains on analytical workloads.
* **Enterprise Security:** Keep your data securely governed within the BigQuery security perimeter.
* **Hybrid Flexibility:** Seamlessly transition between distributed BigQuery processing and local pandas analysis with ``to_pandas()`` and ``read_pandas()``.

Getting started with BigQuery DataFrames
----------------------------------------

Notebooks
~~~~~~~~~

The easiest way to get started is to open a sample notebook in
`Colab <https://colab.research.google.com/github/google/bigframes/blob/main/notebooks/getting_started/bq_dataframes_template.ipynb>`_
or
`BigQuery Studio <https://console.cloud.google.com/bigquery/import?url=https://github.com/google/bigframes/blob/main/notebooks/getting_started/bq_dataframes_template.ipynb>`_.
More details can be found in the `BigFrames quickstart <https://cloud.google.com/bigquery/docs/dataframes-quickstart>`_
and in the `introduction to notebooks <https://cloud.google.com/bigquery/docs/notebooks-introduction>`_.

Local environment
~~~~~~~~~~~~~~~~~

To use BigFrames in your local development environment:

1. Run ``pip install --upgrade bigframes`` to install the latest version.

2. Set up `Application Default Credentials <https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment>`_
   for your local development environment.

3. Create a `Google Cloud project with the BigQuery API enabled <https://cloud.google.com/bigquery/docs/sandbox>`_.

4. Use the ``bigframes`` package to query data.

.. code-block:: python

    import bigframes.pandas as bpd

    bpd.options.bigquery.project = your_gcp_project_id  # Optional in BQ Studio.
    bpd.options.bigquery.ordering_mode = "partial"  # Recommended for performance.
    df = bpd.read_gbq("bigquery-public-data.usa_names.usa_1910_2013")
    print(
        df.groupby("name")
        .agg({"number": "sum"})
        .sort_values("number", ascending=False)
        .head(10)
        .to_pandas()
    )

Documentation
-------------

To learn more about BigQuery DataFrames, visit these pages

* `Introduction to BigQuery DataFrames (BigFrames) <https://cloud.google.com/bigquery/docs/bigquery-dataframes-introduction>`_
* `Sample notebooks <https://github.com/google/bigframes/tree/main/notebooks>`_
* `API reference <https://dataframes.bigquery.dev/>`_
* `Source code (GitHub) <https://github.com/google/bigframes>`_

License
-------

BigQuery DataFrames is distributed with the `Apache-2.0 license
<https://github.com/google/bigframes/blob/main/LICENSE>`_.

It also contains code derived from the following third-party packages:

* `Ibis <https://ibis-project.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `Python <https://www.python.org/>`_
* `scikit-learn <https://scikit-learn.org/>`_
* `XGBoost <https://xgboost.readthedocs.io/en/stable/>`_
* `SQLGlot <https://sqlglot.com/sqlglot.html>`_

For details, see the `third_party
<https://github.com/google/bigframes/tree/main/third_party/bigframes_vendored>`_
directory.


Contact Us
----------

For further help and provide feedback, you can email us at `bigframes-feedback@google.com <https://mail.google.com/mail/?view=cm&fs=1&tf=1&to=bigframes-feedback@google.com>`_.
