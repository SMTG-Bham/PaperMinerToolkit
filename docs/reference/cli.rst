Command-line reference
======================

The reference below is generated from the installed Click commands. Use the
task-oriented guides for complete workflows.

Corpus discovery and content
----------------------------

.. click:: paperminer.cli:paper_search
   :prog: ps_search
   :nested: full

.. click:: paperminer.cli:import_pdf_folder
   :prog: ps_import_pdfs
   :nested: full

.. click:: paperminer.cli:import_author
   :prog: ps_import_author
   :nested: full

.. click:: paperminer.cli:enrich
   :prog: ps_enrich
   :nested: full

.. click:: paperminer.cli:download
   :prog: ps_download
   :nested: full

.. click:: paperminer.cli:corpus_status
   :prog: ps_corpus_stats
   :nested: full

Filtering
---------

.. click:: paperminer.cli:filter_regex
   :prog: ps_filter_regex
   :nested: full

.. click:: paperminer.cli:filter_topic
   :prog: ps_filter_topic
   :nested: full

.. click:: paperminer.cli:filter_status
   :prog: ps_filter_status
   :nested: full

.. click:: paperminer.cli:filter_reset
   :prog: ps_filter_reset
   :nested: full

LDA topics
----------

.. click:: paperminer.cli:topics_train
   :prog: ps_topics_train
   :nested: full

.. click:: paperminer.cli:topics_compare
   :prog: ps_topics_compare
   :nested: full

.. click:: paperminer.cli:topics_show
   :prog: ps_topics_show
   :nested: full

.. click:: paperminer.cli:topics_name
   :prog: ps_topics_name
   :nested: full

.. click:: paperminer.cli:topics_predict
   :prog: ps_topics_predict
   :nested: full

.. click:: paperminer.cli:topics_trends
   :prog: ps_topics_trends
   :nested: full

.. click:: paperminer.cli:topics_store
   :prog: ps_topics_store
   :nested: full

.. click:: paperminer.cli:topics_models
   :prog: ps_topics_models
   :nested: full

Extraction and storage
----------------------

.. click:: paperminer.cli:scrape
   :prog: ps_scrape
   :nested: full

.. click:: paperminer.cli:store
   :prog: ps_store
   :nested: full

Model and credential configuration
----------------------------------

.. click:: paperminer.cli:model_config
   :prog: ps_model_config
   :nested: full

.. click:: paperminer.cli:model_status
   :prog: ps_model_status
   :nested: full

The credential commands are interactive prompts rather than Click option
parsers:

.. list-table::
   :header-rows: 1

   * - Command
     - Setting saved
   * - ``ps_elsevier_key``
     - Elsevier API key
   * - ``ps_core_key``
     - CORE API key
   * - ``ps_unpaywall_email``
     - Unpaywall contact email
   * - ``ps_crossref_email``
     - Crossref contact email
   * - ``ps_openalex_key``
     - OpenAlex API key
   * - ``ps_ncbi_key``
     - NCBI E-utilities API key
   * - ``ps_ncbi_email``
     - NCBI contact email
   * - ``ps_openai_key``
     - OpenAI API key
   * - ``ps_anthropic_key``
     - Anthropic API key

Each command validates or stores the entered value without accepting
positional arguments or options.

Pipeline maintenance
--------------------

.. click:: paperminer.cli:scraper_status
   :prog: ps_status
   :nested: full

.. click:: paperminer.cli:reset_scraper
   :prog: ps_reset
   :nested: full
