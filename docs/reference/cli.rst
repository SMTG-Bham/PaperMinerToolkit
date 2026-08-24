Command-line reference
======================

The reference below is generated from the installed Click commands. Use the
task-oriented guides for complete workflows.

Corpus discovery and content
----------------------------

.. click:: paperminer.cli:paper_search
   :prog: pm_search
   :nested: full

.. click:: paperminer.cli:import_pdf_folder
   :prog: pm_import_pdfs
   :nested: full

.. click:: paperminer.cli:import_author
   :prog: pm_import_author
   :nested: full

.. click:: paperminer.cli:enrich
   :prog: pm_enrich
   :nested: full

.. click:: paperminer.cli:download
   :prog: pm_download
   :nested: full

.. click:: paperminer.cli:corpus_status
   :prog: pm_corpus_stats
   :nested: full

Filtering
---------

.. click:: paperminer.cli:filter_regex
   :prog: pm_filter_regex
   :nested: full

.. click:: paperminer.cli:filter_topic
   :prog: pm_filter_topic
   :nested: full

.. click:: paperminer.cli:filter_status
   :prog: pm_filter_status
   :nested: full

.. click:: paperminer.cli:filter_reset
   :prog: pm_filter_reset
   :nested: full

LDA topics
----------

.. click:: paperminer.cli:topics_train
   :prog: pm_topics_train
   :nested: full

.. click:: paperminer.cli:topics_compare
   :prog: pm_topics_compare
   :nested: full

.. click:: paperminer.cli:topics_show
   :prog: pm_topics_show
   :nested: full

.. click:: paperminer.cli:topics_name
   :prog: pm_topics_name
   :nested: full

.. click:: paperminer.cli:topics_predict
   :prog: pm_topics_predict
   :nested: full

.. click:: paperminer.cli:topics_trends
   :prog: pm_topics_trends
   :nested: full

.. click:: paperminer.cli:topics_store
   :prog: pm_topics_store
   :nested: full

.. click:: paperminer.cli:topics_models
   :prog: pm_topics_models
   :nested: full

Extraction and storage
----------------------

.. click:: paperminer.cli:scrape
   :prog: pm_scrape
   :nested: full

.. click:: paperminer.cli:store
   :prog: pm_store
   :nested: full

Model and credential configuration
----------------------------------

.. click:: paperminer.cli:model_config
   :prog: pm_model_config
   :nested: full

.. click:: paperminer.cli:model_status
   :prog: pm_model_status
   :nested: full

The credential commands are interactive prompts rather than Click option
parsers:

.. list-table::
   :header-rows: 1

   * - Command
     - Setting saved
   * - ``pm_elsevier_key``
     - Elsevier API key
   * - ``pm_core_key``
     - CORE API key
   * - ``pm_unpaywall_email``
     - Unpaywall contact email
   * - ``pm_crossref_email``
     - Crossref contact email
   * - ``pm_openalex_key``
     - OpenAlex API key
   * - ``pm_ncbi_key``
     - NCBI E-utilities API key
   * - ``pm_ncbi_email``
     - NCBI contact email
   * - ``pm_openai_key``
     - OpenAI API key
   * - ``pm_anthropic_key``
     - Anthropic API key

Each command validates or stores the entered value without accepting
positional arguments or options.

Pipeline maintenance
--------------------

.. click:: paperminer.cli:miner_status
   :prog: pm_status
   :nested: full

.. click:: paperminer.cli:reset_miner
   :prog: pm_reset
   :nested: full
