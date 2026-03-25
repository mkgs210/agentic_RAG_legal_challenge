import json
import click
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs
from src.platform_submission import PlatformRunConfig, run_platform_pipeline

@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass

@cli.command()
def download_models():
    """Download required docling models."""
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Parsing PDFs (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization')
def serialize_tables(max_workers):
    """Serialize tables in parsed reports using parallel threading."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab', 'local_ser_tab', 'local_no_ser_tab', 'local_no_ser_tab_bm25', 'local_no_ser_tab_bm25_auto']), default='no_ser_tab', help='Configuration preset to use')
def process_reports(config):
    """Process parsed reports through the pipeline stages."""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking', 'gemini_free', 'openrouter_free', 'sambanova_free', 'groq_free', 'mistral_free', 'cerebras_free', 'cohere_free', 'local_free', 'local_free_bm25', 'local_free_bm25_auto']), default='base', help='Configuration preset to use')
def process_questions(config):
    """Process questions using the pipeline."""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()

@cli.command(name="platform-submit")
@click.option('--provider', default='openai', help='Answer-model provider')
@click.option('--model', default='gpt-4.1-mini', help='Answer-model name')
@click.option('--analysis-model', default=None, help='Optional LLM model for query analysis / routing')
@click.option('--free-text-model', default=None, help='Optional stronger model for free_text answers only')
@click.option(
    '--strategy',
    type=click.Choice(
        [
            'baseline',
            'adaptive_lexical',
            'dense_doc_diverse',
            'dense_doc_diverse_lexical',
            'late_interaction',
            'multi_query_expansion',
            'corrective_retrieval',
            'evidence_first',
            'citation_focus',
            'hybrid_mq_evidence',
            'hybrid_mq_late',
            'hybrid_prod_v1',
            'hybrid_prod_v2',
            'doc_profile_hybrid',
            'prod_auto_v1',
        ]
    ),
    default='dense_doc_diverse',
    help='Retrieval strategy preset',
)
@click.option('--top-k-chunks', default=5, help='How many reranked chunks to pass to the answer model')
@click.option(
    '--chunk-source',
    type=click.Choice(['standard', 'contextual']),
    default='standard',
    help='Chunk corpus variant used for retrieval and reranking',
)
@click.option('--work-dir', default='challenge_workdir', help='Directory for parsed/indexed challenge artifacts')
@click.option('--limit', default=0, help='Optional question limit for smoke runs')
@click.option('--reuse-downloads/--fresh-downloads', default=True, help='Reuse previously downloaded platform resources')
@click.option('--download-only', is_flag=True, help='Only download and prepare corpus, do not answer')
@click.option('--submit', is_flag=True, help='Submit the built submission to the platform')
def platform_submit(provider, model, analysis_model, free_text_model, strategy, top_k_chunks, chunk_source, work_dir, limit, reuse_downloads, download_only, submit):
    """Run the src DIFC pipeline against the Agentic Challenge API and optionally submit."""
    run_config = PlatformRunConfig(
        provider=provider,
        model=model,
        analysis_model=analysis_model,
        free_text_model=free_text_model,
        strategy=strategy,
        top_k_chunks=top_k_chunks,
        chunk_source=chunk_source,
        work_dir=work_dir,
        limit=limit,
        reuse_downloads=reuse_downloads,
        download_only=download_only,
        submit=submit,
    )
    result = run_platform_pipeline(run_config)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    cli()
